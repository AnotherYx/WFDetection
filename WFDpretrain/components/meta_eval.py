import numpy as np
import scipy
from scipy.stats import t
from tqdm import tqdm

import torch
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .util import mlp_fit, mlp_predict, MLP, accuracy


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h


def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out



def meta_test(net, testloader, use_logit=True, is_norm=True, classifier='LR', opt=None):
    net = net.eval()
    acc = []

     # Generate labels
    label = torch.arange(opt.n_ways).repeat(opt.n_queries)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
    else:
        label = label.type(torch.LongTensor)
    label_shot = torch.arange(opt.n_ways).repeat(opt.n_shots)
    if torch.cuda.is_available():
        label_shot = label_shot.type(torch.cuda.LongTensor)
    else:
        label_shot = label_shot.type(torch.LongTensor)


    for idx, data in tqdm(enumerate(testloader)):

        if torch.cuda.is_available():
            data, _ = [_.cuda() for _ in data]
        else:
            print("cuda is unavailable")

        # data = data.unsqueeze(dim=1)
        k = opt.n_ways * opt.n_shots
        support_xs, query_xs = data[:k], data[k:]
        support_ys, query_ys = label_shot, label
        with torch.no_grad():
            if use_logit:
                support_features = net(support_xs)
                support_features = support_features['linear'].view(support_xs.size(0), -1)
                query_features = net(query_xs)
                query_features = query_features['linear'].view(query_xs.size(0), -1)
            else:
                # feat_support, _ = net(support_xs, is_feat=True)
                support_features = net(support_xs)["res5"]
                support_features = support_features.view(support_xs.size(0), -1)
                # feat_query, _ = net(query_xs, is_feat=True)
                query_features = net(query_xs)["res5"]
                query_features = query_features.view(query_xs.size(0), -1)

        if is_norm:
            support_features = normalize(support_features)
            query_features = normalize(query_features)
        None_acc = 0
        #My triditional
        if classifier == 'MLP' and use_logit == False:
            mlp = MLP(num_classes = opt.n_ways)
            mlp_fit(mlp, support_features, support_ys)
            mpl_acc = mlp_predict(mlp, query_features,query_ys)
            mpl_acc = [i.detach().cpu().numpy() for i in mpl_acc]
        elif classifier == "None" and use_logit == True:
            None_acc = accuracy(query_features, query_ys, topk=(1,))
            None_acc = [i.detach().cpu().numpy() for i in None_acc]
        else:
            mpl_acc = 0

        support_features = support_features.detach().cpu().numpy()
        query_features = query_features.detach().cpu().numpy()
        support_ys = support_ys.detach().cpu().numpy()
        query_ys = query_ys.detach().cpu().numpy()

        if classifier == 'LR':
            clf = LogisticRegression(penalty='l2',
                                        random_state=0,
                                        C=1.0,
                                        solver='lbfgs',
                                        max_iter=1000,
                                        multi_class='multinomial')
            clf.fit(support_features, support_ys)
            query_ys_pred = clf.predict(query_features)
        elif classifier == 'SVM':
            clf = make_pipeline(StandardScaler(), SVC(gamma='auto',
                                                        C=1,
                                                        kernel='linear',
                                                        decision_function_shape='ovr'))
            clf.fit(support_features, support_ys)
            query_ys_pred = clf.predict(query_features)
        elif classifier == 'NN':
            query_ys_pred = NN(support_features, support_ys, query_features)
        elif classifier == 'Cosine':
            query_ys_pred = Cosine(support_features, support_ys, query_features)
        elif classifier == 'Proto':
            query_ys_pred = Proto(support_features, support_ys, query_features, opt)

        if classifier != 'MLP' and classifier != 'None':
            acc.append(metrics.accuracy_score(query_ys, query_ys_pred))
        elif classifier == "None":
            acc.append(None_acc)
        else:
            acc.append(mpl_acc)

    return mean_confidence_interval(acc)
    

def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        exp_minmax = lambda x: np.exp(x - np.max(x))
        denom = lambda x: 1.0 / np.sum(x)
        x = np.apply_along_axis(exp_minmax,1,x)
        denominator = np.apply_along_axis(denom,1,x) 
        
        if len(denominator.shape) == 1:
            denominator = denominator.reshape((denominator.shape[0],1))
        
        x = x * denominator
    else:
        # Vector
        x_max = np.max(x)
        x = x - x_max
        numerator = np.exp(x)
        denominator =  1.0 / np.sum(numerator)
        x = numerator.dot(denominator)
    
    assert x.shape == orig_shape
    return x

def Proto(support, support_ys, query, opt):
    """Protonet classifier"""
    nc = support.shape[-1]
    support = np.reshape(support, (-1, 1, opt.n_ways, opt.n_shots, nc))
    support = support.mean(axis=3)
    batch_size = support.shape[0]
    query = np.reshape(query, (batch_size, -1, 1, nc))
    logits = - ((query - support)**2).sum(-1)
    pred = softmax(logits)
    # pred = np.argmax(logits, axis=-1)
    # pred = np.reshape(pred, (-1,))
    return pred


def NN(support, support_ys, query):
    """nearest classifier"""
    support = np.expand_dims(support.transpose(), 0)
    query = np.expand_dims(query, 2)

    diff = np.multiply(query - support, query - support)
    distance = diff.sum(1)

    max_distance = np.max(distance, axis = 1)
    max_distance = np.expand_dims(max_distance, 1)
    max_distance = np.broadcast_to(max_distance, distance.shape)
    logits = np.abs(distance - max_distance)
    pred = softmax(logits)
    # min_idx = np.argmin(distance, axis=1)
    # pred = [support_ys[idx] for idx in min_idx]
    return pred


def Cosine(support, support_ys, query):
    """Cosine classifier"""
    support_norm = np.linalg.norm(support, axis=1, keepdims=True)
    support = support / support_norm
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    query = query / query_norm

    cosine_distance = query @ support.transpose()
    pred = softmax(cosine_distance)
    # max_idx = np.argmax(cosine_distance, axis=1)
    # pred = [support_ys[idx] for idx in max_idx]
    return pred
