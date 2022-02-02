import torch
import torch.nn as nn
import numpy as np

from cvpods.layers import ShapeSpec
from cvpods.modeling.backbone import build_resnet1d_backbone
from cvpods.configs import BaseConfig


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def create_model(n_cls,depth):

    _cfg = dict()
    input_shape = 0


    input_shape = None
    if input_shape is None:
        # input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
        input_shape = ShapeSpec(channels=1)
    
    _cfg = dict(
        MODEL=dict(
            RESNETS=dict({'ACTIVATION': {'INPLACE': True, 'NAME': 'ReLU'}, 
                        'NUM_CLASSES' : n_cls, 
                        'DEEP_STEM': False, 
                        'DEPTH': depth, 
                        'NORM': 'BN1d', 
                        'NUM_GROUPS': 1, 
                        'OUT_FEATURES': ['res5','linear'], 
                        'RES2_OUT_CHANNELS': 256,
                        'RES5_DILATION': 1, 
                        'STEM_OUT_CHANNELS': 64, 
                        'STRIDE_IN_1X1': True, 
                        'WIDTH_PER_GROUP': 64, 
                        'ZERO_INIT_RESIDUAL': False})))
    cfg = BaseConfig(_cfg)
    model = build_resnet1d_backbone(cfg, input_shape)
    print(model)

    return model

def mlp_fit(mlp, support_features, support_ys):
    optimizer = torch.optim.Adam(mlp.parameters(), lr = 0.01)
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        mlp = mlp.cuda()
        criterion = criterion.cuda()
    for epoch in range(1, 100 + 1):
        logits = mlp(support_features)
        loss = criterion(logits, support_ys)
        # acc1, acc5 = accuracy(logits, support_ys, topk=(1,5))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def mlp_predict(mlp, query_features, query_ys):
    if torch.cuda.is_available():
        mlp = mlp.cuda()
    with torch.no_grad():
        logits = mlp(query_features)
    acc1 = accuracy(logits, query_ys, topk=(1,))
    return acc1

def mlp_predict_open(mlp, query_features, query_ys):
    if torch.cuda.is_available():
        mlp = mlp.cuda()
    with torch.no_grad():
        logits = mlp(query_features)
    m = nn.Softmax(dim=1)
    logits = m(logits)
    
    logits = logits.detach().cpu().numpy()
    # print(logits.shape, type(logits))
    # predict_label = np.argmax(logits, axis=1)
    return logits
    # acc1 = accuracy(logits, query_ys, topk=(1,))
    # return acc1


class MLP(nn.Module):

    def __init__(self, num_classes=100, init_weights=True):
        super(MLP, self).__init__()
        # self.classifier = nn.Sequential(
        #     nn.Linear(256, 512), 
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Dropout(0.7),
        #     nn.Linear(512, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, num_classes)
        #     )
        self.classifier = nn.Sequential(
            nn.Linear(256, 512), 
            nn.Linear(512, num_classes)
            )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class CategoriesSampler():
    """The class to generate episodic data"""
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch
