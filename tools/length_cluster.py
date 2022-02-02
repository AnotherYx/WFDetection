import numpy as np
from os.path import join, abspath, dirname, pardir
import multiprocessing as mp
import glob
import random

LOG_FORMAT = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
BASE_DIR = abspath(join(dirname(__file__), pardir))


def trace_len(params):
    fdir = params[0]
    with open(fdir,'r') as f:
        tmp = f.readlines()
    return len(tmp)

def zipping(directions):
    ziptrace = []
    if directions[0] > 0:
        ziptrace.append(1)
    else:
        ziptrace.append(-1)
    for i in range(1,len(directions)):
        if directions[i] > 0 and ziptrace[-1] > 0: 
            ziptrace[-1] = ziptrace[-1] + 1
        elif directions[i] > 0 and ziptrace[-1] < 0:
            ziptrace.append(1)
        elif directions[i] < 0 and ziptrace[-1] > 0:
            ziptrace.append(-1)
        else:
            ziptrace[-1] = ziptrace[-1] - 1
    return ziptrace

def brust_trace_len(params,t=999):
    fdir = params[0]
    pkts = []
    with open(fdir,'r') as f:
        for line in f:
            try:    
                timestamp, direction = line.strip().split('\t')
                pkts.append(int(direction))
                if float(timestamp) >= t+0.5:
                    break
            except ValueError:
                exit()
    brust_trace = zipping(pkts)
    return len(brust_trace)

def brust_parallel(flist, n_jobs = 20):
    pool = mp.Pool(n_jobs)
    params = zip(flist)
    tracelens = pool.map(brust_trace_len, params)
    return tracelens


def parallel(flist, n_jobs = 20):
    pool = mp.Pool(n_jobs)
    params = zip(flist)
    tracelens = pool.map(trace_len, params)
    return tracelens


def distance(x, y):
    """
    input:
        x: shape=(n_samples, n_features)
        y: shape=(k, n_features)
    output:
        z: shape=(n_smaples, k)
    """
    # shape=(n_samples, k, n_features)
    z = np.expand_dims(x, axis=1) - y
    z = np.square(z)
    z = np.sqrt(np.sum(z, axis=2))
    return z


def k_means(data, k, max_iter):
    data = np.asarray(data, dtype=int)
    n_samples, n_features = data.shape
    # Random initialization cluster center
    indices = random.sample(range(n_samples), k)
    center = np.copy(data[indices])
    cluster = np.zeros(data.shape[0], dtype=int)
    i = 1
    while i <= max_iter:
        dis = distance(data, center)
        # New cluster of samples
        cluster = np.argmin(dis, axis=1)
        onehot = np.zeros(n_samples * k, dtype=int)
        onehot[cluster + np.arange(n_samples) * k] = 1.
        onehot = np.reshape(onehot, (n_samples, k))
        # The cluster center is averaged in the form of matrix multiplication
        # (n_samples, k)^T * (n_samples, n_features) = (k, n_features)
        new_center = np.matmul(np.transpose(onehot, (1, 0)), data)
        new_center = new_center / np.expand_dims(np.sum(onehot, axis=0), axis=1)
        center = new_center
        i += 1
    cluster = np.asarray(cluster, dtype=int)
    center = np.asarray(center, dtype=int)
    return cluster, center


if __name__ == "__main__":
    """
    just an example of how to use our code.
    You may want to write your own script with your datasets and other customizations.
    """
    traces_path = join(BASE_DIR, "data/ds19/")
    list_all = glob.glob(join(traces_path, '*'))
    num = len(list_all)
    tracelens = parallel(list_all)
    total = 0
    for l in tracelens:
        total+=l
    averagelen = total/num
    print("average length of single trace: %d"%averagelen)

    listpath = join(BASE_DIR, 'simulation/results/noise_overlap/noise_overlap_all','list')
    tracenum = 0
    with open(listpath,'r') as f:
        tmp = f.readlines()
        for line in tmp:
            mergelist = line[:-2].split("\t")
            tracenum+=len(mergelist)

    print("cells of hole overlap: %d"%(tracenum*averagelen))


    traces_path = join(BASE_DIR, "data/ds19/")
    list_sensitive = glob.glob(join(traces_path, '*-*'))
    num = len(list_sensitive)
    tracelens = parallel(list_sensitive)
    tracelens = np.array(tracelens)
    
    data = np.ones((num,2), dtype=int)
    data[:,0] = tracelens

    k = 10
    max_iter=5000
    cluster, center = k_means(data, k, max_iter)
    
    center = center[:,0].tolist()
    center.sort()
    print("Cluster centers of single-tarce(cell): ",center)

    data = data[data[:,0].argsort()]
    print("Tracelen range: [%d - %d] \n"%(data[0][0],data[-1][0]))
    #just for example
