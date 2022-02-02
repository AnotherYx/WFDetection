import glob,os
import multiprocessing as mp
import pandas as pd 
from os.path import join, abspath, dirname, pardir
from os import mkdir


# Directories
BASE_DIR = abspath(join(dirname(__file__),pardir))


attacktrain = join(BASE_DIR, "data/attacktrain/")
splittrain = join(BASE_DIR, "data/splittrain/")
evaluation = join(BASE_DIR, "data/evaluation/")


def readtrace(fname):
    with open(fname, 'r') as f:
        tmp = f.readlines()
        trace = pd.Series(tmp).str.slice(0,-1).str.split('\t',expand = True).astype(float)
        trace.columns = ['timestamp','direction']
        trace.iloc[:,1] /= abs(trace.iloc[:,1]) 
    return trace 

def dump(trace, fdir):
    with open(fdir, 'w') as fo:
        for packet in trace:
            fo.write("{:.4f}".format(packet[0]) +'\t' + "{}".format(int(packet[1]))+ '\n')

def zippingFunc(param):
    fdir, target = param[0], param[1]
    ziptrace = []

    trace = readtrace(fdir)
    timestamp = trace.iloc[0,0]

    if trace.iloc[0,1] > 0:   
        direction = 1
    else:        
        direction = -1
    for i in range(1,len(trace)):
        if int(trace.iloc[i,1]) > 0 and direction > 0:  
            direction = direction + 1
        elif int(trace.iloc[i,1]) > 0 and direction < 0:
            ziptrace.append([timestamp, direction])
            timestamp = trace.iloc[i,0]
            direction = 1
        elif int(trace.iloc[i,1]) < 0 and direction > 0:
            ziptrace.append([timestamp, direction])
            timestamp = trace.iloc[i,0]
            direction = -1
        else:
            direction = direction - 1
        if i == len(trace)-1:
            ziptrace.append([timestamp, direction])

    with open(join(BASE_DIR, "data", target.split('/')[-2]+"_compress_rate.txt"), 'a') as f:
        rate = len(ziptrace)/len(trace)
        f.write("{:.4f}".format(rate) +'\n')    

    name = fdir.split('/')[-1]
    output_addr = join(BASE_DIR, "data/zipping_" + target.split('/')[-2], name)
    # print("Dumped to {}".format(output_addr))
    dump(ziptrace, output_addr)

def zippingParallel(flist, target, n_jobs = 20):
    pool = mp.Pool(n_jobs)
    params = zip(flist, [target]*len(flist))
    pool.map(zippingFunc, params)

def zipping(target):
    folder = join(BASE_DIR, "data/zipping_" + target.split('/')[-2])
    if not os.path.exists(folder):
        mkdir(folder)
    flist  = glob.glob(os.path.join(target,'*.cell'))
    #print(flist[:10])
    zippingParallel(flist, target)

if __name__ == "__main__":
    zipping(attacktrain)
    # zipping(evaluation)
    