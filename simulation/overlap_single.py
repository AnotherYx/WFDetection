import glob,os
import multiprocessing as mp
import pandas as pd 
import numpy as np
import random
from os.path import join, abspath, dirname, pardir
import argparse
import logging
import sys

# Directories
BASE_DIR = abspath(join(dirname(__file__), pardir))
logger = logging.getLogger('single-overlap')

attacktrain = join(BASE_DIR, "data/attacktrain/")
splittrain = join(BASE_DIR, "data/splittrain/")
evaluation = join(BASE_DIR, "data/evaluation/")

# zipping_attacktrain = join(BASE_DIR, "data/zipping_attacktrain/")
# zipping_evaluation = join(BASE_DIR, "data/zipping_evaluation/")

def config_logger(args):
    # Set file
    log_file = sys.stdout
    if args.log != 'stdout':
        log_file = open(args.log, 'w')
    ch = logging.StreamHandler(log_file)

    LOG_FORMAT = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    # Set logging format
    ch.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(ch)

    # Set level format
    logger.setLevel(logging.INFO)
    

def parse_arguments():
    parser = argparse.ArgumentParser(description='It overlap web traffic traces with noise.')

    parser.add_argument('traces_path',
                        metavar='<traces path>',
                        help='Path to the directory with the traffic traces to be overlaped.')

    parser.add_argument('-stored_type',
                        type=str,
                        metavar='<final stored type>',
                        default= 'clean')

    parser.add_argument('--log',
                        type=str,
                        dest="log",
                        metavar='<log path>',
                        default='stdout',
                        help='path to the log file. It will print to stdout by default.')

    args = parser.parse_args()
    config_logger(args)

    return args


def load_trace(fdir):
    with open(fdir,'r') as f:
        tmp = f.readlines()
    t = pd.Series(tmp).str.slice(0,-1).str.split('\t',expand = True).astype('float')
    return np.array(t)

def dump(trace, fdir):
    with open(fdir, 'w') as fo:
        for packet in trace:
            fo.write("{:.4f}".format(packet[0]) +'\t' + "{}".format(int(packet[1]))\
                + '\n')

def mixFunc(trace1,trace2,position):
    timegap = trace1[-1][0]-trace1[0][0]
    if position == "head" or position == "both_h":
        curtime = trace2[-1][0] - timegap
        for i,packet in enumerate(reversed(trace2)):
            if packet[0] <= curtime:
                pick = trace2[len(trace2)-1-i:]
                pick[:,0] -= pick[0][0]
                result = np.concatenate((pick,trace1), axis = 0)
                result = result[result[:,0].argsort()]
                return result
        pick = trace2
        pick[:,0] -= pick[0][0]
        result = np.concatenate((pick,trace1), axis = 0)
        result = result[result[:,0].argsort()]
        return result        
          
    if position == "tail" or position == "both_t":
        curtime = timegap
        for i,packet in enumerate(trace2):
            if packet[0] >= curtime:
                pick = trace2[:i]
                pick[:,0] += trace1[0][0]
                result = np.concatenate((pick,trace1), axis = 0)
                result = result[result[:,0].argsort()]
                return result 
        pick = trace2
        pick[:,0] += trace1[0][0]
        result = np.concatenate((pick,trace1), axis = 0)
        result = result[result[:,0].argsort()]
        return result 

def OverlapFunc(param):
    global ratios,positions
    fdir, target, stored_types = param[0], param[1], param[2]
    trace = load_trace(fdir)
    # if len(trace) < 50:
    #     return False
    name = fdir.split('/')[-1]

    flist1 = glob.glob(os.path.join(target,'*.cell'))
    flist2 = glob.glob(os.path.join(target,'*-*.cell'))
    flist_um  = list(set(flist1)-set(flist2))

    for ratio in ratios:    
        for pos in positions:
            while 1:
                noisecell1 = random.choice(flist_um)
                noisetrace1 = load_trace(noisecell1)
                if len(noisetrace1)>100:
                    break

            if pos == "head":
                cleantrace = trace[round(len(trace)*ratio):]
                dirtytrace = mixFunc(trace[:round(len(trace)*ratio)], noisetrace1, pos)
                fulltrace = np.concatenate((dirtytrace, cleantrace), axis = 0)
            elif pos == "tail":
                cleantrace = trace[:len(trace)-round(len(trace)*ratio)]
                dirtytrace = mixFunc(trace[len(trace)-round(len(trace)*ratio):], noisetrace1, pos)
                fulltrace = np.concatenate((cleantrace, dirtytrace), axis = 0)
            else:
                cleantrace = trace[round(len(trace)*ratio/2):len(trace)-round(len(trace)*ratio/2)]
                dirtytrace1 = mixFunc(trace[:round(len(trace)*ratio/2)], noisetrace1, pos+"_h")
                while 1:
                    noisecell2 = random.choice(flist_um)
                    noisetrace2 = load_trace(noisecell2)
                    if len(noisetrace2)>100:
                        break
                dirtytrace2 = mixFunc(trace[len(trace)-round(len(trace)*ratio/2):], noisetrace2, pos+"_t")
                fulltrace = np.concatenate((dirtytrace1, cleantrace, dirtytrace2), axis = 0)

            fatherpath = join(BASE_DIR,"data/overlap_" + target.split('/')[-2])
            childpath  = pos[0] + '_overlap_' +  str(int(ratio*10))
            
            if "clean" in stored_types:   
                clean_addr = join(fatherpath + "_clean_" + childpath, name) 
                dump(cleantrace, clean_addr)
            elif "dirty" in stored_types and pos != "both":
                dirty_addr = join(fatherpath + "_dirty_" + childpath, name)
                dump(dirtytrace, dirty_addr)
            elif "full" in stored_types:
                full_addr = join(fatherpath + "_full_" + childpath, name)
                dump(fulltrace, full_addr)  

                    

def parallel(flist, target, stored_types, n_jobs = 20):
    pool = mp.Pool(n_jobs)
    params = zip(flist, [target]*len(flist), [stored_types]*len(flist))
    pool.map(OverlapFunc, params)

def overlap(target, stored_types):
    global ratios,positions
    for ratio in ratios:
        for pos in positions:
            if "clean" in stored_types:
                folder_clean = join(BASE_DIR, "data/overlap_" + target.split('/')[-2] + "_clean_" \
                    + pos[0] + '_overlap_' + str(int(ratio*10)))
                if not os.path.exists(folder_clean):
                    os.makedirs(folder_clean)
            elif "dirty" in stored_types and pos != "both":
                folder_dirty = join(BASE_DIR, "data/overlap_" + target.split('/')[-2] + "_dirty_" \
                    + pos[0] + '_overlap_' + str(int(ratio*10)))
                if not os.path.exists(folder_dirty):
                    os.makedirs(folder_dirty)
            elif "full" in stored_types:
                folder_full = join(BASE_DIR, "data/overlap_" + target.split('/')[-2] + "_full_" \
                    + pos[0] + '_overlap_' +  str(int(ratio*10)))
                if not os.path.exists(folder_full):
                    os.makedirs(folder_full)

    flist  = glob.glob(os.path.join(target,'*.cell'))
    print(len(flist))
    parallel(flist, target, stored_types)

if __name__ == "__main__":
    args = parse_arguments()
    logger.info("Arguments: %s" % (args))

    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    positions = ["head", "tail", "both"]

    stored_init = []
    stored_init.append(args.stored_type)
    stored_types = list(set(stored_init))
    overlap(args.traces_path, stored_types)

