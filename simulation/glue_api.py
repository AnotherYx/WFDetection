#Modified from: https://github.com/websitefingerprinting/WebsiteFingerprinting, created by WFDetection.
import sys
import os
import multiprocessing as mp
from os import mkdir
from os.path import join, abspath, dirname, pardir

import numpy as np
import pandas as pd
import json

import argparse
import logging
import datetime

BASE_DIR = abspath(join(dirname(__file__), pardir))
logger = logging.getLogger('glue')


def config_logger(args):
    # Set file
    log_file = sys.stdout
    if args.log != 'stdout':
        log_file = open(args.log, 'w')
    ch = logging.StreamHandler(log_file)

    # Set logging format
    LOG_FORMAT = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    ch.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(ch)

    # Set level format
    logger.setLevel(logging.INFO)
    

def parse_arguments():

    parser = argparse.ArgumentParser(description='It simulates adaptive padding on a set of web traffic traces.')

    parser.add_argument('traces_path',
                        metavar='<traces path>',
                        help='Path to the directory with the traffic traces to be simulated.')
    
    parser.add_argument('-listpath',
                        type=str,
                        metavar='<mergelist>',
                        help='Give an order of traces in a l-traces')
                        
    parser.add_argument('-noise',
                        type=str,
                        metavar='<pad noise>',
                        default= 'False',
                        help='Simulate whether pad glue noise or not')

    parser.add_argument('-glue',
                        type=str,
                        metavar='<pad front noise>',
                        default= 'False',
                        help='Simulate whether pad front noise or not')

    parser.add_argument('-forWFD',
                        type=str,
                        metavar='<only save .npy>',
                        default= 'False',
                        help='Only save .npy of directions, for training data generate WFD')

    parser.add_argument('-output',
                        type=str,
                        metavar='<output_dir>',
                        help='Output directory for l-traces')

    parser.add_argument('--log',
                        type=str,
                        dest="log",
                        metavar='<log path>',
                        default='stdout',
                        help='path to the log file. It will print to stdout by default.')

    args = parser.parse_args()
    #config = dict(conf_parser._sections[args.section])
    config_logger(args)

    return args


# '''used to save single traces'''
# global list_names

def load_trace(fname, t = 999, noise = False):
    '''load a trace from fpath/fname up to t time.'''
    '''return trace'''       
    pkts = []
    with open(fname, 'r') as f:
        for line in f:
            try:    
                timestamp, length = line.strip().split('\t')
                pkts.append([float(timestamp), int(length)])
                if float(timestamp) >= t+0.5:
                    break
            except ValueError:
                logger.warn("Could not split line: %s in %s", line, fname)
        return np.array(pkts)
        

def weibull(k = 0.75):
    return np.random.weibull(0.75)
def uniform():
    return np.random.uniform(1,10)


def simulate(trace):
    # logger.debug("Simulating trace {}".format(fdir))
    np.random.seed(datetime.datetime.now().microsecond)
    front_trace = RP(trace)
    return front_trace

def RP(trace):
    # format: [[time, pkt],[...]]
    # trace, cpkt_num, spkt_num, cwnd, swnd

    client_dummy_pkt_num = 1100
    server_dummy_pkt_num = 1100
    client_min_dummy_pkt_num = 1
    server_min_dummy_pkt_num = 1
    start_padding_time = 0
    max_wnd = 14
    min_wnd = 1
    
    client_wnd = np.random.uniform(min_wnd, max_wnd)
    server_wnd = np.random.uniform(min_wnd, max_wnd)
    if client_min_dummy_pkt_num != client_dummy_pkt_num:
        client_dummy_pkt = np.random.randint(client_min_dummy_pkt_num,client_dummy_pkt_num)
    else:
        client_dummy_pkt = client_dummy_pkt_num
    if server_min_dummy_pkt_num != server_dummy_pkt_num:
        server_dummy_pkt = np.random.randint(server_min_dummy_pkt_num,server_dummy_pkt_num)
    else:
        server_dummy_pkt = server_dummy_pkt_num
    logger.debug("client_wnd:",client_wnd)
    logger.debug("server_wnd:",server_wnd)
    logger.debug("client pkt:", client_dummy_pkt)
    logger.debug("server pkt:", server_dummy_pkt)


    first_incoming_pkt_time = trace[np.where(trace[:,1] <0)][0][0]
    #This is to find the last pkt time of first trace. We cant use last pkt time of the whole trace for MP
    last_pkt_time = trace[np.where(abs(trace[:,1]) == 1 )][-1][0]
    # print("last_pkt_time",last_pkt_time)    
    
    client_timetable = getTimestamps(client_wnd, client_dummy_pkt)
    client_timetable = client_timetable[np.where(start_padding_time+client_timetable[:,0] <= last_pkt_time)]

    server_timetable = getTimestamps(server_wnd, server_dummy_pkt)
    server_timetable[:,0] += first_incoming_pkt_time
    server_timetable =  server_timetable[np.where(start_padding_time+server_timetable[:,0] <= last_pkt_time)]

    # print("client_timetable")
    # print(client_timetable[:10])
    client_pkts = np.concatenate((client_timetable, 1*np.ones((len(client_timetable),1))),axis = 1)
    server_pkts = np.concatenate((server_timetable, -1*np.ones((len(server_timetable),1))),axis = 1)

    noisy_trace = np.concatenate( (trace, client_pkts, server_pkts), axis = 0)
    noisy_trace = noisy_trace[ noisy_trace[:, 0].argsort(kind = 'mergesort')]
    return noisy_trace

def getTimestamps(wnd, num):
    # timestamps = sorted(np.random.exponential(wnd/2.0, num))
    timestamps = sorted(np.random.rayleigh(wnd,num))
    # print(timestamps[:5])
    # timestamps = np.fromiter(map(lambda x: x if x <= wnd else wnd, timestamps),dtype = float)
    return np.reshape(timestamps, (len(timestamps),1))


def dump(trace, fpath):
    '''Write trace packet into file `fpath`.'''
    with open(fpath, 'w') as fo:
        for packet in trace:
            fo.write("{}".format(packet[0]) +'\t' + "{}".format(int(packet[1])) + '\n')
 

def merge(this, other, start, cnt = 1):
    '''t = 999, pad all pkts, otherwise pad up to t seconds'''
    other[:,0] -= other[0][0]
    other[:,0] += start
    other[:,1] *= cnt
    if this is None:
        this = other
    else:
        this = np.concatenate((this,other), axis = 0)
    return this
    

def est_iat(trace):
    trace_1 = np.concatenate((trace[1:], trace[0:1]),axis=0)
    itas = trace_1[:-1,0] - trace[:-1,0]
    return np.random.uniform(np.percentile(itas,20), np.percentile(itas,80))


def choose_site():
    with open(join(BASE_DIR,"data/nonsens.txt"),"r") as f:
        list_names = list(pd.Series(f.readlines()).str.slice(0,-1))
    noise_site = np.random.choice(list_names,1)[0]
    noise_site = join(BASE_DIR,noise_site)
    return noise_site


def compress_merge(dir_trace, monitored_margins, output_dir, mergeid):
    ziptrace = []
    ziptrace_monitored_margins = []
    for i in range(len(monitored_margins)):
        ziptrace_monitored_margins.append([0,0,monitored_margins[i][2]])
    if dir_trace[0] > 0:   
        direction = 1
    else:        
        direction = -1

    cur = 0
    for i in range(1,len(dir_trace)):

        if len(monitored_margins) > 0 and cur<=len(monitored_margins)-1:
            start = monitored_margins[cur][0]
            end = monitored_margins[cur][1]
            if i==start:
                ziptrace_monitored_margins[cur][0] = len(ziptrace)
            elif i==end:
                ziptrace_monitored_margins[cur][1] = len(ziptrace)+1
                cur+=1
        if dir_trace[i] > 0 and direction > 0:  #direction not change and still positive
            direction = direction + 1
        elif dir_trace[i] > 0 and direction < 0:
            ziptrace.append(direction)
            direction = 1
        elif dir_trace[i] < 0 and direction > 0:
            ziptrace.append(direction)
            direction = -1
        else:
            direction = direction - 1
        if i == len(dir_trace)-1:
            ziptrace.append(direction)
    
    compressed_merge_output_dir = join(output_dir, "brust_merge", mergeid + ".npy")
    ziptrace = np.array(ziptrace)
    np.save(compressed_merge_output_dir, ziptrace)

    brust_annotations = []
    for ziptrace_monitored_margin in ziptrace_monitored_margins:
        width = ziptrace_monitored_margin[1] - ziptrace_monitored_margin[0]
        annotation = {
            "id": 0,
            "area": width, 
            "iscrowd": 0,
            "trace_id": int(mergeid),
            "bbox": [ziptrace_monitored_margin[0], 0, width, 1],
            "category_id": ziptrace_monitored_margin[2],
        }     
        brust_annotations.append(annotation)   
    return brust_annotations, len(ziptrace)


def pick_annotations(mergedtrace, mergelist, output_dir, mergeid):
    annotations = []
    ann_index = []
    monitored_margins = []
    for i,tracename in enumerate(mergelist):
        if '-' in tracename:
            cat = int(tracename.split('/')[-1].split('.')[0].split('-')[0])
            ann_index.append([i+1,cat])
            if i==0:
                glue_cat=cat+100
            else:
                glue_cat=cat
            monitored_margins.append([0, 0, glue_cat])
    directions = mergedtrace[:,1]
    
    for i, index in enumerate(ann_index):
        dir_pos = np.where(directions == index[0])
        dir_neg = np.where(directions == -index[0])
        start = dir_pos[0][0] if dir_pos[0][0]<dir_neg[0][0] else dir_neg[0][0]
        end = dir_pos[0][-1] if dir_pos[0][-1]>dir_neg[0][-1] else dir_neg[0][-1]
        width = end-start
        annotation = {
            "id": 0,
            "area": int(width), #<class 'numpy.int64'> can not be read by json.dumps, convert to int.
            "iscrowd": 0,
            "trace_id": int(mergeid),
            "bbox": [int(start), 0, int(width), 1],
            "category_id": index[1],
        }
        annotations.append(annotation)
        monitored_margins[i][0]=start
        monitored_margins[i][1]=end
    brust_annotations, brust_mergelen = compress_merge(directions, monitored_margins, output_dir, mergeid)

    return annotations, brust_annotations, brust_mergelen


def MergePad2(output_dir, outputname ,noise, glue, forWFD, mergelist = None, waiting_time = 10):
    '''mergelist is a list of file names'''
    '''write in 2 files: the merged trace; the merged trace's name'''
    this = None
    start = 0.0 
    
    for cnt,fname in enumerate(mergelist):
        trace = load_trace(fname)
        this = merge(this, trace, start, cnt = cnt + 1)
        start = this[-1][0]
        '''pad noise or not'''
        if noise:
            noise_fname = choose_site()
            if cnt == len(mergelist)-1:
                ###This is a param in mergepadding###
                t = np.random.uniform(waiting_time, waiting_time+5)  
            else:
                t = uniform()
            small_time = est_iat(trace)
            logger.debug("Delta t is %.5f seconds"%(small_time))
            noise_site = load_trace(noise_fname, max(t - small_time, 0),True)
            this = merge(this, noise_site,start+small_time, cnt = 999)
            # logger.info("Dwell time is %.2f seconds"%(t))
            start = start + t
        else:
            t = uniform()
            start = start + t

    if noise:
        this = this[this[:,0].argsort(kind = "mergesort")]
    if glue:
        this = simulate(this)

    if not forWFD:
        dump(this, join(output_dir, outputname+'.merge'))
        logger.debug("Merged trace is dumpped to %s.merge"%join(output_dir, outputname))  

    annotations, brust_annotations, brust_mergelen = pick_annotations(this, mergelist, output_dir, outputname)
    mergelen = len(this)     
    
    return annotations, mergelen, brust_annotations, brust_mergelen   

    
def work(param):
    output_dir, cnt, noise, glue, forWFD, T = param[0],param[1], param[2], param[3], param[4], param[5]
    return MergePad2(output_dir,  str(cnt) , noise, glue, forWFD, T)


def parallel(output_dir, noise, glue, forWFD, mergedTrace, n_jobs = 20): 
    cnt = range(len(mergedTrace))
    l = len(cnt)

    param_dict = zip([output_dir]*l, cnt, [noise]*l, [glue]*l, [forWFD]*l, mergedTrace)
    pool = mp.Pool(n_jobs)
    merge_results = pool.map(work, param_dict)
    return merge_results

def save_annotationjson(output_dir, merge_results):
    traces_results = []
    brust_traces_results = []

    annotation_id = 0
    annotations_results = []
    brust_annotations_results = []

    for i, merge_result in enumerate(merge_results): #list of items, each item contains [annotations, mergelen, brust_annotations, brust_mergelen]   
        
        annotations = merge_result[0]
        brust_annotations = merge_result[2]
        assert len(annotations) == len(brust_annotations), "Error occurs in compressing mergedTrace, please check."

        for j in range(len(annotations)):
            annotations[j]["id"] = annotation_id
            brust_annotations[j]["id"] = annotation_id
            annotation_id += 1
            annotations_results.append(annotations[j])              
            brust_annotations_results.append(brust_annotations[j])       

        mergelen = merge_result[1]
        trace = {
            "id": i,
            "file_name": str(i)+'.merge',
            "width": mergelen,
            "height": 1
        }
        traces_results.append(trace)

                    
        brust_mergelen = merge_result[3]
        brust_trace = {
            "id": i,
            "file_name": str(i)+'.npy',
            "width": brust_mergelen,
            "height": 1
        }
        brust_traces_results.append(brust_trace)
  

    categories = []
    categories_num = 100
    for i in range(categories_num):
        categorie = {
            "id": i,
            "name": str(i),
            "supercategory": "none"
        }
        categories.append(categorie)

    brust_glue_categories = []
    brust_glue_categories_num = 200
    for i in range(brust_glue_categories_num):
        categorie = {
            "id": i,
            "name": str(i),
            "supercategory": "none"
        }
        brust_glue_categories.append(categorie)

    merge_dict = {
        "traces": traces_results,
        "annotations": annotations_results,
        "categories": categories
    }

    brust_merge_dict = {
        "traces": brust_traces_results,
        "annotations": brust_annotations_results,
        "categories": brust_glue_categories
    }   

    with open(join(output_dir, 'annotations.json'), 'w') as json_file:
        json.dump(merge_dict, json_file)

    with open(join(output_dir, 'annotations_brust.json'), 'w') as json_file:
        json.dump(brust_merge_dict, json_file)


if __name__ == '__main__':
    
    args = parse_arguments()
    logger.info("Arguments: %s" % (args))

    mergedTrace = []
    with open(args.listpath,'r') as f:
        tmp = f.readlines()
        for line in tmp:
            mergelist = line[:-2].split("\t")
            for i, item in enumerate(mergelist):
                mergelist[i] = join(args.traces_path, item)
            mergedTrace.append(np.array(mergelist))

    if not os.path.exists(join(args.output,"brust_merge")):
        mkdir(join(args.output,"brust_merge"))

    results = parallel(args.output, eval(args.noise), eval(args.glue), eval(args.forWFD), mergedTrace, 20)
    logger.info("Totally generate %d l-traces into directory: %s" %(len(mergedTrace), args.output))

    save_annotationjson(args.output, results)
    logger.info("Generate annotations.json.")



    

