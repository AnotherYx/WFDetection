import random
import sys
import os
import multiprocessing as mp
from os import mkdir
from os.path import join

import numpy as np
import json
import argparse
import logging

from math import floor
logger = logging.getLogger('overlap')


def config_logger(args):
    # Set file
    log_file = sys.stdout
    if args.log != 'stdout':
        log_file = open(args.log, 'w')
    ch = logging.StreamHandler(log_file)

    LOG_FORMAT = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    ch.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(ch)

    # Set level format
    logger.setLevel(logging.INFO)
    

def parse_arguments():
    parser = argparse.ArgumentParser(description='It simulates overlaping on a set of web traffic traces.')

    parser.add_argument('traces_path',
                        metavar='<traces path>',
                        help='Path to the directory with the traffic traces to be simulated.')
    
    parser.add_argument('-listpath',
                        type=str,
                        metavar='<mergelist>',
                        help='Give an order of traces in a l-traces')
                        
    parser.add_argument('-output',
                        type=str,
                        metavar='<output_dir>',
                        default= 'False',
                        help='Output directory for l-traces')

    parser.add_argument('-random',
                        type=str,
                        metavar='<random overlap>',
                        default= 'False',
                        help='randomly overlap or not')

    parser.add_argument('-position',
                        type=str,
                        metavar='<overlap position>',
                        default= 'head')

    parser.add_argument('-ratio',
                        type=str,
                        metavar='<overlap ratio>',
                        default= '0.1')

    parser.add_argument('--log',
                        type=str,
                        dest="log",
                        metavar='<log path>',
                        default='stdout',
                        help='path to the log file. It will print to stdout by default.')

    args = parser.parse_args()

    config_logger(args)

    return args


def load_trace(fname, t = 999):
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


def dump(trace, fpath):
    '''Write trace packet into file `fpath`.'''
    with open(fpath, 'w') as fo:
        for packet in trace:
            fo.write("{}".format(packet[0]) +'\t' + "{}".format(int(packet[1])) + '\n')


def merge(this, other):
    if this is None and other is None:
        this = None
    elif this is None:
        this = other
    elif other is None:
        this = this
    else:
        this = np.concatenate((this,other), axis = 0)
    return this
    
def compress_merge(dir_trace, monitored_margins, monitored_full_margins, output_dir, mergeid):
    ziptrace = []
    ziptrace_monitored_margins = []
    ziptrace_monitored_full_margins = []
    for i in range(len(monitored_margins)):
        ziptrace_monitored_margins.append([0,0,monitored_margins[i][2]])
        ziptrace_monitored_full_margins.append([0,0,monitored_full_margins[i][2]])
    if dir_trace[0] > 0:   
        direction = 1
    else:        
        direction = -1

    cur = 0
    cur_full =0
    for i in range(1,len(dir_trace)):

        if len(monitored_margins) > 0 and cur<=len(monitored_margins)-1:
            start = monitored_margins[cur][0]
            end = monitored_margins[cur][1]
            if i==start:
                ziptrace_monitored_margins[cur][0] = len(ziptrace)
            elif i==end:
                ziptrace_monitored_margins[cur][1] = len(ziptrace)+1
                cur+=1

        if len(monitored_full_margins) > 0 and cur_full<=len(monitored_full_margins)-1:
            start_full = monitored_full_margins[cur_full][0]
            end_full = monitored_full_margins[cur_full][1]
            if i==start_full:
                ziptrace_monitored_full_margins[cur_full][0] = len(ziptrace)
            elif i==end_full:
                ziptrace_monitored_full_margins[cur_full][1] = len(ziptrace)+1
                cur_full+=1

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

    brust_annotations_full = []
    for ziptrace_monitored_full_margin in ziptrace_monitored_full_margins:
        width_full = ziptrace_monitored_full_margin[1] - ziptrace_monitored_full_margin[0]
        annotation_full = {
            "id": 0,
            "area": width_full, 
            "iscrowd": 0,
            "trace_id": int(mergeid),
            "bbox": [ziptrace_monitored_full_margin[0], 0, width_full, 1],
            "category_id": ziptrace_monitored_full_margin[2],
        }     
        brust_annotations_full.append(annotation_full)   

    return brust_annotations, brust_annotations_full, len(ziptrace)


def pick_annotations(mergedtrace, mergelist, position, output_dir, mergeid):
    annotations = []
    annotations_full = []
    ann_index = []
    monitored_margins = []
    monitored_full_margins = []
    for i,tracename in enumerate(mergelist):
        if '-' in tracename:
            cat = int(tracename.split('/')[-1].split('.')[0].split('-')[0])
            ann_index.append([i+1,cat])
            monitored_margins.append([0, 0, cat])
            monitored_full_margins.append([0, 0, cat])
    directions = mergedtrace[:,1]
    
    for i, index in enumerate(ann_index):
        dir_pos = np.where(directions == index[0])
        dir_neg = np.where(directions == -index[0])

        if position == "head":
            end = dir_pos[0][-1] if dir_pos[0][-1]>dir_neg[0][-1] else dir_neg[0][-1]
            if (index[0]-1) > 0:
                pre_dir_pos = np.where(directions == (index[0]-1))
                pre_dir_neg = np.where(directions == -(index[0]-1))
                pre_dir_end = pre_dir_pos[0][-1] if pre_dir_pos[0][-1]>pre_dir_neg[0][-1] else pre_dir_neg[0][-1]
                start = pre_dir_end+1
            else:
                start = 0 #monitored site is the first one in merge.
        elif position == "tail":
            start = dir_pos[0][0] if dir_pos[0][0]<dir_neg[0][0] else dir_neg[0][0]
            if (index[0]+1) <= len(mergelist):
                next_dir_pos = np.where(directions == (index[0]+1))
                next_dir_neg = np.where(directions == -(index[0]+1))
                next_dir_start = next_dir_pos[0][0] if next_dir_pos[0][0]<next_dir_neg[0][0] else next_dir_neg[0][0]
                end = next_dir_start-1
            else:
                end = len(directions)-1 #monitored site is the last one in merge.
        else:
            if (index[0]-1) > 0:
                pre_dir_pos = np.where(directions == (index[0]-1))
                pre_dir_neg = np.where(directions == -(index[0]-1))
                pre_dir_end = pre_dir_pos[0][-1] if pre_dir_pos[0][-1]>pre_dir_neg[0][-1] else pre_dir_neg[0][-1]
                start = pre_dir_end+1
            else:
                start = 0 #monitored site is the first one in merge.
            if (index[0]+1) <= len(mergelist):
                next_dir_pos = np.where(directions == (index[0]+1))
                next_dir_neg = np.where(directions == -(index[0]+1))
                next_dir_start = next_dir_pos[0][0] if next_dir_pos[0][0]<next_dir_neg[0][0] else next_dir_neg[0][0]
                end = next_dir_start-1
            else:
                end = len(directions)-1 #monitored site is the last one in merge.

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
        
        start_full = dir_pos[0][0] if dir_pos[0][0]<dir_neg[0][0] else dir_neg[0][0]
        end_full = dir_pos[0][-1] if dir_pos[0][-1]>dir_neg[0][-1] else dir_neg[0][-1]

        width_full = end_full-start_full
        annotation_full = {
            "id": 0,
            "area": int(width_full), #<class 'numpy.int64'> can not be read by json.dumps, convert to int.
            "iscrowd": 0,
            "trace_id": int(mergeid),
            "bbox": [int(start_full), 0, int(width_full), 1],
            "category_id": index[1],
        }
        annotations_full.append(annotation_full)
        monitored_full_margins[i][0]=start_full
        monitored_full_margins[i][1]=end_full
        # if start>end:   #a bug of this code, maybe caused by traces which last for a long time,dump them for analyse.
        #     dump(mergedtrace, join(output_dir, mergeid+'.merge'))
    brust_annotations, brust_annotations_full, brust_mergelen = compress_merge(directions, monitored_margins, monitored_full_margins, output_dir, mergeid)

    return annotations, annotations_full, brust_annotations, brust_annotations_full, brust_mergelen


def MergePad_head(output_dir, ratio, outputname, mergelist = None, forWFD = False):
    '''mergelist is a list of file names'''

    trace_list = []
    for cnt, fname in enumerate(mergelist):
        tmp = load_trace(fname)
        tmp[:,1] *= (cnt+1)
        if "-" in fname:
            monitored = True
        else:
            monitored = False
        trace_list.append(dict(trace=tmp,length=len(tmp),mon=monitored))

    this = None
    start = 0.0 
    overlaped = 0
    for cnt in range(1,len(mergelist)):
        trace_left = trace_list[cnt-1]["trace"][overlaped:]
        trace_left_monitored = trace_list[cnt-1]["mon"]
        trace_left[:,0] -= trace_left[0][0] 
        trace_left[:,0] += start
        trace_left_length = len(trace_left)
        trace_left_origin = trace_list[cnt-1]["length"]

        trace_right = trace_list[cnt]["trace"]
        trace_right_monitored = trace_list[cnt]["mon"]
        right_forOverlap = trace_right[:round(len(trace_right)*ratio)]
        timegap = right_forOverlap[-1][0]-right_forOverlap[0][0]
        

        re_calculate = False
        pick_left = pick = None # pick_left + pick = trace_left 

        if not trace_right_monitored:
            clean_threshold = 0.3 # Usually,we make sure at least 30% of the previous traceis left to be clean.
        else:
            clean_threshold = 0.2 # if the right trace is monitored, loosen the threshold.

        if trace_left_monitored: #can only overlap monitored website'trace on it's head.
            pick_left = trace_left 
            pick = None 
            re_calculate = True 
        elif (trace_left_length)/trace_left_origin <= clean_threshold: 
            pick_left = trace_left 
            pick = None 
            re_calculate = True 
        else:
            cur_time = trace_left[-1][0] - timegap
            for i,packet in enumerate(reversed(trace_left)):
                if packet[0] <= cur_time:
                    if (trace_left_length - i)/trace_left_origin >= clean_threshold:  # make sure we left at least 40% clean segment for the previous trace.
                        pick_left = trace_left[:trace_left_length-i]
                        pick = trace_left[trace_left_length-i:]
                    break
            if pick is None and pick_left is None:
                pick_left = trace_left[:floor(trace_left_origin*clean_threshold)]  
                pick = trace_left[floor(trace_left_origin*clean_threshold):]  
                re_calculate = True       

        if re_calculate:#need to re_calculate right_forOverlap.
            if pick is None:
                right_forOverlap = None
                overlaped = 0
            else:
                real_timegap = pick[-1][0]-pick[0][0]
                for i,packet in enumerate(trace_right):
                    if packet[0] >= real_timegap:
                        right_forOverlap = trace_right[:i]
                        break

        this = merge(this, pick_left)
        
        if right_forOverlap is not None:
            right_forOverlap[:,0] += pick[0][0]
            overlaped = len(right_forOverlap)
            start = right_forOverlap[-1][0]
        else:
            start = pick_left[-1][0]
        overlap_seg = merge(pick, right_forOverlap)
        this = merge(this, overlap_seg) #this = this + pick_left + overlap(pick + right_forOverlap)


    last_trace_left = trace_list[-1]["trace"][overlaped:]
    last_trace_left[:,0] -= last_trace_left[0][0] 
    last_trace_left[:,0] += start
    this = merge(this, last_trace_left)
    this = this[this[:,0].argsort()]

    if not forWFD:
        dump(this, join(output_dir, outputname+'.merge'))
        logger.debug("Merged trace is dumpped to %s.merge"%join(output_dir, outputname))   
    
    annotations, annotations_full, brust_annotations, brust_annotations_full, brust_mergelen = pick_annotations(this, mergelist, "head", output_dir, outputname)
    mergelen = len(this)  

    return annotations, annotations_full, mergelen, brust_annotations, brust_annotations_full, brust_mergelen           


def MergePad_tail(output_dir, ratio, outputname, mergelist = None, forWFD = False):
    '''mergelist is a list of file names'''

    trace_list = []
    for cnt, fname in enumerate(mergelist):
        tmp = load_trace(fname)
        tmp[:,1] *= (cnt+1)
        if "-" in fname:
            monitored = True
        else:
            monitored = False
        trace_list.append(dict(trace=tmp,length=len(tmp),mon=monitored))

    this = None
    start = 0.0 
    overlaped = 0
    for cnt in range(1,len(mergelist)):
        trace_left = trace_list[cnt-1]["trace"][overlaped:]
        trace_left_monitored = trace_list[cnt-1]["mon"]
        trace_left[:,0] -= trace_left[0][0] 
        trace_left[:,0] += start
        trace_left_length = len(trace_left)
        trace_left_origin = trace_list[cnt-1]["length"]  

        # left_forclean + left_forOverlap = trace_left
        left_forclean = trace_left[:trace_left_length-floor(trace_left_origin*ratio)]  
        left_forOverlap = trace_left[trace_left_length-floor(trace_left_origin*ratio):]   
           
        timegap = left_forOverlap[-1][0]-left_forOverlap[0][0]

        trace_right = trace_list[cnt]["trace"]
        trace_right_monitored = trace_list[cnt]["mon"]
        trace_right_origin = trace_list[cnt]["length"]       
        
        pick = None # pick + pick_right = trace_right 

        if not trace_left_monitored:
            clean_threshold = 0.3 # Usually,we make sure at least 30% of the previous traceis left to be clean.
        else:
            clean_threshold = 0.2 # if the right trace is monitored, loosen the threshold.

        if trace_right_monitored: #can only overlap monitored website'trace on it's tail.
            pick = None 
            overlaped = 0
            left_forOverlap = None   
            left_forclean = trace_left
        else:
            for i,packet in enumerate(trace_right):
                if packet[0] >= timegap:
                    if (trace_right_origin - i)/trace_right_origin >= (clean_threshold+ratio): 
                        pick = trace_right[:i]
                        overlaped = len(pick)                        
                    break
            if pick is None:
                pick = trace_right[:floor(trace_right_origin*(1-clean_threshold-ratio))] 
                overlaped = len(pick)    
                real_timegap = pick[-1][0]

                curtime = trace_left[-1][0] - real_timegap
                for i,packet in enumerate(reversed(trace_left)):
                    if packet[0] <= curtime:
                        left_forclean = trace_left[:trace_left_length-i]
                        left_forOverlap = trace_left[trace_left_length-i:]
                        break

        if left_forOverlap is not None: # which means pick is not None
            pick[:,0] += left_forOverlap[0][0]            
            start = pick[-1][0]
        else:
            start = trace_left[-1][0]

        this = merge(this, left_forclean)
        overlap_seg = merge(left_forOverlap, pick)
        this = merge(this, overlap_seg) 

    last_trace_left = trace_list[-1]["trace"][overlaped:]
    last_trace_left[:,0] -= last_trace_left[0][0] 
    last_trace_left[:,0] += start
    this = merge(this, last_trace_left)
    this = this[this[:,0].argsort()]

    if not forWFD:
        dump(this, join(output_dir, outputname+'.merge'))
        logger.debug("Merged trace is dumpped to %s.merge"%join(output_dir, outputname))  
    
    annotations, annotations_full, brust_annotations, brust_annotations_full, brust_mergelen = pick_annotations(this, mergelist, "tail", output_dir, outputname)
    mergelen = len(this)  

    return annotations, annotations_full, mergelen, brust_annotations, brust_annotations_full, brust_mergelen  


def MergePad_both(output_dir, ratio, outputname, mergelist = None, forWFD = False):
    '''mergelist is a list of file names'''
    trace_list = []
    for cnt, fname in enumerate(mergelist):
        tmp = load_trace(fname)
        tmp[:,1] *= (cnt+1)
        if "-" in fname:
            monitored = True
        else:
            monitored = False
        trace_list.append(dict(trace=tmp,length=len(tmp),mon=monitored))
    
    this = None
    start = 0.0 
    overlaped = 0
    
    for cnt in range(1,len(mergelist)):
        trace_left = trace_list[cnt-1]["trace"][overlaped:]
        trace_left_monitored = trace_list[cnt-1]["mon"]
        trace_left[:,0] -= trace_left[0][0] 
        trace_left[:,0] += start
        trace_left_length = len(trace_left)
        trace_left_origin = trace_list[cnt-1]["length"]  

        # left_forclean + left_forOverlap = trace_left
        left_forclean = trace_left[:trace_left_length-floor(trace_left_origin*ratio/2)]  
        left_forOverlap = trace_left[trace_left_length-floor(trace_left_origin*ratio/2):]

        trace_right = trace_list[cnt]["trace"]
        trace_right_monitored = trace_list[cnt]["mon"]
        trace_right_origin = trace_list[cnt]["length"]    

        # right_forOverlap + right_forclean = trace_right
        right_forOverlap = trace_right[:floor(trace_right_origin*ratio/2)]  

        clean_threshold = 0.3
        if trace_left_monitored and not trace_right_monitored:
            clean_threshold = 0.2
            right_forOverlap = None
            timegap = left_forOverlap[-1][0] -left_forOverlap[0][0]
            curtime = timegap
            for i,packet in enumerate(trace_right):
                if packet[0] >= curtime:
                    if (trace_right_origin - i)/trace_right_origin >= (clean_threshold+ratio/2): 
                        right_forOverlap = trace_right[:i]              
                    break
            if right_forOverlap is None:
                right_forOverlap = trace_right[:floor(trace_right_origin*(1-clean_threshold-ratio/2))] 
                real_timegap = right_forOverlap[-1][0]
                curtime = trace_left[-1][0] - real_timegap
                for i,packet in enumerate(reversed(trace_left)):
                    if packet[0] <= curtime:
                        left_forclean = trace_left[:trace_left_length-i]
                        left_forOverlap = trace_left[trace_left_length-i:]
                        break            
        elif trace_right_monitored and not trace_left_monitored:
            clean_threshold = 0.2
            left_forOverlap = None
            timegap = right_forOverlap[-1][0]
            curtime = trace_left[-1][0] - timegap
            for i,packet in enumerate(reversed(trace_left)):
                if packet[0] <= curtime:
                    if (trace_left_length - i)/trace_left_origin >= clean_threshold: 
                        left_forclean = trace_left[:trace_left_length - i]
                        left_forOverlap = trace_left[trace_left_length - i:]                       
                    break
            if left_forOverlap is None:
                left_forclean = trace_left[:floor(trace_left_origin*clean_threshold)]
                left_forOverlap = trace_left[floor(trace_left_origin*clean_threshold):]
                real_timegap = left_forOverlap[-1][0] - left_forOverlap[0][0]
                curtime = real_timegap
                for i,packet in enumerate(trace_right):
                    if packet[0] >= curtime:
                        right_forOverlap = trace_right[:i]              
                        break
        else:
            timegap_left = left_forOverlap[-1][0] -left_forOverlap[0][0]
            timegap_right = right_forOverlap[-1][0]
            if timegap_left >= timegap_right:
                real_timegap = timegap_right
                curtime = trace_left[-1][0] - real_timegap
                for i,packet in enumerate(reversed(trace_left)):
                    if packet[0] <= curtime:
                        left_forclean = trace_left[:trace_left_length - i]
                        left_forOverlap = trace_left[trace_left_length - i:]                       
                        break                
            else:
                real_timegap = timegap_left
                curtime = real_timegap
                for i,packet in enumerate(trace_right):
                    if packet[0] >= curtime:
                        right_forOverlap = trace_right[:i]              
                        break
        

        overlaped = len(right_forOverlap) 
        this = merge(this,left_forclean)
        right_forOverlap[:,0] += left_forOverlap[0][0]
        start = right_forOverlap[-1][0]
        overlap_seg = merge(left_forOverlap, right_forOverlap)

        this = merge(this, overlap_seg) 

    last_trace_left = trace_list[-1]["trace"][overlaped:]
    last_trace_left[:,0] -= last_trace_left[0][0] 
    last_trace_left[:,0] += start
    this = merge(this, last_trace_left)
    this = this[this[:,0].argsort()]

    if not forWFD:
        dump(this, join(output_dir, outputname+'.merge'))
        logger.debug("Merged trace is dumpped to %s.merge"%join(output_dir, outputname))   
    
    annotations, annotations_full, brust_annotations, brust_annotations_full, brust_mergelen = pick_annotations(this, mergelist, "both", output_dir, outputname)
    mergelen = len(this)  

    return annotations, annotations_full, mergelen, brust_annotations, brust_annotations_full, brust_mergelen  

def work(param):
    output_dir, position, ratio, cnt, T = param[0],param[1], param[2], param[3], param[4]
    if position == "head":
        return MergePad_head(output_dir, ratio, str(cnt), T)
    if position == "tail":
        return MergePad_tail(output_dir, ratio, str(cnt), T)
    if position == "both":
        return MergePad_both(output_dir, ratio, str(cnt), T)


def parallel(output_dir, position, ratio, mergedTrace, n_jobs = 20): 
    cnt = range(len(mergedTrace))
    l = len(cnt)
    param_dict = zip([output_dir]*l, [position]*l, [ratio]*l, cnt, mergedTrace)
    pool = mp.Pool(n_jobs)
    merge_results = pool.map(work, param_dict)
    return merge_results

def random_work(param):
    output_dir, random_positions, random_ratios, cnt, T = param[0],param[1], param[2], param[3], param[4]

    position = random_positions[random.randint(0,len(random_positions)-1)]
    ratio = random_ratios[random.randint(0,len(random_ratios)-1)]
    forWFD = True #Do not save mergetrace in .merge files.
    if position == "head":
        return MergePad_head(output_dir, ratio, str(cnt), T, forWFD)
    elif position == "tail":
        return MergePad_tail(output_dir, ratio, str(cnt), T, forWFD)
    else:
        return MergePad_both(output_dir, ratio, str(cnt), T, forWFD)


def random_parallel(output_dir, random_positions, random_ratios, mergedTrace, n_jobs = 20): 
    cnt = range(len(mergedTrace))
    l = len(cnt)
    param_dict = zip([output_dir]*l, [random_positions]*l, [random_ratios]*l, cnt, mergedTrace)
    pool = mp.Pool(n_jobs)
    merge_results = pool.map(random_work, param_dict)
    return merge_results

def save_annotationjson(output_dir, merge_results):
    traces_results = []
    brust_traces_results = []

    annotation_id = 0
    annotations_results = []
    annotations_results_full = []
    brust_annotations_results = []
    brust_annotations_results_full = []

    for i, merge_result in enumerate(merge_results): 
        #list of items, each item contains 
        #annotations, annotations_full, mergelen, brust_annotations, brust_annotations_full, brust_mergelen 
        
        annotations = merge_result[0]
        annotations_full = merge_result[1]
        brust_annotations = merge_result[3]
        brust_annotations_full = merge_result[4]
        assert len(annotations) == len(annotations_full) == len(brust_annotations) == len(brust_annotations_full), "Error occurs in compressing mergedTrace, please check."

        for j in range(len(annotations)):
            annotations[j]["id"] = annotation_id
            annotations_full[j]["id"] = annotation_id
            brust_annotations[j]["id"] = annotation_id
            brust_annotations_full[j]["id"] = annotation_id
            annotation_id += 1
            annotations_results.append(annotations[j])   
            annotations_results_full.append(annotations_full[j])           
            brust_annotations_results.append(brust_annotations[j])       
            brust_annotations_results_full.append(brust_annotations_full[j])       

        mergelen = merge_result[2]
        trace = {
            "id": i,
            "file_name": str(i)+'.merge',
            "width": mergelen,
            "height": 1
        }
        traces_results.append(trace)

                    
        brust_mergelen = merge_result[5]
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

    merge_dict = {
        "traces": traces_results,
        "annotations": annotations_results,
        "categories": categories
    }

    merge_dict_full = {
        "traces": traces_results,
        "annotations": annotations_results_full,
        "categories": categories
    }

    brust_merge_dict = {
        "traces": brust_traces_results,
        "annotations": brust_annotations_results,
        "categories": categories
    }   


    brust_merge_dict_full = {
        "traces": brust_traces_results,
        "annotations": brust_annotations_results_full,
        "categories": categories
    }   

    with open(join(output_dir, 'annotations.json'), 'w') as json_file:
        json.dump(merge_dict, json_file)

    with open(join(output_dir, 'annotations_full.json'), 'w') as json_file:
        json.dump(merge_dict_full, json_file)

    with open(join(output_dir, 'annotations_brust.json'), 'w') as json_file:
        json.dump(brust_merge_dict, json_file)

    with open(join(output_dir, 'annotations_brust_full.json'), 'w') as json_file:
        json.dump(brust_merge_dict_full, json_file)



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

    if eval(args.random):
        random_positions = ["head","tail","both"]
        random_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        results = random_parallel(args.output, random_positions, random_ratios, mergedTrace, 20)
    else:
        results = parallel(args.output, args.position, eval(args.ratio), mergedTrace, 20)

    logger.info("Totally generate %d l-traces into directory: %s" %(len(mergedTrace), args.output))

    save_annotationjson(args.output, results)
    logger.info("Generate annotations.json.")

    

