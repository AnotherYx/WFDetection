import sys
import logging
import numpy as np
import json
from os.path import join, abspath, dirname, pardir


LOG_FORMAT = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
BASE_DIR = abspath(join(dirname(__file__), pardir))
logger = logging.getLogger('results2precision')


def init_logger():
    # Set file
    log_file = sys.stdout
    ch = logging.StreamHandler(log_file)

    LOG_FORMAT = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    # Set logging format
    ch.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(ch)

    # Set level format
    logger.setLevel(logging.INFO)


def pick_score(item):
    return item[3]

def gts_restore(gts,length):
    annotlist = []
    for i in range(length):
        annotlist.append([])

    for item in gts:
        region = [item["bbox"][0],item["bbox"][0]+item["bbox"][2],item["category_id"]]
        annotlist[item["trace_id"]].append(region)
    for i in range(length):
        annotlist[i].sort()

    return annotlist

def dets_restore(dets,length):
    annotlist = []
    pflag = 0
    for i in range(length):
        annotlist.append([])

    for item in dets:
        region = [item["bbox"][0], item["bbox"][0]+item["bbox"][2], item["category_id"], item["score"], pflag]
        annotlist[item["trace_id"]].append(region)
    for i in range(length):
        annotlist[i].sort()
    return annotlist


def transform(gts, dets, length):  

    with open(dets, 'r') as json_file:
        tmp = json.load(json_file)
    with open(gts, 'r') as json_file:
        tmp2 = json.load(json_file)

    dets_anno = dets_restore(tmp,length)
    gts_anno = gts_restore(tmp2["annotations"],length)

    return gts_anno,dets_anno

def load_trace(fname, t = 999):
    '''load a trace from fpath/fname up to t time.'''
    '''return trace'''
    pkts = []
    with open(fname, 'r') as f:
        for line in f:
            try:    
                timestamp, direction = line.strip().split('\t')
                pkts.append([float(timestamp), int(direction)])
                if float(timestamp) >= t+0.5:
                    break
            except ValueError:
                logger.warn("Could not split line: %s in %s", line, fname)
        return np.array(pkts)

def calcu_NP(mergelist):
    N = 0
    P = 0
    for merge in mergelist:
        merge = merge[:-2].split('\t')
        um_num = 0
        for name in merge:
            if "-" not in name:
                um_num += 1
        N = N + um_num
        P = P + (len(merge) - um_num)
    return N, P


def calcu_IOU(det_anno, gt_anno):
    if gt_anno[1]<=det_anno[0] or gt_anno[0]>=det_anno[1]:
        IOU = 0
    else:
        tmp = [det_anno[0],det_anno[1],gt_anno[0],gt_anno[1]]
        tmp.sort()
        IOU = (tmp[2]-tmp[1])/(tmp[3]-tmp[0])
    return IOU


def choose_det(dets, neighbor_threshlold, Score_threshlold):
    det_choose = []
    tmp = []
    for item in dets:
        if item[3] >= Score_threshlold:
            tmp.append(item)
    while(len(tmp)):
        neighbors_id = [0]
        neighbors = []
        for j in range(1, len(tmp)):
            iou = calcu_IOU(tmp[0], tmp[j]) 
            if iou >= neighbor_threshlold:
                neighbors_id.append(j)
            elif iou ==0:
                break
        for id in neighbors_id:
            neighbors.append(tmp[id])
        neighbors_id.sort(reverse=True)
        for id in neighbors_id:
            del tmp[id]
        neighbors.sort(key=pick_score, reverse=True)
        det_choose.append(neighbors[0])

    return det_choose
    

def analyzer(gts_anno, dets_anno, neighbor_threshlold, IOU_threshlold, Score_threshlold):
    #note that IOU_threshold shoud lager than neighbor_threshlold
    TP, FP, WP = 0,0,0
    count = 0
    for i,dets in enumerate(dets_anno):
        det_choose = choose_det(dets, neighbor_threshlold, Score_threshlold)
        count+=len(det_choose)
        for gt_anno in gts_anno[i]:
            for det_anno in det_choose:
                if calcu_IOU(det_anno, gt_anno) > IOU_threshlold and det_anno[4] != 1:
                    det_anno[4]=1
                    if det_anno[2] != gt_anno[2]:
                            WP += 1
                    elif det_anno[2] == gt_anno[2]:
                            TP += 1 
                    else:
                        print("-----------unexpected ERROR-----------")
        #count the FP
        for det_anno in det_choose:
            if det_anno[4]==0:
                FP += 1

    print("dets num in total:%d"%count)
    return TP, FP, WP


def main_func(gtsname, detsname, listpath, neighbor_threshlold, IOU_threshlold, Score_threshlold):
    with open(listpath, 'r') as f:
        mergelist = f.readlines()

    '''choose valuable results and using wang's metrics to evaluate'''
    gts_anno, dets_anno = transform(gtsname,detsname,len(mergelist))
    N, P = calcu_NP(mergelist)
    TP, FP, WP = analyzer(gts_anno, dets_anno, neighbor_threshlold, IOU_threshlold, Score_threshlold)
    precison = (TP/P)/((TP/P)+(WP/P)+10*(FP/N))
    TPR = TP/P
    print("N = %-10d\tP = %-10d\tTP = %-10d\tFP = %-10d\tWP = %-10d\t\nprecison = %.10f\tTPR = %.10f"%(N, P, TP, FP, WP, precison, TPR))


if __name__ == "__main__":
    """
    just an example of how to use our code.
    You may want to write your own script with your datasets and other customizations.
    """
    init_logger()

    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    pos = ["head", "tail", "both"]

    neighbor_threshlold = 0.5
    IOU_threshlold = 0.8
    Score_threshlold = 0.5

    mergelist_all = []
    for l in range(2,18):
        test_Dts = join(BASE_DIR, "simulation/results/noise_glue/noise_glue_"+str(l))
        listpath = join(test_Dts,"list")

        logger.info("results of: %s, "%(test_Dts.split('/')[-1]))
        gtsname = join(test_Dts, "annotations_brust.json")
        detsname = join(test_Dts, test_Dts.split('/')[-1] + "_glue_det.json")
        # main_func(gtsname, detsname, listpath, neighbor_threshlold, IOU_threshlold, Score_threshlold)
        listpath = join(test_Dts, 'list')
        with open(listpath, 'r') as f:
            mergelist = f.readlines()        
        mergelist_all.extend(mergelist)

    alllist = join(BASE_DIR, "simulation/results/noise_glue/list")
    with open(alllist,'w') as f:
        for mergelist in mergelist_all:
            f.write(mergelist)        

    gtsname = join(BASE_DIR, "simulation/results/noise_glue/noise_glue_all/annotations_brust.json")
    detsname = join(BASE_DIR, "simulation/results/noise_glue/noise_glue_all/noise_glue_all_glue_det.json")
    main_func(gtsname, detsname, alllist, neighbor_threshlold, IOU_threshlold, Score_threshlold)
    #just for example