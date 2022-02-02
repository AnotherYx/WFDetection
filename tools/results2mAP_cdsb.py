import sys
import json
import logging
import numpy as np
from os.path import join, abspath, dirname, pardir
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import contextlib
import io
import os

LOG_FORMAT = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
BASE_DIR = abspath(join(dirname(__file__), pardir))
logger = logging.getLogger('results2mAP')

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

def evaluation_func(gts_json, dets_json):
    annType = 'bbox'  
    metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm",	"ARl"]

    redirect_string = io.StringIO()    
    with contextlib.redirect_stdout(redirect_string):
        cocoGt=COCO(gts_json)
        cocoDt=cocoGt.loadRes(dets_json)

        CatIds=cocoGt.getCatIds()

        # running evaluation
        cocoEval = COCOeval(cocoGt,cocoDt,annType)
        cocoEval.params.maxDets  = [1, 10, 100]
        cocoEval.params.areaRng  = [[0, 45000], [0, 5000], [5000, 10000], [10000, 45000]]
        cocoEval.evaluate()
        cocoEval.accumulate()   
        
    redirect_string.getvalue()

    cocoEval.summarize()

    results = {
        metric: float(cocoEval.stats[idx]*100 if cocoEval.stats[idx] >= 0 else "nan")
        for idx, metric in enumerate(metrics)
    }
    
    print('AP  = %5f\tAP50 = %5f\tAP75 = %5f'%(results["AP"],results["AP50"],results["AP75"]))
    print('APs = %5f\tAPm = %5f\tAPl = %5f'%(results["APs"],results["APm"],results["APl"]))
    print('AR1 = %5f\tAR10 = %5f\tAR100 = %5f'%(results["AR1"],results["AR10"],results["AR100"]))
    print('ARs = %5f\tARm = %5f\tARl = %5f'%(results["ARs"],results["ARm"],results["ARl"]))
  
def read_p_splits(splitfile, traces):
    '''input: a file: odd line true split; even line file name'''
    p_splits = []  #predict_split
    with open(splitfile,'r') as f:
        '''first row is comment'''
        tmp = f.readlines()[1:]
        p_splits_tmp = np.array(tmp[0::2]) #pick out predict_split
    for i,ss in enumerate(p_splits_tmp):
        ss = ss[:-1].split('\t')
        ss = [0] + ss
        ss.append(traces[i]["width"])
        p_splits.append(np.array(ss).astype('int'))
    return p_splits

def read_p_results(attackpath, name, size):
    p_results = []
    for i in range(size):
        headf = join(attackpath, 'randomresults/', name, 'head/', str(i)+"-predresult.txt") #detection result
        otherf = join(attackpath, 'randomresults/', name, 'other/', str(i)+"-predresult.txt")
        tmp = []
        with open(headf,'r') as hf:
            lines = hf.readlines()
            for line in lines:
                r = line[:-1].split('\t')
                tmp.append(r)
        with open(otherf,'r') as of:
            lines = of.readlines()
            for line in lines:
                r = line[:-1].split('\t')
                tmp.append(r)
        results_tmp = np.array(np.array(tmp).astype('float'))
        p_results.append(results_tmp)

    return p_results

def annos_gen(predict_results, predict_splits):
    annotation_index = 0
    annotations = []

    for i, category_scores in enumerate(predict_results):
        splitpoints = predict_splits[i]
        for j, c_s in enumerate(category_scores):
            if c_s[0] < 100:
                start = int(splitpoints[j])
                end = int(splitpoints[j+1])
                width = end - start
                annotation = {
                    "id": annotation_index,
                    "area": width,
                    "iscrowd": 0,
                    "trace_id": i,
                    "bbox": [
                        start,
                        0,
                        width,
                        1
                    ],
                    "category_id": int(c_s[0]),
                    "score": c_s[1]
                }
                annotation_index = annotation_index + 1
                annotations.append(annotation)

    return annotations

def resultsTrans(gts_anno, dets_jsonfile, attack, target):
    attackpath = join(BASE_DIR, 'CDSB_series', attack) 

    splitresult_file = join(BASE_DIR, "CDSB_series/xgboost/scores", target.split("/")[-2], "splitresult_decision.txt")

    predict_splits = read_p_splits(splitresult_file, gts_anno["traces"])
    predict_results = read_p_results(attackpath, target.split("/")[-2], len(gts_anno["traces"]))

    dets_annos = annos_gen(predict_results, predict_splits)
    with open(dets_jsonfile, 'w') as json_file:
            json.dump(dets_annos, json_file) #only save dets_annos, work as input of evaluation_func() 
    return dets_annos

def map_evaluation(attacks, targets, tp): 
    gts_anno_all = []
    gts_trace_all = []
    dets_annos_all_df = []  
    dets_annos_all_knn = [] 
    dets_annos_all_cumul = [] 
    dets_annos_all_kfp = []
    gts_annotation_counter = 0 
    dets_annotation_counter_df = 0
    dets_annotation_counter_cumul = 0
    dets_annotation_counter_kfp = 0
    dets_annotation_counter_knn = 0
    total = 0
    for target in targets:
        gts_jsonfile = join(target, "annotations.json")
        with open(gts_jsonfile, 'r') as json_file:
            gts_anno = json.load(json_file)

        for attack in attacks:
            dets_jsonfile = join(target,  'annotations_' + attack + "_dets.json")
            if not os.path.exists(dets_jsonfile):
                dets_annos = resultsTrans(gts_anno, dets_jsonfile, attack, target)
            else:
                with open(dets_jsonfile, 'r') as json_file:
                    dets_annos = json.load(json_file)
            # logger.info("results of %s by WFattack: %s"%(target.split('/')[-2], attack) )  
            # evaluation_func(gts_jsonfile, dets_jsonfile)

            if attack == 'df':
                for anno in dets_annos:
                    anno['id'] = dets_annotation_counter_df
                    dets_annotation_counter_df+=1
                    anno['trace_id']+=total
                    dets_annos_all_df.append(anno)
            elif attack == 'cumul':
                for anno in dets_annos:
                    anno['id'] = dets_annotation_counter_cumul
                    dets_annotation_counter_cumul+=1
                    anno['trace_id']+=total
                    dets_annos_all_cumul.append(anno)
            elif attack == 'kfingerprinting':
                for anno in dets_annos:
                    anno['id'] = dets_annotation_counter_kfp
                    dets_annotation_counter_kfp+=1
                    anno['trace_id']+=total
                    dets_annos_all_kfp.append(anno)
            elif attack == 'knn':
                for anno in dets_annos:
                    anno['id'] = dets_annotation_counter_knn
                    dets_annotation_counter_knn+=1
                    anno['trace_id']+=total
                    dets_annos_all_knn.append(anno)

        for trace in gts_anno["traces"]:
            trace['id'] += total
            trace['file_name'] = str(int(trace['file_name'].split('.')[0])+total)+'.npy'
            gts_trace_all.append(trace)

        for anno in gts_anno["annotations"]:
            anno['id'] = gts_annotation_counter
            gts_annotation_counter+=1
            anno['trace_id']+=total
            gts_anno_all.append(anno)

        total += len(gts_anno["traces"])


    categories = []
    categories_num = 100
    for i in range(categories_num):
        categorie = {
            "id": i,
            "name": str(i),
            "supercategory": "none"
        }
        categories.append(categorie)

    gts_all_dict = {
        "traces": gts_trace_all,
        "annotations": gts_anno_all,
        "categories": categories
    } 

    gts_all = join(BASE_DIR, "simulation/results", targets[0].split('/')[-3], "annotations_"+ tp +"_gts.json")
    with open(gts_all, 'w') as json_file:
        json.dump(gts_all_dict, json_file)

    dets_all_df = join(BASE_DIR, "simulation/results", targets[0].split('/')[-3], "annotations_"+ tp +"_df_dets.json")
    with open(dets_all_df, 'w') as json_file:
        json.dump(dets_annos_all_df, json_file)
    logger.info("results of %s by WFattack: df"%tp)  
    evaluation_func(gts_all, dets_all_df)

    dets_all_cumul = join(BASE_DIR, "simulation/results", targets[0].split('/')[-3], "annotations_"+ tp +"_cumul_dets.json")
    with open(dets_all_cumul, 'w') as json_file:
        json.dump(dets_annos_all_cumul, json_file)
    logger.info("results of %s by WFattack: cumul"%tp)  
    evaluation_func(gts_all, dets_all_cumul)

    dets_all_kfp = join(BASE_DIR, "simulation/results", targets[0].split('/')[-3], "annotations_"+ tp +"_kfp_dets.json")
    with open(dets_all_kfp, 'w') as json_file:
        json.dump(dets_annos_all_kfp, json_file)
    logger.info("results of %s by WFattack: kfingerprinting"%tp)  
    evaluation_func(gts_all, dets_all_kfp)
        
    dets_all_knn = join(BASE_DIR, "simulation/results", targets[0].split('/')[-3], "annotations_"+ tp +"_knn_dets.json")
    with open(dets_all_knn, 'w') as json_file:
        json.dump(dets_annos_all_knn, json_file)
    logger.info("results of %s by WFattack: knn"%tp)  
    evaluation_func(gts_all, dets_all_knn)



if __name__ == '__main__':
    """
    just an example of how to use our code.
    You may want to write your own script with your datasets and other customizations.
    """
    init_logger()
    attacks = ["df", "cumul", "kfingerprinting", "knn"]
    glue_targets = []
    for i in range(2,18):
        target = join(BASE_DIR, "simulation/results/noise_glue/noise_glue_"+str(i) + "/")
        glue_targets.append(target)
    map_evaluation(attacks,glue_targets,"all")
