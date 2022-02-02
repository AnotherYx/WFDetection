import sys
import json
import logging
from os.path import join, abspath, dirname, pardir
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import contextlib
import io

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

def evaluation_func(gts_json, dets_json, tp):
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

        if 'full' in tp:
            cocoEval.params.areaRng  = [[0, 9000], [0, 1000], [1000, 4000], [4000, 9000]] 
        else:
            cocoEval.params.areaRng  = [[0, 4000], [0, 128], [128, 1000], [1000, 4000]]
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


def map_evaluation(targets, tp, name):   
    gts_anno_all = []
    gts_trace_all = []
    dets_annos_all = []  

    gts_annotation_counter = 0 
    dets_annotation_counter = 0

    categories = []
    categories_num = 100
    for i in range(categories_num):
        categorie = {
            "id": i,
            "name": str(i),
            "supercategory": "none"
        }
        categories.append(categorie)

    total = 0
    for target in targets:
        if 'full' in tp:
            gts_jsonfile = join(target, "annotations_brust_full.json")
        else:
            gts_jsonfile = join(target, "annotations_brust.json")
        dets_jsonfile = join(target, target.split('/')[-2]+ '_'+ tp +"_dets.json")

        with open(gts_jsonfile, 'r') as json_file:
            gts_anno = json.load(json_file)

        with open(dets_jsonfile, 'r') as json_file:
            dets_annos = json.load(json_file)


        if "glue" in tp:
            gts_anno_shift = []
            for anno in gts_anno["annotations"]:
                if anno['category_id']>=100:
                    anno['category_id']-=100
                gts_anno_shift.append(anno)

            gts_100cat_dict = {
                "traces": gts_anno["traces"],
                "annotations": gts_anno_shift,
                "categories": categories
            } 
            gts_100cat = join(target, "annotations_brust_100cat.json")
            with open(gts_100cat, 'w') as json_file:
                json.dump(gts_100cat_dict, json_file)            

            dets_anno_shift = []
            for anno in dets_annos:
                if anno['category_id']>=100:
                    anno['category_id']-=100
                dets_anno_shift.append(anno)

            dets_100cat = join(target, target.split('/')[-2]+ '_'+ tp +"_100cat_dets.json")
            with open(dets_100cat, 'w') as json_file:
                json.dump(dets_anno_shift, json_file)  
            logger.info("Evaluating glue results in 100-categories: %s"%target.split('/')[-2])  
            evaluation_func(gts_100cat, dets_100cat, "glue")

        logger.info("processing dets of %s"%target.split('/')[-2])  
        #evaluation_func(gts_jsonfile, dets_jsonfile, tp)

        for anno in dets_annos:
            anno['id'] = dets_annotation_counter
            dets_annotation_counter+=1
            anno['trace_id']+=total
            if "glue" in tp:
                if anno['category_id']>=100:
                    anno['category_id']-=100
            dets_annos_all.append(anno)

        for trace in gts_anno["traces"]:
            trace['id'] += total
            trace['file_name'] = str(int(trace['file_name'].split('.')[0])+total)+'.npy'
            gts_trace_all.append(trace)


        for anno in gts_anno["annotations"]:
            anno['id'] = gts_annotation_counter
            gts_annotation_counter+=1
            anno['trace_id']+=total
            if "glue" in tp:
                if anno['category_id']>=100:
                    anno['category_id']-=100
            gts_anno_all.append(anno)

        total += len(gts_anno["traces"])

    gts_all_dict = {
        "traces": gts_trace_all,
        "annotations": gts_anno_all,
        "categories": categories
    } 

    gts_all = join(BASE_DIR, "simulation/results", targets[0].split('/')[-3], name + "_"+ tp +"_gts.json")
    with open(gts_all, 'w') as json_file:
        json.dump(gts_all_dict, json_file)

    dets_all = join(BASE_DIR, "simulation/results", targets[0].split('/')[-3], name + "_"+ tp +"_dets.json")
    with open(dets_all, 'w') as json_file:
        json.dump(dets_annos_all, json_file)
    logger.info("results of %s [%s]"%(name,tp))  
    evaluation_func(gts_all, dets_all, tp)



if __name__ == '__main__':
    """
    just an example of how to use our code.
    You may want to write your own script with your datasets and other customizations.
    """
    init_logger()

    targets = []
    for l in range(2,18):
        target = join(BASE_DIR, "simulation/results/noise_glue/noise_glue_"+str(l)+'/')
        targets.append(target)
    map_evaluation(targets, "glue", "noise_glue_all")
    #just for example