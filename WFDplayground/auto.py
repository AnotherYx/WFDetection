import subprocess
from shutil import copyfile
import logging
import sys
from shutil import copyfile
from os.path import join, abspath, dirname, pardir

# Directories
BASE_DIR = abspath(join(dirname(__file__), pardir))

train2022 = join(BASE_DIR,"datasets/WFD/train2022")
train2022_json = join(BASE_DIR,"datasets/WFD/annotations/instances_train2022.json")
test2022 = join(BASE_DIR,"datasets/WFD/test2022")
test2022_json = join(BASE_DIR,"datasets/WFD/annotations/instances_test2022.json")

datasets_BASE = join(BASE_DIR,"simulation/results")

final_model = join(BASE_DIR,"WFDplayground/output/model_final.pth")
inference_results = join(BASE_DIR,"WFDplayground/output/inference/coco_instances_results.json")
logger = logging.getLogger('Auto WFD')

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


def sflink_generator_full(train_Dts, test_Dts):
    subprocess.call("rm " + train2022, shell= True)
    subprocess.call("rm " + train2022_json, shell= True)
    subprocess.call("rm " + test2022, shell= True)
    subprocess.call("rm " + test2022_json, shell= True)
    subprocess.call("ln -s " + join(train_Dts, "brust_merge") + " " + train2022, shell= True)
    subprocess.call("ln -s " + join(train_Dts, "annotations_brust_full.json") + " " + train2022_json, shell= True)
    subprocess.call("ln -s " + join(test_Dts, "brust_merge") + " " + test2022, shell= True)
    subprocess.call("ln -s " + join(test_Dts, "annotations_brust_full.json") + " " + test2022_json, shell= True)

def sflink_generator_glue(train_Dts, test_Dts):
    subprocess.call("rm " + train2022, shell= True)
    subprocess.call("rm " + train2022_json, shell= True)
    subprocess.call("rm " + test2022, shell= True)
    subprocess.call("rm " + test2022_json, shell= True)
    subprocess.call("ln -s " + join(train_Dts, "brust_merge") + " " + train2022, shell= True)
    subprocess.call("ln -s " + join(train_Dts, "annotations_brust.json") + " " + train2022_json, shell= True)
    subprocess.call("ln -s " + join(test_Dts, "brust_merge") + " " + test2022, shell= True)
    subprocess.call("ln -s " + join(test_Dts, "annotations_brust.json") + " " + test2022_json, shell= True)


if __name__ == '__main__':
    """
    just an example of how to use our code.
    You may want to write your own script with your datasets and other customizations.
    """
    init_logger()
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    pos = ["head", "tail", "both"]
    logger.info("Experiment3: noise_glue")

    train_Dts = join(datasets_BASE, 'noise_glue_WFD')
    noise_glue_model = join(BASE_DIR, 'models',"noise_glue.pth")
    test_Dts = join(datasets_BASE, "noise_glue/noise_glue_2")
    sflink_generator_glue(train_Dts, test_Dts)
    subprocess.call("pods_train --num-gpus 1 MODEL.WFD.DECODER.NUM_CLASSES 200", shell= True)
    copyfile(final_model, noise_glue_model)
    copyfile(inference_results, join(test_Dts,test_Dts.split('/')[-1]+'_glue_dets.json'))
    logger.info("Traing finished, tested on: %s, final clean model saved"%(test_Dts.split('/')[-1]))


    for l in range(3,18):
        test_Dts = join(datasets_BASE, "noise_glue/noise_glue_"+str(l))
        logger.info("Testing on: %s"%(test_Dts.split('/')[-1]))
        sflink_generator_glue(train_Dts, test_Dts)
        subprocess.call("pods_test --num-gpus 1 MODEL.WFD.DECODER.NUM_CLASSES 200 MODEL.WEIGHTS " + noise_glue_model, shell= True)
        copyfile(inference_results, join(test_Dts,test_Dts.split('/')[-1]+'_glue_dets.json'))

    test_Dts = join(datasets_BASE, "noise_glue/noise_glue_all")
    sflink_generator_glue(train_Dts, test_Dts)
    logger.info("Testing on: %s"%(test_Dts.split('/')[-1]))
    subprocess.call("pods_test --num-gpus 1  MODEL.WFD.DECODER.NUM_CLASSES 200 MODEL.WEIGHTS " + noise_glue_model, shell= True)
    copyfile(inference_results, join(test_Dts,test_Dts.split('/')[-1]+'_glue_dets.json'))
    #just an example





