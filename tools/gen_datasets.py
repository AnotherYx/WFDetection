import subprocess
import glob
import os
from os import mkdir
import numpy as np
from os.path import join, abspath, dirname, pardir
from os import mkdir
import logging
import sys

# Directories
BASE_DIR = abspath(join(dirname(__file__), pardir))
logger = logging.getLogger('Dataset Generator')

attacktrain_splittrain = join(BASE_DIR, "data/attacktrain_splittrain/")

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

def CreateMergedTrace(traces_path, nums, BaseRate):
    '''generate length-N merged trace'''
    '''with prob baserate/(baserate+1) a nonsensitive trace is chosen'''
    '''with prob 1/(baserate+1) a sensitive trace is chosen'''
    list_names = glob.glob(join(traces_path, '*'))
    list_sensitive = glob.glob(join(traces_path, '*-*'))
    list_nonsensitive = list(set(list_names) - set(list_sensitive))
    
    s1 = len(list_sensitive)
    s2 = len(list_nonsensitive)
    
    mergedTrace = []
    for num in nums:
        mergedTrace.append(np.random.choice(list_sensitive+list_nonsensitive, num, replace = False,\
                                  p = [1.0/(s1*(BaseRate+1))]*s1 + [BaseRate /(s2*(BaseRate+1))]*s2))
    return mergedTrace


def save_listfile(listpath, mergedTrace):
    with open(listpath,'w') as f:
        for mergelist in mergedTrace:
            labels = ""
            for fname in mergelist:
                label = fname.split('/')[-1]
                labels += label + '\t'
            f.write(labels+'\n')


if __name__ == "__main__":
    """
    just an example of how to use our code.
    You may want to write your own script with your datasets and other customizations.
    """
    init_logger()

    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    positions = ["head", "tail", "both"]

    '''--------------noise_random--------------'''
    noise_random_WFD_output_dir = join(BASE_DIR, "simulation/results", 'noise_random_WFD')
    if not os.path.exists(noise_random_WFD_output_dir):
        mkdir(noise_random_WFD_output_dir)       
    baserate = 1
    num = (11000//3) * len(ratios) * len(positions)
    nums = [3]*num
    mergedTrace = CreateMergedTrace(attacktrain_splittrain, nums, baserate) 
    listpath = join(noise_random_WFD_output_dir, 'list')
    numspath = join(noise_random_WFD_output_dir, "num.npy")
    np.save(numspath, np.array(nums))
    save_listfile(listpath, mergedTrace)

    cmd = "python " + join(BASE_DIR,"simulation/overlap_merge_api.py ") + attacktrain_splittrain + \
        " -listpath " + listpath + " -output " + noise_random_WFD_output_dir + \
        ' -random True'
    subprocess.call(cmd, shell= True)
    logger.info("Successfully generating datasets for train2: noise_random_WFD")


    '''-------------noise_glue-------------'''
    noise_glue_WFD_output_dir = join(BASE_DIR, "simulation/results", 'noise_glue_WFD')
    if not os.path.exists(noise_glue_WFD_output_dir):
        mkdir(noise_glue_WFD_output_dir)

    length_list = []
    num_of_merge = 0
    for l in range(2,18):
        num_of_merge += (9900//l)
        length_list.append(l)
    baserate = 10
    nums = np.random.choice(length_list, num_of_merge)
    mergedTrace = CreateMergedTrace(attacktrain_splittrain, nums, baserate) 
    listpath = join(noise_glue_WFD_output_dir, 'list')
    numspath = join(noise_glue_WFD_output_dir, "num.npy")
    np.save(numspath, np.array(nums))
    save_listfile(listpath, mergedTrace)

    cmd = "python " + join(BASE_DIR,"simulation/glue_api.py ") + attacktrain_splittrain + \
        " -listpath " + listpath + " -noise True"  + " -glue True" + " -forWFD True" + " -output " + noise_glue_WFD_output_dir
    subprocess.call(cmd, shell= True)
    logger.info("Successfully generating datasets for train3: noise_glue_WFD")
