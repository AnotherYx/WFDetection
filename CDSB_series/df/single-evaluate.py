#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is for experiment: random number of splits

@author: aaron
"""
# import logging
# import glob 
# import multiprocessing as mp 
# import os 
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# import sys
# from keras.models import load_model
# import argparse
# import numpy as np
# import configparser
# from keras.preprocessing.sequence import pad_sequences
# import const
# import pandas as pd 
# import tensorflow as tf
# import keras 


# config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 20} ) 
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config) 
# keras.backend.set_session(sess)

import logging
import glob 
import multiprocessing as mp 
import os 
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # No INFO and WARNING
import sys
from tensorflow.keras.models import load_model
import argparse
import numpy as np
import configparser
from tensorflow.keras.preprocessing.sequence import pad_sequences
import const
import pandas as pd 
import tensorflow as tf
import tensorflow.python.keras as keras


tf.device('/GPU:0')
config = tf.compat.v1.ConfigProto(device_count = {'GPU': 1 , 'CPU': 20})
config.gpu_options.allow_growth = True
sess=tf.compat.v1.Session(config=config) 
keras.backend.set_session(sess)

logger = logging.getLogger('df')
LENGTH = 10000
def read_conf(file):
    cf = configparser.ConfigParser()
    cf.read(file)  
    return dict(cf['default'])
def config_logger(args):
    # Set file
    log_file = sys.stdout
    if args.log != 'stdout':
        log_file = open(args.log, 'w')
    ch = logging.StreamHandler(log_file)

    # Set logging format
    ch.setFormatter(logging.Formatter(const.LOG_FORMAT))
    logger.addHandler(ch)

    # Set level format
    logger.setLevel(logging.INFO)

def parse_arguments():

    parser = argparse.ArgumentParser(description='Random Evaluate.')

    parser.add_argument('-m',
                        metavar='<model path>',
                        help='Path to the directory of the model')
    parser.add_argument('-p',
                        metavar='<raw feature path>',
                        help='Path to the directory of the extracted features')   
    parser.add_argument('--log',
                        type=str,
                        dest="log",
                        metavar='<log path>',
                        default='stdout',
                        help='path to the log file. It will print to stdout by default.')
    # Parse arguments
    args = parser.parse_args()
    config_logger(args)
    return args


def extractfeature(f):
    fname = f.split('/')[-1]
    with open(f,'r') as f:
        tcp_dump = f.readlines()
#            return tcp_dump, 1

    feature = pd.Series(tcp_dump).str.slice(0,-1).str.split('\t',expand = True).astype("float")
    feature = np.array(feature.iloc[:,1]).astype("int")
    return feature

def resultParser(resultfile):
    p_results = []
    tp, wp, fp, tn, fn, p, n = 0, 0, 0, 0, 0, 0, 0

    with open(resultfile,'r') as f:
        lines = f.readlines()
        for line in lines:
            flag = line[:-1].split('\t')
            det = int(flag[1])
            if "-" in flag[0]:
                gt = int(flag[0].split('-')[0])
                p += 1
                if det < 100:
                    if det == gt:
                        tp += 1
                    else:
                        wp += 1
                else:
                    fn += 1
            else:
                gt = 100
                n += 1
                if det < 100:
                    fp += 1
                else:
                    tn += 1
            
    print('total\ttp={} wp={} fp={} tn={} fn={} p={} n={}'.format(tp, wp, fp, tn, fn, p, n))
    precision = tp / (tp+fp)
    accuracy = (tp+tn) / (p+n)
    recall = tp / p
    TPR = tp/p
    print("TPR = {} \t recall = {}  \t accuracy = {} \t precision = {}".format(TPR, recall, accuracy, precision))

def pred_sing_trace(fdirs,prtype):
    global MON_SITE_NUM, model
    X_test  = []
    [X_test.append(extractfeature(f)) for f in fdirs]
    X_test = pad_sequences(X_test, padding ='post', truncating = 'post', value = 0, maxlen = LENGTH)
    X_test = X_test[:, :, np.newaxis]
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis = 1)

    outputdir = os.path.join(const.randomdir, fdirs[0].split('/')[-3], fdirs[0].split('/')[-2]+"_"+prtype)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    resultfile = os.path.join(outputdir,'predresult.txt')
    with open(resultfile,'w') as f:
        for i,filename in enumerate(fdirs):
            f.write(filename.split("/")[-1].split('.')[0]+'\t'+str(y_pred[i])+'\n')
    resultParser(resultfile)
    # logger.info("Write into file {}".format(os.path.join(outputdir,trace_id + '-predresult.txt')))



if __name__ == '__main__':    
    global MON_SITE_NUM, model
    args = parse_arguments()
    # logger.info("Arguments: %s" % (args))
    cf = read_conf(const.confdir)
    MON_SITE_NUM = int(cf['monitored_site_num'])
    
    
    model = load_model(args.m)

    testfolder = args.p
    fdirs = glob.glob(os.path.join(args.p,'*'))

    pred_sing_trace(fdirs,args.m.split('.')[1][-2:])
