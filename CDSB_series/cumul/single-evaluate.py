#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is for experiment: random number of splits

@author: aaron
"""
from main import *
import glob 
import multiprocessing as mp 
import os 
from extract import *
import joblib
logger = logging.getLogger('cumul')



def parse_arguments():

    parser = argparse.ArgumentParser(description='Evaluate.')

    parser.add_argument('-m',
                        metavar='<model path>',
                        help='Path to the directory of the model')
    parser.add_argument('-p',
                        metavar='<raw trace path>')
    parser.add_argument('-o',
                        metavar='<feature path>',
                        help='Path to the directory of the extracted features')    
    parser.add_argument('-mode',
                        metavar='<head or other>',
                        help='To test head or other')    

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
    global MON_SITE_NUM
    fname = f.split('/')[-1].split(".")[0]
    # logger.info('Processing %s...'%f)
    try:
        t = parse(f)
        features = extract(t)
        if '-' in fname:
            label = int(fname.split('-')[0])
        else:
            label = int(MON_SITE_NUM)

        return (features, label)
    except Exception as e:
        print(e)
        return None


def pred_sing_trace(fdirs, scaler, model, prtype):
    X_test  = []
    [X_test.append(extractfeature(f)[0]) for f in fdirs]
    X_test = scaler.transform(X_test)
    y_pred = model.predict(X_test)

    outputdir = os.path.join(ct.randomdir, fdirs[0].split('/')[-3], fdirs[0].split('/')[-2]+"_"+prtype)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    resultfile = os.path.join(outputdir,'predresult.txt')
    with open(resultfile,'w') as f:
        for i,filename in enumerate(fdirs):
            f.write(filename.split("/")[-1].split('.')[0]+'\t'+str(y_pred[i])+'\n')


    
if __name__ == '__main__':    
    global MON_SITE_NUM
    args = parse_arguments()
    logger.info("Arguments: %s" % (args))
    cf = read_conf(ct.confdir)
    MON_SITE_NUM = int(cf['monitored_site_num'])

    model =  joblib.load(args.m)
    logger.info('loading original data...')
    dic = np.load(args.o, allow_pickle = True).item()
    X = np.array(dic['feature'])
    y = np.array(dic['label'])

    # normalize the data
    scaler = preprocessing.MinMaxScaler((-1,1))
    scaler.fit(X)

    testfolder = args.p
    fdirs = glob.glob(os.path.join(args.p,'*'))
    # for f in fdirs[:]:
    #     pred_sing_trace((f,scaler,model))
    pred_sing_trace(fdirs, scaler, model, args.m.split('.')[1][-2:]):

    # dic = np.load(args.p).item()   
    # X = np.array(dic['feature'])
    # y = np.array(dic['label'])
    # X = scaler.transform(X)
    # logger.info('data are transformed into [-1,1]')

    # y_pred = model.predict(X)
    # score_func(y, y_pred)