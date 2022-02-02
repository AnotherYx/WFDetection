#Modified from: https://github.com/websitefingerprinting/WebsiteFingerprinting, created by WFDetection.
import numpy as np 
import argparse
import logging
import sys
import pandas as pd
import os
from os.path import join
import multiprocessing as mp
import time
import datetime
import glob
from os import mkdir

logger = logging.getLogger('front')

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

    parser.add_argument('p',
                        metavar='<traces path>',
                        help='Path to the directory with the traffic traces to be simulated.')
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
    config_logger(args)
    return args

def load_trace(fdir):
    with open(fdir,'r') as f:
        tmp = f.readlines()
    t = pd.Series(tmp).str.slice(0,-1).str.split('\t',expand = True).astype('float')
    return np.array(t)

def dump(trace, fname):
    global output_dir
    with open(join(output_dir,fname), 'w') as fo:
        for packet in trace:
            fo.write("{:.4f}".format(packet[0]) +'\t' + "{}".format(int(packet[1]))\
                + '\n')

def simulate(fdir):
    if not os.path.exists(fdir):
        return
    # logger.debug("Simulating trace {}".format(fdir))
    np.random.seed(datetime.datetime.now().microsecond)
    trace = load_trace(fdir)
    trace = RP(trace)
    fname = fdir.split('/')[-1]
    dump(trace, fname)

def RP(trace):

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
    last_pkt_time = trace[-1][0]    
    
    client_timetable = getTimestamps(client_wnd, client_dummy_pkt)
    client_timetable = client_timetable[np.where(start_padding_time+client_timetable[:,0] <= last_pkt_time)]

    server_timetable = getTimestamps(server_wnd, server_dummy_pkt)
    server_timetable[:,0] += first_incoming_pkt_time
    server_timetable = server_timetable[np.where(start_padding_time+server_timetable[:,0] <= last_pkt_time)]

    client_pkts = np.concatenate((client_timetable, 1*np.ones((len(client_timetable),1))),axis = 1)
    server_pkts = np.concatenate((server_timetable, -1*np.ones((len(server_timetable),1))),axis = 1)

    noisy_trace = np.concatenate( (trace, client_pkts, server_pkts), axis = 0)
    noisy_trace = noisy_trace[ noisy_trace[:, 0].argsort(kind = 'mergesort')]
    return noisy_trace

def getTimestamps(wnd, num):
    # timestamps = sorted(np.random.exponential(wnd/2.0, num))   
    # print(wnd, num)
    # timestamps = sorted(abs(np.random.normal(0, wnd, num)))
    timestamps = sorted(np.random.rayleigh(wnd,num))
    # print(timestamps[:5])
    # timestamps = np.fromiter(map(lambda x: x if x <= wnd else wnd, timestamps),dtype = float)
    return np.reshape(timestamps, (len(timestamps),1))


def parallel(flist, n_jobs = 20):
    pool = mp.Pool(n_jobs)
    pool.map(simulate, flist)


if __name__ == '__main__':

    args = parse_arguments()
    logger.info(args)

    flist = glob.glob(join(args.p,'*'))
    output_dir = args.output
    if not os.path.exists(output_dir):
        mkdir(output_dir)
    logger.info("Traces are dumped to {}".format(output_dir))
    start = time.time()

    parallel(flist)
    logger.info("Time: {}".format(time.time()-start))
