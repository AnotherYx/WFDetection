import sys
import logging
import numpy as np
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


def evaluation_func(mergelist, p_results, size):
    tp, wp, fp, p, n = 0, 0, 0, 0 ,0
    assert len(mergelist)==size==len(p_results),"Length Error"
    for i in range(size):
        flist = mergelist[i][:-2].split('\t')
        t_category = []
        p_result = p_results[i]

        for fpath in flist: 
            cellname = fpath.split('/')[-1].split('.')[0]
            if '-' in cellname:
                t_category.append(int(cellname.split('-')[0]))
            else:
                t_category.append(100)

        for truth in t_category:
            if truth < 100:
                p += 1
            else:
                n += 1

        for truth, prediction in zip(t_category, p_result):
            if prediction[0] < 100:
                if truth == prediction[0]:
                    tp += 1
                else:
                    if truth < 100:
                        wp += 1                     
                    else:
                        fp += 1

    return tp, wp, fp, p, n



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


if __name__ == "__main__":
    """
    just an example of how to use our code.
    You may want to write your own script with your datasets and other customizations.
    """
    init_logger()
    r=10
    attacks = ["df", "cumul", "kfingerprinting", "knn"]

    tp_total, wp_total, fp_total, p_total, n_total = [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]    
    for i in range(2,18):
        target = join(BASE_DIR, "simulation/results/noise_glue/noise_glue_"+str(i) + "/")
        mergelistf = join(target, "list")
        with open(mergelistf,'r') as f:
            mergelist = f.readlines()        
        for j,attack in enumerate(attacks):
            attackpath = join(BASE_DIR, 'CDSB_series', attack) 
            p_results = read_p_results(attackpath, target.split("/")[-2], 9900//i)
            logger.info("results of %s by WFattack: %s"%(target.split('/')[-2], attack))  
    
            tp, wp, fp, p, n = evaluation_func(mergelist, p_results, 9900//i)
            print('total\ttp={} wp={} fp={} p={} n={}'.format(tp, wp, fp, p, n))
            precision = tp*n / (tp*n+wp*n+r*p*fp)
            TPR = tp/p
            print("precision = %f\tTPR = %f"%(precision, TPR))

            tp_total[j] += tp
            wp_total[j] += wp
            fp_total[j] += fp
            p_total[j] += p
            n_total[j] += n

    for i,attack in enumerate(attacks):
        precision = tp_total[i]*n_total[i] / (tp_total[i]*n_total[i]+wp_total[i]*n_total[i]+r*p_total[i]*fp_total[i])
        TPR = tp_total[i]/p_total[i]
        
        logger.info("results of all by WFattack: %s"%attack)  
        print('total\ttp={} wp={} fp={} p={} n={}'.format(tp_total[i], wp_total[i], fp_total[i], p_total[i], n_total[i]))
        print("precision = %f\tTPR = %f"%(precision, TPR))

    #just for example
