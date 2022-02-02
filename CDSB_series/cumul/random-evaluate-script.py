import subprocess
from os.path import join
import argparse
from os.path import abspath, dirname

# Directories
BASE_DIR = abspath(join(dirname(__file__)))

# #prefix = "../split/randomresults/"#with decision
# prefix = "../split/results/"#without decision
# targets = [ #glue 
# # "ranpad2_0706_0829/",
# # "ranpad2_0706_0830/",
# # "ranpad2_0706_0831/",
# # "ranpad2_0706_0832/",
# # "ranpad2_0706_0833/",
# # "ranpad2_0706_0834/",
# # "ranpad2_0706_0835/",
# # "ranpad2_0706_0836/",
# # "ranpad2_0706_0837/",
# # "ranpad2_0706_0838/",
# # "ranpad2_0706_0839/",
# # "ranpad2_0706_0840/",
# # "ranpad2_0706_0841/",
# # "ranpad2_0706_0842/",
# "ranpad2_0706_0843/",
# ]


# for target in targets:
# 	target = join(prefix, target)
# 	cmd1 = "python3 random-evaluate.py -m ./models/dirty.pkl -o ./results/dirty.npy -mode head -p "+ target
# 	cmd2 = "python3 random-evaluate.py -m ./models/clean.pkl -o ./results/clean.npy -mode other -p "+ target
# 	subprocess.call(cmd1, shell= True)
# 	subprocess.call(cmd2, shell= True)
# 	# print("\n\n\n\n\n\n\n")

def parse_arguments():

    parser = argparse.ArgumentParser(description='DF ATTACK.')

    parser.add_argument('-d',
                        metavar='<dirtymodel path>',
                        help='Path to the directory of the dirtymodel')
    parser.add_argument('-c',
                        metavar='<cleanmodel path>',
                        help='Path to the directory of the cleanmodel')  
    parser.add_argument('-od',
                        metavar='<feature path>',
                        help='Path to the directory of the extracted features')   
    parser.add_argument('-oc',
                        metavar='<feature path>',
                        help='Path to the directory of the extracted features')  
    parser.add_argument('-t',
                        metavar='<target>',
                        help='Target to test')    
    # Parse arguments
    args = parser.parse_args()
    return args

if __name__ == '__main__':
	args = parse_arguments()
	cmd1 = "python3 " + join(BASE_DIR,"random-evaluate.py") + " -m " + args.d + " -o " + args.od + " -mode head -p " + args.t
	cmd2 = "python3 " + join(BASE_DIR,"random-evaluate.py") + " -m " + args.c + " -o " + args.oc + " -mode other -p " + args.t
	subprocess.call(cmd1, shell= True)
	subprocess.call(cmd2, shell= True)
    # target = "../split/randomresults/mergepad_evaluation_16_200_10_random/"
    # cmdtest1 = "python3 " + join(BASE_DIR,"random-evaluate.py") + " -m " + "./models/attacktrain_clean.pkl" + " -o " + "./results/attacktrain_clean.npy" + " -mode head -p " + target
    # subprocess.call(cmdtest1, shell= True)
    # cmdtest2 = "python3 " + join(BASE_DIR,"random-evaluate.py") + " -m " + "./models/attacktrain_clean.pkl" + " -o " + "./results/attacktrain_clean.npy" + " -mode other -p " + target
    # subprocess.call(cmdtest2, shell= True)
