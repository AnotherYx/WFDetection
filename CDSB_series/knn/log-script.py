import subprocess
import multiprocessing as mp
import argparse
from os.path import abspath, dirname, join

# Directories
BASE_DIR = abspath(join(dirname(__file__)))
# name = "./log/"
# targets = [ #glue 
# "ranpad2_0706_0829",
# "ranpad2_0706_0830",
# # "ranpad2_0706_0831",
# # "ranpad2_0706_0832",
# # "ranpad2_0706_0833",
# # "ranpad2_0706_0834",
# # "ranpad2_0706_0835/",
# # "ranpad2_0706_0836/",
# # "ranpad2_0706_0837/",
# # "ranpad2_0706_0838/",
# # "ranpad2_0706_0839/",
# # "ranpad2_0706_0840/",
# # "ranpad2_0706_0841/",
# # "ranpad2_0706_0842/",
# # "ranpad2_0706_0843/",
# ]




# def work(target):
# 	log1= name + target+'-head.log'
# 	log2= name + target+'-other.log'
# 	cmd1 = "python3 parselog.py " + log1
# 	cmd2 = "python3 parselog.py " + log2
# 	subprocess.call(cmd1, shell= True)
# 	subprocess.call(cmd2, shell= True)

# pool = mp.Pool(5)
# pool.map(work, targets)


def parse_arguments():

    parser = argparse.ArgumentParser(description='DF ATTACK.')

    parser.add_argument('logname',
                        metavar='<log path>',
                        help='Path of the WF log')

    # Parse arguments
    args = parser.parse_args()
    return args

if __name__ == '__main__':
	args = parse_arguments()

	cmd1 = "python3 " + join(BASE_DIR, "parselog.py") + " " + args.logname +'-head.log'
	cmd2 = "python3 " + join(BASE_DIR, "parselog.py") + " " + args.logname +'-other.log'
	subprocess.call(cmd1, shell= True)
	subprocess.call(cmd2, shell= True)