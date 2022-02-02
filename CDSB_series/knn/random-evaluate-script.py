import subprocess
import multiprocessing as mp

import argparse
import os
from os.path import abspath, dirname,join

# Directories
BASE_DIR = abspath(join(dirname(__file__)))



def parse_arguments():

	parser = argparse.ArgumentParser(description='KNN ATTACK.')

	parser.add_argument('-dirty',
						metavar='<dirty trainset path>',
						help='Path to the directory of the trainset')
	parser.add_argument('-clean',
						metavar='<clean trainset path>',
						help='Path to the directory of the trainset')
	parser.add_argument('-test',
						metavar='<trainset path>',
						help='Path to the directory of the testset')
	# Parse arguments
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_arguments()

	flearner_head = join(BASE_DIR, "flearner-head")
	flearner_other = join(BASE_DIR, "flearner-other")
	if not os.path.exists(flearner_head):
		cmd = "g++ " + join(BASE_DIR, "flearner-head.cpp") + " -o " + flearner_head
		subprocess.call(cmd, shell= True)
	if not os.path.exists(flearner_other):
		cmd = "g++ " + join(BASE_DIR, "flearner-other.cpp") + " -o " + flearner_other
		subprocess.call(cmd, shell= True)


	logname_head = join(BASE_DIR, "log/", args.test.split('/')[-2]+"-head.log")
	logname_other = join(BASE_DIR, "log/", args.test.split('/')[-2]+"-other.log")

    #head
	cmd1 = "python " + join(BASE_DIR,"fextractor.py") + " " + args.dirty + " -mode train"
	subprocess.call(cmd1, shell= True)
	cmd2 = "python " + join(BASE_DIR,"fextractor.py") + " " + args.test + " -mode test"
	subprocess.call(cmd2, shell= True)
	cmd3 = "python " + join(BASE_DIR,"gen-list.py") + " " + join(BASE_DIR,"options-kNN.txt") + " " + args.dirty + " " + args.test
	subprocess.call(cmd3, shell= True)
	cmd4 = join(BASE_DIR,"flearner-head") + " " + join(BASE_DIR,"options-kNN.txt") + " " +\
	 args.dirty + " " + args.test + " >> " + logname_head
	subprocess.call(cmd4, shell= True)

	#other
	cmd5 = "python " + join(BASE_DIR,"fextractor.py") + " " + args.clean + " -mode train"
	subprocess.call(cmd5, shell= True)
	cmd6 = "python " + join(BASE_DIR,"gen-list.py") + " " + join(BASE_DIR,"options-kNN.txt") + " " + args.clean + " " + args.test
	subprocess.call(cmd6, shell= True)
	cmd7 = join(BASE_DIR,"flearner-other") + " " + join(BASE_DIR,"options-kNN.txt") + " "+\
	 args.clean + " " + args.test + " >> " + logname_other
	subprocess.call(cmd7, shell= True)


	cmd3 = "python " + join(BASE_DIR, "log-script.py") + " " + join(BASE_DIR, "log/", args.test.split('/')[-2])
	subprocess.call(cmd3, shell= True)
