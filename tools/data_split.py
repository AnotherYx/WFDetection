import random 
import glob
import os
from shutil import copyfile
from os.path import join, abspath, dirname, pardir

# Directories
BASE_DIR = abspath(join(dirname(__file__), pardir))

def randomly_gen():
	list_names = glob.glob(join(BASE_DIR,'data/ds19/*.cell'))	
	random.shuffle(list_names)

	attacktrain_splittrain = join(BASE_DIR, "data/attacktrain_splittrain")
	if not os.path.exists(attacktrain_splittrain):
		os.makedirs(attacktrain_splittrain)

	attacktrain = join(BASE_DIR, "data/attacktrain")
	if not os.path.exists(attacktrain):
		os.makedirs(attacktrain)
	with open(join(BASE_DIR, "data/attacktrain.txt"),"w") as f:
		for i in range(0,9000):
			target = join(attacktrain, list_names[i].split("/")[-1])
			copyfile(list_names[i], target)
			a_s_target = join(attacktrain_splittrain, list_names[i].split("/")[-1])
			copyfile(list_names[i], a_s_target)
			f.write(list_names[i]+'\t\n')

	splittrain = join(BASE_DIR, "data/splittrain")
	if not os.path.exists(splittrain):
		os.makedirs(splittrain)
	with open(join(BASE_DIR, "data/splittrain.txt"),"w") as f:
		for i in range(9000,11000):
			target = join(splittrain, list_names[i].split("/")[-1])
			copyfile(list_names[i], target)
			a_s_target = join(attacktrain_splittrain, list_names[i].split("/")[-1])
			copyfile(list_names[i], a_s_target)
			f.write(list_names[i]+'\t\n')

	evaluation = join(BASE_DIR, "data/evaluation")
	if not os.path.exists(evaluation):
		os.makedirs(evaluation)
	with open(join(BASE_DIR, "data/evaluation.txt"),"w") as f:
		for i in range(11000,20000):
			target = join(evaluation, list_names[i].split("/")[-1])
			copyfile(list_names[i], target)
			f.write(list_names[i]+'\t\n')	

if __name__ == '__main__':
	randomly_gen()

