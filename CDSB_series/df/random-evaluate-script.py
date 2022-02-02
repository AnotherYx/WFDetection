import subprocess
from os.path import join
import argparse
from os.path import abspath, dirname

# Directories
BASE_DIR = abspath(join(dirname(__file__)))

# prefix = "../split/results/"
# targets = [  #undefended
# "mergepad_0701_2018/",
# "mergepad_0701_2019/",
# "mergepad_0701_2020/",
# "mergepad_0701_2021/",
# "mergepad_0701_2022/",
# "mergepad_0701_2023/",
# "mergepad_0701_2024/",
# "mergepad_0701_2025/",
# "mergepad_0701_2026/",
# "mergepad_0701_2027/",
# "mergepad_0701_2028/",
# "mergepad_0701_2029/",
# "mergepad_0701_2030/",
# "mergepad_0701_2031/",
# "mergepad_0701_2032/",
# ]

#prefix = "../split/randomresults/"#with decision
# prefix = "../split/results/"#without decision
# targets = [ #glue 
# "ranpad2_0706_0829/",
# "ranpad2_0706_0830/",
# "ranpad2_0706_0831/",
# "ranpad2_0706_0832/",
# "ranpad2_0706_0833/",
# "ranpad2_0706_0834/",
# "ranpad2_0706_0835/",
# "ranpad2_0706_0836/",
# "ranpad2_0706_0837/",
# "ranpad2_0706_0838/",
# "ranpad2_0706_0839/",
# "ranpad2_0706_0840/",
# "ranpad2_0706_0841/",
# "ranpad2_0706_0842/",
# "ranpad2_0706_0843/",
# ]

# for target in targets:
# 	target  = join(prefix, target) 
# 	print("process {}".format(target))
# 	#cmd1 = "python3 random-evaluate.py -m models/ranpad2_0610_2057_norm.h5 -mode head -p "+ target
# 	cmd1 = "python3 random-evaluate.py -m models/attacktrain.h5 -mode head -p "+ target
# 	cmd2 = "python3 random-evaluate.py -m models/ranpad2_0610_2057_norm.h5 -mode other -p "+ target
# 	# cmd2 = "python3 random-evaluate.py -m clean-trained-kf.pkl -o tor_leaf.npy -mode other -p "+ target
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
    parser.add_argument('-t',
                        metavar='<target>',
                        help='Target to test')    
    # Parse arguments
    args = parser.parse_args()
    return args

if __name__ == '__main__':
	args = parse_arguments()
	cmd1 = "python3 " + join(BASE_DIR,"random-evaluate.py") + " -m " + args.d + " -mode head -p " + args.t
	cmd2 = "python3 " + join(BASE_DIR,"random-evaluate.py") + " -m " + args.c + " -mode other -p " + args.t
	subprocess.call(cmd1, shell= True)
	subprocess.call(cmd2, shell= True)
    # target = "../split/randomresults/mergepad_evaluation_16_200_10_random/"
    # cmdtest1 = "python3 " + join(BASE_DIR,"random-evaluate.py") + " -m " + "./models/attacktrain_clean.h5" + " -mode head -p " + target
    # subprocess.call(cmdtest1, shell= True)
    # cmdtest2 = "python3 " + join(BASE_DIR,"random-evaluate.py") + " -m " + "./models/attacktrain_clean.h5" + " -mode other -p " + target
    # subprocess.call(cmdtest2, shell= True)