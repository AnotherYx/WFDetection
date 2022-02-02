import subprocess
from os.path import join
import argparse
from os.path import abspath, dirname

# Directories
BASE_DIR = abspath(join(dirname(__file__)))

# train = "../../defenses/results/mergepad_0701_2039_9000_m20/"
# # train = "../../defense/results/ranpad2_0611_1051/"
# prefix = "../../defenses/results/"
# targets = [
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
# # targets = [
# # "ranpad2_0610_1951/",
# # "ranpad2_0610_1952/",
# # "ranpad2_0610_1953/",
# # "ranpad2_0610_1954/",
# # "ranpad2_0610_1955/",
# # "ranpad2_0610_1956/",
# # "ranpad2_0610_1958/",
# # "ranpad2_0610_1959/",
# # "ranpad2_0610_2001/",
# # "ranpad2_0610_2004/",
# # "ranpad2_0610_2006/",
# # "ranpad2_0610_2008/",
# # "ranpad2_0610_2010/",
# # "ranpad2_0610_2013/",
# # "ranpad2_0610_2016/",
# # ]

# for i,target in enumerate(targets):
# 	target = join(prefix, target)
# 	i = i+2 
# 	cmd = "python3 run_attack.py -train "+ train + " -test "\
# 	+ target +" -num "+ str(i)
# 	# print(cmd)
# 	subprocess.call(cmd, shell= True)

def parse_arguments():

	parser = argparse.ArgumentParser(description='DF ATTACK.')

	parser.add_argument('-train',
						metavar='<dirty trainset path>',
						help='Path to the directory of the trainset')
	parser.add_argument('-target',
						metavar='<target path>',
						help='Path to the directory of the taeget set')
	# Parse arguments
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_arguments()
	cmd = "python3 " + join(BASE_DIR, "run_attack.py") + " -train "+ args.train + " -test " + args.target
	# print(cmd)
	subprocess.call(cmd, shell= True)