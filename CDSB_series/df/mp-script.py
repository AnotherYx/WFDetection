import subprocess
from os.path import join
prefix = "../split/results/"
targets = [
"mergepad_0701_2018/",
"mergepad_0701_2019/",
"mergepad_0701_2020/",
"mergepad_0701_2021/",
"mergepad_0701_2022/",
"mergepad_0701_2023/",
"mergepad_0701_2024/",
"mergepad_0701_2025/",
"mergepad_0701_2026/",
"mergepad_0701_2027/",
"mergepad_0701_2028/",
"mergepad_0701_2029/",
"mergepad_0701_2030/",
"mergepad_0701_2031/",
"mergepad_0701_2032/",
]

# print("Making datasets...")

# for target in targets:
	
# 	extract_cmd = "python3 makedata.py -mode test " + target
# 	subprocess.call(extract_cmd, shell =True)


for target in targets:
	target = join(prefix, target)
	print("Evaluating {}".format(target))
	fname = target.split('/')[-2]
	ex_cmd  = "python3 makedata.py "+ target + " -mode test"
	# head_cmd  = "python3 evaluate.py -m ./models/ranpad2_0610_2057_norm.h5 -p ./results/"+ fname + "_head.npy"
	head_cmd  = "python3 evaluate.py -m ./models/attacktrain.h5 -p ./results/"+ fname + "_head.npy"
	other_cmd  = "python3 evaluate.py -m ./models/attacktrain.h5 -p ./results/"+ fname + "_other.npy"
	subprocess.call(ex_cmd, shell =True)
	subprocess.call(head_cmd, shell =True)
	subprocess.call(other_cmd, shell =True)
	print("\n")