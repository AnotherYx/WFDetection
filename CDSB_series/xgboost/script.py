import subprocess
from os.path import join

name = "./scores/"
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
for target in targets:
	target = name + target
	fname = target.split('/')[-2]
	cmd = "python3 getsplit-base-rate.py "+ target + " -k " + join("../decision/results/", fname+".npy")
	# cmd = "python3 getsplit-base-rate.py "+ target 
	subprocess.call(cmd, shell= True)