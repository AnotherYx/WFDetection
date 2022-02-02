import subprocess
from os.path import join
original = "/store1/WebsiteFingerprinting/defenses/results/"
split = "/store1/WebsiteFingerprinting/attacks/xgboost/scores/"

targets = [
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
"mergepad_evaluation_16_200_10_random/",
]

# targets = [
# "mergepad_0622_2205/",
# "mergepad_0622_2206/",
# "mergepad_0622_2207/",
# "mergepad_0622_2208/",
# "mergepad_0622_2209/",
# "mergepad_0622_2210/",
# "mergepad_0622_2211/",
# "mergepad_0622_2212/",
# "mergepad_0622_2213/",
# "mergepad_0622_2214/",
# "mergepad_0622_2215/",
# "mergepad_0622_2216/",
# "mergepad_0622_2217/",
# "mergepad_0622_2218/",
# "mergepad_0622_2219/",
# ] 

splitpath = "/store1/WebsiteFingerprinting/attacks/split/"


for target in targets:
	a = join(original, target)
	b = join(split, target, "splitresult.txt")
	print(target)
	cmd = "python " + splitpath + "split-random.py " + a + " -split "+ b
	# print(cmd)
	# exit(0)
	subprocess.call(cmd, shell= True)