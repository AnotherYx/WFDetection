import subprocess
from os.path import join

original = "../../defenses/results/"
split = "../xgboost/scores/"

# undefended
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

#glue
# targets = [
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

targets = [
	"mergepad_evaluation_16_200_10_random/",
]

for target in targets:
	a = join(original, target)
	b = join(split, target, "splitresult.txt")

	cmd = "python3 split-base-rate.py " + a + " -split "+ b
	# print(cmd)
	# exit(0)
	subprocess.call(cmd, shell= True)
