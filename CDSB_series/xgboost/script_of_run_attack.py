import subprocess
from os.path import join

#undefended 
# tests = [  
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

#GLUE
# tests = [
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

tests = [
	"mergepad_evaluation_16_200_10_random/",
]

prefix = "../../defenses/results/"


# train = "../../defenses/results/mergepad_0701_2040_4000_b1/" #undefended
train = "../../defenses/results/mergepad_splittrain_2_4000_1_fix/" #defended with glue noise

# with decision
# for test in tests:
# 	test = join(prefix, test)
# 	kdir = join("../decision/results/", test.split('/')[-2]+ ".npy")
# 	cmd = "python3 run_attack.py -train "+train + " -test "+ test + " -mode decision" + " -kdir " + kdir
# 	subprocess.call(cmd, shell= True)


#without decision
for test in tests:
	test = join(prefix, test)
	cmd = "python3 run_attack.py -train "+train + " -test "+ test 
	subprocess.call(cmd, shell= True)