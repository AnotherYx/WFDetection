import subprocess
from os.path import join


prefix = "scores/"
# name = "scores/mergepad_0131_"
targets = [
"ranpad2_0610_1951/",
"ranpad2_0610_1952/",
"ranpad2_0610_1953/",
"ranpad2_0610_1954/",
"ranpad2_0610_1955/",
"ranpad2_0610_1956/",
"ranpad2_0610_1958/",
"ranpad2_0610_1959/",
"ranpad2_0610_2001/",
"ranpad2_0610_2004/",
"ranpad2_0610_2006/",
"ranpad2_0610_2008/",
"ranpad2_0610_2010/",
"ranpad2_0610_2013/",
"ranpad2_0610_2016/",

] 


for target in targets:
	target = join(prefix, target)
	kdir = join("../decision/results/", target.split('/')[-2]+ ".npy")
	cmd = "python3 getsplit-base-rate.py "+ target + " -k "+ kdir
	subprocess.call(cmd, shell= True)
