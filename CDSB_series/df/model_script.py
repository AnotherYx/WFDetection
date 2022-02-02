import subprocess
import argparse
import const as ct
from os.path import join, abspath, dirname

# Directories
BASE_DIR = abspath(join(dirname(__file__)))


def parse_arguments():
    parser = argparse.ArgumentParser(description='Auto training.')

    parser.add_argument('t',
                        metavar='<trainset path>',
                        help='Path to the trainset.')
    parser.add_argument('type',
                        metavar='<model type>',
                        help='train a clean or dirty model',
                        default = "None")

    # Parse arguments
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    dirname = args.t.split('/')[-2]
    #print(dirname)
    extract_cmd = "python " + BASE_DIR + "/makedata.py " + args.t + " -mode train " + args.type
    npyfile = ct.outputdir + dirname + "_" + args.type + ".npy"
    #print(npyfile)
    train_cmd = "python " + BASE_DIR + "/2-main.py " + npyfile + " " + args.type

    #print(extract_cmd)
    subprocess.call(extract_cmd, shell =True)
    #print(train_cmd)
    subprocess.call(train_cmd, shell = True)

