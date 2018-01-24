import configparser
import argparse
import os
import pdb
from law_pre_master.net.data_formatter import check

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c')
parser.add_argument('--gpu', '-g')
args = parser.parse_args()

configFilePath = args.config
if configFilePath is None:
    print("python *.py\t--config/-c\tconfigfile")
usegpu = True
# if args.use is None:
#    print("python *.py\t--use/-u\tcpu/gpu")
if args.gpu is None:
    usegpu = False
else:
    usegpu = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

config = configparser.RawConfigParser()
config.read(configFilePath)

import json

#inpath = '/disk/mysql/law_data/final_data/'
#outpath = '/disk/mysql/law_data/goodData/'
inpath = 'data/'
outpath = 'goodData/'

fileList = os.listdir(inpath)
#fout = open(outpath + 'goodData.json', 'w')
for file in fileList:
    fin = open(inpath + file, 'r')
    fout = open(outpath + file, 'w')
    line = fin.readline()
    while line:
        line = json.loads(line)
        if check(line, config):
            print(json.dumps(line, config), file = fout)
        line = fin.readline()
    fout.close()