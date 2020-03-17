from unigram import UnigramLM
import sys
import config
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input')
parser.add_argument('-o', '--output')
parser.add_argument('-e', '--epoch', default=500, type=int)
args = parser.parse_args()

if args.epoch < 100:
    print('--epoch should be more than 100')
    exit()

uni = UnigramLM()
data = [line.strip() for line in open(args.input)]
uni.train(data, args.epoch, args.output)
print('dump trained model as %s'%args.output)

