import sys
import os
import numpy as np

query = sys.argv[2] if len(sys.argv)==3 else ''

maxEpoch = 30
windowSize = 3 #epoch

def show(path):
    data = [list(map(float, line.strip().split())) for line in open(path)]
    if len(data)==0:
        # 実験が終わってないものは表示しない
        # (すぐに止めた実験)
        return

    maxV = 0
    maxT = 0
    aveT = 0
    
    ts = []

    for line in data:
        v,t = line
        if maxV < v:
            maxV = v
            maxT = t
        if maxV==v and maxT<t:
            maxT = t
        aveT += t
        ts.append(t)

    ave = aveT/len(data) if len(data)>0 else 0
    var = np.var(ts)
    method = path.split('/')[-1].replace('_eval.log','')
    
    # max average point
    windowTs = [[t for v,t in data[i:i+windowSize]] for i in range(len(ts)-windowSize)]
    ave = max([np.average(ts) for ts in windowTs])

    print('%d\t%.2f\t%.2f\t%.2f\t%s'%(len(data), maxV*100, maxT*100, ave*100, method))
    #print(path.split('/')[-1].replace('_eval.log',''), len(data), maxV,maxT, ave)
    #print(path.split('/')[-1].replace('_eval.log',''),maxT,ave,var)
    #print(maxT)

root = sys.argv[1]
pathes = [root+path for path in os.listdir(root) if '_eval' in path]

# search query
pathes = [path for path in pathes if query in path]

# sort path
char = [path for path in pathes if 'char_' in path]

## sp
sp = [path for path in pathes if 'sp_' in path]
sp_tmp = []

sp_tmp += sorted([s for s in sp if 'useWord' in s and 'useChar' not in s])
sp_tmp += sorted([s for s in sp if 'useWord' not in s and 'useChar' in s])
sp_tmp += sorted([s for s in sp if 'useWord' in s and 'useChar' in s])

sp = sp_tmp

## uni
uni = [path for path in pathes if 'uni_' in path]
uni_tmp = []

uni_tmp += sorted([u for u in uni if 'unitrain' not in u and '_nosamp' in u])
uni_tmp += sorted([u for u in uni if 'unitrain' not in u and '_samp' in u])
uni_tmp += sorted([u for u in uni if 'unitrain' in u])
uni = uni_tmp

## dict
dt = [path for path in pathes if 'dict_' in path]

pathes = char+sp+uni+dt

print(root)
print('epoch\tvalid\ttest\tave\tmethod')
for path in pathes:
    show(path)

