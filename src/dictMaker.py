import pickle
import sys
from collections import defaultdict

def getDict(data):
    # 頻度辞書を作る
    char2id = {}
    id2char = {}

    charCountDict = defaultdict(lambda:0)
    for line in data:
        for c in line:
            charCountDict[c] += 1

    charCountDict['<BOS>'] = len(data)
    charCountDict['<EOS>'] = len(data)
    charCountDict['<UNK>'] = 1

    # 頻度辞書を降順にしてid化
    for k,v in reversed(sorted(charCountDict.items(), key=lambda x:x[1])):
        char2id[k] = len(char2id)
        id2char[char2id[k]] = k

    return char2id, id2char

if __name__ == '__main__':
    path = sys.argv[1]
    data = [line.strip() for line in open(path) if line.strip()]
    char2id, id2char = getDict(data)

    dictPath = path.replace('.txt','_dict.pickle')
    pickle.dump((char2id,id2char), open(dictPath,'wb'))
