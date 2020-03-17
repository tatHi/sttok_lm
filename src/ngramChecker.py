import sys
path = sys.argv[1]
n = int(sys.argv[2])

data = [line.strip() for line in open(path)]
ngram = set()
charSize = 0

for line in data:
    for i in range(len(line)-n+1):
        w = line[i:i+n]
        ngram.add(w)
    charSize += len(line)
print(len(ngram)/charSize)
