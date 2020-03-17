import numpy as np

class Logger:
    def __init__(self, path):
        self.path = path
        f = open(self.path,'w')
        f.close()

    def write(self, line):
        #if type(line)==np.ndarray:
        #    line = line.round(4)
        f = open(self.path, 'a')
        f.write(str(line))
        f.write('\n')
        f.close()

