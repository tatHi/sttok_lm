class SegDataset:
    def __init__(self, segPath):
        data = [line.strip().split() for line in open(segPath) if line.strip()]
        self.segData = []
        self.setSegData(data)

    def setSegData(self,data):
        for line in data:
            segLine = []
            for w in line:
                segLine += [0]*(len(w)-1)+[1]
            self.segData.append(segLine)
