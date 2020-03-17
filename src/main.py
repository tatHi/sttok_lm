# for training
import argparse
import trainer

parser = argparse.ArgumentParser()
parser.add_argument('--trainText')
parser.add_argument('--trainLabel')
parser.add_argument('--validText')
parser.add_argument('--validLabel')
parser.add_argument('--testText')
parser.add_argument('--testLabel')
parser.add_argument('--dumpTo', help='should be dir path such as ../result/')
parser.add_argument('--lmPath')
parser.add_argument('--classSize', type=int)
parser.add_argument('--sampling',action='store_true')
parser.add_argument('--useChar', action='store_true')
parser.add_argument('--useWord', action='store_true')
parser.add_argument('--useCache', action='store_true')
parser.add_argument('--uniTrain',action='store_true')
parser.add_argument('--segMode', default='uni', choices=['uni','sp','char','dict'])
parser.add_argument('--clType', default='dan', choices=['lstm','bilstm','dan','attn'])
parser.add_argument('--smoothingHyp', type=int, default=1)
parser.add_argument('--spSize', type=int, default=8000)
parser.add_argument('--gpuId',type=int)
parser.add_argument('--dumpModels', action='store_true')
args = parser.parse_args()


textPathes   = [args.trainText,
                args.validText,
                args.testText]

labelPathes  = [args.trainLabel,
                args.validLabel,
                args.testLabel]

loggerPathes = [args.dumpTo+'/eval.log',
                args.dumpTo+'/loss.log']

dumpPath     = args.dumpTo + '/validMax.models'

tr = trainer.Trainer(textPathes=textPathes,
                     labelPathes=labelPathes,
                     loggerPathes=loggerPathes,
                     dumpPath=dumpPath,
                     segMode=args.segMode,
                     useChar=args.useChar,
                     useWord=args.useWord,
                     useCache=args.useCache,
                     spSize=args.spSize,
                     clType=args.clType,
                     sampling=args.sampling,
                     uniTrain=args.uniTrain,
                     modelPath=args.lmPath,
                     classSize=args.classSize,
                     gpuid=args.gpuId,
                     smoothingHyp=args.smoothingHyp)
tr.train()
