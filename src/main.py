# for training
import argparse
import trainer

parser = argparse.ArgumentParser()
parser.add_argument('--trainText',
                    help='path to training text')
parser.add_argument('--trainLabel',
                    help='path to training label')                
parser.add_argument('--validText',
                    help='path to validation text')
parser.add_argument('--validLabel',
                    help='path to validation label')
parser.add_argument('--testText',
                    help='path to test text')
parser.add_argument('--testLabel',
                    help='path to test label')
parser.add_argument('--dumpTo', 
                    help='path to dir where results will be dumped. This should be dir path such as ../result/')
parser.add_argument('--lmPath',
                    help='path to pretrained a language model. When omitted, the LM is initialized randomly.')
parser.add_argument('--sampling', action='store_true',
                    help='switch on sampling segmentation while training.')
parser.add_argument('--useChar', action='store_true',
                    help='use character embedding to make word embedding.')
parser.add_argument('--useWord', action='store_true',
                    help='use word embedding in addition to vectors composed of char-embeddings.')
parser.add_argument('--useCache', action='store_true',
                    help='use cache embedding for unfixed vocaburary.')
parser.add_argument('--uniTrain',action='store_true',
                    help='switch on updating the lanugage model by the sampled tokenization.')
parser.add_argument('--segMode', default='uni', choices=['uni','sp','char','dict'],
                    help='select segmentation mode corresponding to table Table1 of the paper [WIP].')
parser.add_argument('--clType', default='dan', choices=['lstm','bilstm','dan','attn'],
                    help='select the type of an encoder for text classification.')
parser.add_argument('--spSize', type=int, default=8000,
                    help='piece size, or cache size.')
parser.add_argument('--gpuId', type=int, default=-1)
parser.add_argument('--dumpModels', action='store_true',
                    help='the model scoring the best score on validation set is dumped.')
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
                     gpuid=args.gpuId)
tr.train()
