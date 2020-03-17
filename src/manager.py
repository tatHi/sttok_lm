import trainer
import argparse
import config

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',choices=['news_ja','ntcir_ja','ntcir_ch','twitter_ja','twitter_en','hotel_ch','ntcir_en'])
parser.add_argument('--sampling',action='store_true')
parser.add_argument('--useChar', action='store_true')
parser.add_argument('--useWord', action='store_true')
parser.add_argument('--uniTrain',action='store_true')
parser.add_argument('--segMode',choices=['uni','sp','char','dict'])
parser.add_argument('--clType',choices=['lstm','bilstm','dan','attn'])
parser.add_argument('--hyp', type=int, default=-1)
parser.add_argument('--smoothingHyp', type=int, default=1)
parser.add_argument('--preEpoch', default=500)
parser.add_argument('--spSize', type=int, default=8000)
parser.add_argument('--gpuId',type=int)
parser.add_argument('--outputPathAffix',default='')
parser.add_argument('--reduceAfterSampling', action='store_true')
parser.add_argument('--dumpModels', default=False, action='store_true')
args = parser.parse_args()

# settings
'''
dataset = 'ntcir_ja' #'twitter_ja'#'ntcir_ja'
embed = 'char' #'subword' #'char'
sampling = True
uniTrain = True
segMode = 'uni' #'uni' # 'sp' # 'char'

clType = 'lstm' #'bilstm' # 'dan'
'''
dataset = args.dataset
sampling = args.sampling
uniTrain = args.uniTrain
segMode = args.segMode
useChar = args.useChar
useWord = args.useWord
useCache = (segMode=='uni' and useWord)
clType = args.clType
reduceAfterSampling = args.reduceAfterSampling
affix = args.outputPathAffix 
hyp = '' if args.hyp==-1 or args.preEpoch=='jieba' or args.preEpoch=='mecab' or args.preEpoch=='original' else '_hyp%d'%args.hyp

# assert for imposible settings
cando = True
if segMode=='uni':
    if useWord and not useChar:
        cando = False
elif segMode=='sp':
    if uniTrain:
        cando = False
elif segMode=='char':
    if sampling or useChar or uniTrain or not useWord:
        cando = False
        print('segMode=char needs useWord option')
assert cando, 'impossible settings'

if dataset=='ntcir_ja' or dataset=='ntcir_ch' or dataset=='ntcir_en':
    classSize = 3 
elif dataset=='twitter_ja':
    classSize = 5
elif dataset=='twitter_en' or dataset=='hotel_ch':
    classSize = 2
elif dataset=='news_ja':
    classSize = 9

spPathRoot = '/m_%d.model'%args.spSize

if dataset == 'ntcir_ja':
    textRoot = '../data/ntcir/raw/ntcir_ja_%s_text.txt'
    labelRoot = '../data/ntcir/raw/ntcir_ja_%s_label.txt'
    dictPath = '../data/ntcir/raw/ntcir_ja_train_text_dict.pickle'
    spModelPath = '../data/ntcir/raw'+spPathRoot
elif dataset == 'ntcir_ch':
    root = '../data/ntcir6_ch'
    textRoot = root+'/ntcir_ch_%s_text.txt'
    labelRoot = root+'/ntcir_ch_%s_label.txt'
    dictPath = root+'/ntcir_ch_train_text_dict.pickle'
    spModelPath = root+spPathRoot
elif dataset == 'ntcir_en':
    root = '../data/ntcir_en'
    textRoot = root+'/ntcir_en_%s_text.txt'
    labelRoot = root+'/ntcir_en_%s_label.txt'
    dictPath = root+'/ntcir_en_train_text_dict.pickle'
    spModelPath = root+spPathRoot
elif dataset == 'twitter_ja':
    root = '../data/%s'%(dataset)
    rootP = root+'/'+dataset
    textRoot = rootP+'_%s_text.txt'
    labelRoot = rootP+'_%s_label.txt'
    dictPath = rootP+'_train_text_dict.pickle'
    spModelPath = root+spPathRoot
elif dataset == 'twitter_en':
    root = '../data/%s'%(dataset)
    rootP = root+'/'+dataset
    textRoot = rootP+'_%s_text.txt'
    labelRoot = rootP+'_%s_label.txt'
    dictPath = rootP+'_train_text_dict.pickle'
    spModelPath = root+spPathRoot
elif dataset == 'hotel_ch':
    root = '../data/hotel_ch'
    textRoot = root+'/hotel_ch_%s_text.txt'
    labelRoot = root+'/hotel_ch_%s_label.txt'
    dictPath = root+'/hotel_ch_train_text_dict.pickle'
    spModelPath = root+spPathRoot
elif dataset == 'news_ja':
    root = '../data/news_ja'
    textRoot = root+'/news_ja_%s_text.txt'
    labelRoot = root+'/news_ja_%s_label.txt'
    dictPath = root+'/news_ja_train_text_dict.pickle'
    spModelPath = root+spPathRoot

# if segMode is dict, use segmented corpus
if segMode == 'dict':
    if '_ja' in dataset:
        tail = '_mecab'
    elif '_ch' in dataset:
        tail = '_jieba'
    else:
        tail = ''
    textRoot = textRoot.replace('text.txt','text%s.txt'%tail) 

uniModelPath = (textRoot%'train').replace('text.txt','unigram_bigram%s_%s.model'%(hyp,str(args.preEpoch)))

if useChar and useWord:
    embed = '_useChar_useWord'
elif useChar:
    embed = '_useChar'
elif useWord:
    embed = '_useWord'

# start time
from datetime import datetime
now = datetime.now().strftime("%Y%m%d%H%M%S")

logFileRoot = '%s%s_%s%s%s'%(segMode,embed,'samp' if sampling else 'nosamp','_unitrain' if uniTrain else '',
                                '_sh%s'%(str(args.smoothingHyp) if segMode=='uni' else ''))
logFileRoot = affix+logFileRoot
loggerRoot = '../result/%s_%s/%s'%(dataset, clType,logFileRoot)+'_%s_'+now+'.log'

# auto
textPathes = [textRoot%ty for ty in ['train','valid','test']]
labelPathes = [labelRoot%ty for ty in ['train','valid','test']]
loggerPathes = [loggerRoot%ty for ty in ['eval','loss']]

# dump file path
if args.dumpModels:
    #dumpPath = '../result/%s_%s/%s_'%(dataset, clType,logFileRoot)+now+'_epoch%d.models'
    dumpPath = '../result/%s_%s/%s_'%(dataset, clType,logFileRoot)+now+'_validMax.models'
else:
    dumpPath = None

if segMode=='uni':
    modelPath = uniModelPath
elif segMode=='sp':
    modelPath = spModelPath
else:
    modelPath = None

print(modelPath)

tr = trainer.Trainer(textPathes=textPathes,
                     labelPathes=labelPathes,
                     dictPath=dictPath,
                     loggerPathes=loggerPathes,
                     dumpPath=dumpPath,
                     segMode=segMode,
                     useChar=useChar,
                     useWord=useWord,
                     useCache=useCache,
                     spSize=args.spSize,
                     clType=clType,
                     sampling=sampling,
                     uniTrain=uniTrain,
                     modelPath=modelPath, 
                     classSize=classSize,
                     gpuid=args.gpuId,
                     smoothingHyp=args.smoothingHyp)
tr.train()
