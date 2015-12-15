from lib import *

trainFile = 'srcData/train.json'
testFile = 'srcData/test.json'

#Get dataset object in sklearn format
recData = toSklearnFormat(trainFile=trainFile, testFile=testFile)
#Serialize full data to speed load times
serialize(recData, 'fullDataset.dat')
#Write 10 data folds
writeKfoldSets(recData['data'], recData['target'], 10)
