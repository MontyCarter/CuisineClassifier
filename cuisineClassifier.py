import sklearn
from vectorizeRecipes import toSklearnFormat
from pprint import pprint

trainFile = 'srcData/train.json'
testFile = 'srcData/test.json'

recData = toSklearnFormat(trainFile=trainFile, testFile=testFile)

pprint(recData)
