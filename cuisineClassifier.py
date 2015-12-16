from lib import *
from pprint import pprint

trainSize = 1500
predictSize = 1000


#recData = unserialize('fullDataset.dat')
#Print statistics
#printSklearnDatasetStats(recData)
print()
print("Training Statistics:")
print("    Training Examples Used For Training: " + str(trainSize))
print("    Training Examples Used For Testing:  " + str(predictSize))





#Ensure we have enough training examples
#assert(trainSize+predictSize <= len(recData['data']))
crossValidateSVM(trainSize, predictSize)


