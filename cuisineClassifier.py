from mlAlgos import *
from pprint import pprint
from lib import *

trainSize = 1500
predictSize = 1000


recData = unserialize('fullDataset.dat')
#Print statistics
printSklearnDatasetStats(recData)

#Ensure we have enough training examples
#assert(trainSize+predictSize <= len(recData['data']))
crossValidateSvmSVCPoly()


