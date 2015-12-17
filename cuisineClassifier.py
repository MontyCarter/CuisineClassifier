from mlAlgos import *
from pprint import pprint
from lib import *
from time import time
import sys

recData = unserialize('fullDataset.dat')
#Print dataset statistics
printSklearnDatasetStats(recData)
sys.stdout.flush()
#Start algorithm
start = time()
aborted = False
if sys.argv[1] == "SVC":
    aborted = crossValidateSvmSVC()
elif sys.argv[1] == "SVR":
    aborted = crossValidateSvmSVR()
elif sys.argv[1] == "SVCPoly":
    aborted = crossValidateSvmSVCPoly()
elif sys.argv[1] == "SVRPoly":
    aborted = crossValidateSvmSVRPoly()
elif sys.argv[1] == "RandFrst":
    aborted = crossValidateRandFrst()
elif sys.argv[1] == 'GaussianBayes':
    aborted = crossValidateGaussianBayes()
else:
    print("Must pass algo to run.  Choose from:")
    print("SVC, SVR, SVCPoly, SVRPoly, RandFrst")
    print("Example: python3 cuisineClassifier.py SVC")
    exit(-1)
end = time()
print("Total Elapsed Time: " + str(end-start))
if aborted:
    exit(-1)
