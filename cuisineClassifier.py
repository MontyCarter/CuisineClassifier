from mlAlgos import *
from pprint import pprint
from lib import *
from time import time
import sys

recData = unserialize('fullDataset.dat')
#Print dataset statistics
printSklearnDatasetStats(recData)
#Start algorithm
start = time()
if sys.argv[1] == "SVC":
    crossValidateSvmSVC()
elif sys.argv[1] == "SVR":
    crossValidateSvmSVR()
elif sys.argv[1] == "SVCPoly":
    crossValidateSvmSVCPoly()
elif sys.argv[1] == "SVRPoly":
    crossValidateSvmSVRPoly()
else:
    print("Must pass algo to run.  Choose from:")
    print("SVC, SVR, SVCPoly, SVRPoly")
    print("Example: python3 cuisineClassifier.py SVC")
    exit(-1)
end = time()
print("Time Elapsed: " + str(end-start))


