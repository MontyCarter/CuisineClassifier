from lib import *
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from time import time

##########################################
# svm SVC
##########################################
def crossValidateSvmSVC():
    print("Using: SVC with RBF Kernel")
    sys.stdout.flush()
    valueLabelPairs = [([1., 10., 100., 1000.], 'C'),
                       ([.0001, .001, .01, .1], 'gamma')]
    return crossValidate(svmSVCFold, valueLabelPairs)

def svmSVCFold(foldNum, paramCombo):
    startTime = time()
    trainData,testData = getFoldData(foldNum)
    clf = svm.SVC(**paramCombo)
    clf.fit(trainData['data'], trainData['target'])
    return predict(clf, paramCombo, foldNum, startTime, testData)

##########################################
# svm SVR
##########################################
def crossValidateSvmSVR():
    print("Using: SVR with RBF Kernel")
    sys.stdout.flush()
    valueLabelPairs = [([1., 10., 100., 1000.], 'C'),
                       ([.0001, .001, .01, .1], 'gamma')]
    return crossValidate(svmSVRFold, valueLabelPairs)

def svmSVRFold(foldNum, paramCombo):
    startTime = time()
    trainData,testData = getFoldData(foldNum)
    clf = svm.SVR(**paramCombo)
    clf.fit(trainData['data'], trainData['target'])
    return predict(clf, paramCombo, foldNum, startTime, testData)

##########################################
# svm SVC poly
##########################################
def crossValidateSvmSVCPoly():
    print("Using: SVC with Polynomial Kernel")
    sys.stdout.flush()
    valueLabelPairs = [([1., 10., 100., 1000., 10000.], 'C'),
                       ([2,3,4],                        'degree'),
                       ([0., 1.],                       'coef0')]
    return crossValidate(svmSVCPolyFold, valueLabelPairs)

def svmSVCPolyFold(foldNum, paramCombo):
    startTime = time()
    trainData,testData = getFoldData(foldNum)
    clf = svm.SVC(**paramCombo, kernel='poly')
    clf.fit(trainData['data'], trainData['target'])
    return predict(clf, paramCombo, foldNum, startTime, testData)

##########################################
# svm SVR poly
##########################################
def crossValidateSvmSVRPoly():
    print("Using: SVR with Polynomial Kernel")
    sys.stdout.flush()
    valueLabelPairs = [([1., 10., 100., 1000., 10000.], 'C'),
                       ([2,3,4],                        'degree'),
                       ([0., 1.],                       'coef0')]
    return crossValidate(svmSVRPolyFold, valueLabelPairs)

def svmSVRPolyFold(foldNum, paramCombo):
    startTime = time()
    trainData,testData = getFoldData(foldNum)
    clf = svm.SVR(**paramCombo, kernel='poly')
    clf.fit(trainData['data'], trainData['target'])
    return predict(clf, paramCombo, foldNum, startTime, testData)

##########################################
# Random Forest Classifier
##########################################
def crossValidateRandFrst():
    print("Using: Random Forest Classifier")
    sys.stdout.flush()
    valueLabelPairs = [([5,10,15,20,50],       'n_estimators'),
                       ([None,2,4,8,16,32,64], 'max_depth'),
                       (['gini', 'entropy'],   'criterion'),
                       (["auto"],              'max_features'),
                       ([1,2,4,8],             'min_samples_split'),
                       ([1,2,4,8],             'min_samples_leaf'),
                       ([None,2,4,8],          'max_leaf_nodes')]
    return crossValidate(randFrstFold, valueLabelPairs)

def randFrstFold(foldNum, paramCombo):
    startTime = time()
    trainData,testData = getFoldData(foldNum)
    clf = RandomForestClassifier(**paramCombo)
    clf.fit(trainData['data'], trainData['target'])
    return predict(clf, paramCombo, foldNum, startTime, testData)
