from lib import *
from sklearn import svm
from time import time

##########################################
# svm SVC
##########################################
def crossValidateSvmSVC():
    Cs = [1., 10., 100., 1000.]
    gammas = [.0001, .001, .01, .1]
    valueLabelPairs = [(Cs,'C'),(gammas,'gamma')]
    crossValidate(svmSVCFold, valueLabelPairs)

def svmSVCFold(foldNum, paramCombo):
    startTime = time()
    trainData,testData = getFoldData(foldNum)
    #Instantiate the svm classifier
    clf = svm.SVC(**paramCombo)
    #Train the svm classifier
    clf.fit(trainData['data'], trainData['target'])
    #Predict the labels of the last predictSize training examples
    return predict(clf, paramCombo, foldNum, startTime, testData)

##########################################
# svm SVR
##########################################
def crossValidateSvmSVR():
    Cs = [1., 10., 100., 1000.]
    gammas = [.0001, .001, .01, .1]
    valueLabelPairs = [(Cs,'C'),(gammas,'gamma')]
    crossValidate(svmSVRFold, valueLabelPairs)

def svmSVRFold(foldNum, paramCombo):
    startTime = time()
    trainData,testData = getFoldData(foldNum)
    #Instantiate the svm classifier
    clf = svm.SVR(**paramCombo)
    #Train the svm classifier
    clf.fit(trainData['data'], trainData['target'])
    #Predict the labels of the last predictSize training examples
    return predict(clf, paramCombo, foldNum, startTime, testData)

##########################################
# svm SVC poly
##########################################
def crossValidateSvmSVCPoly():
    Cs = [1., 10., 100., 1000., 10000.]
    degree = [2,3,4]
    coef0 = [0., 1.]
    valueLabelPairs = [(Cs,'C'),(degree,'degree'),(coef0, 'coef0')]
    crossValidate(svmSVCPolyFold, valueLabelPairs)

def svmSVCPolyFold(foldNum, paramCombo):
    startTime = time()
    trainData,testData = getFoldData(foldNum)
    #Instantiate the svm classifier
    clf = svm.SVC(**paramCombo, kernel='poly')
    #Train the svm classifier
    clf.fit(trainData['data'], trainData['target'])
    #Predict the labels of the last predictSize training examples
    return predict(clf, paramCombo, foldNum, startTime, testData)

##########################################
# svm SVR poly
##########################################
def crossValidateSvmSVRPoly():
    Cs = [1., 10., 100., 1000., 10000.]
    degree = [2,3,4]
    coef0 = [0., 1.]
    valueLabelPairs = [(Cs,'C'),(degree,'degree'),(coef0, 'coef0')]
    crossValidate(svmSVRPolyFold, valueLabelPairs)

def svmSVCPolyFold(foldNum, paramCombo):
    startTime = time()
    trainData,testData = getFoldData(foldNum)
    #Instantiate the svm classifier
    clf = svm.SVR(**paramCombo, kernel='poly')
    #Train the svm classifier
    clf.fit(trainData['data'], trainData['target'])
    #Predict the labels of the last predictSize training examples
    return predict(clf, paramCombo, foldNum, startTime, testData)

