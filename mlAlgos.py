from lib import *
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from time import time
from sklearn import naive_bayes
from sklearn import tree
from sklearn import neighbors

##########################################
# svm LinearSVC 
##########################################
def crossValidateLinearSVC():
    print("Using: Linear SVC")
    sys.stdout.flush()
    valueLabelPairs = [([1., 10., 100., 1000.], 'C')]
    return crossValidate(LinearSVCFold, valueLabelPairs)

def LinearSVCFold(foldNum, paramCombo):
    startTime = time()
    trainData,testData = getFoldData(foldNum)
    clf = svm.LinearSVC(**paramCombo)
    clf.fit(trainData['data'], trainData['target'])
    return predict(clf, paramCombo, foldNum, startTime, testData)

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
    valueLabelPairs = [([50,75,100,150,200],    'n_estimators'),
                       ([None],                 'max_depth'),
                       (['gini', 'entropy'],    'criterion'),
                       (["auto"],               'max_features'),
                       ([1,2,4,8],              'min_samples_split'),
                       ([1,2,4,8],              'min_samples_leaf'),
                       ([None,2,4,8],           'max_leaf_nodes')]
    return crossValidate(randFrstFold, valueLabelPairs)

def randFrstFold(foldNum, paramCombo):
    startTime = time()
    trainData,testData = getFoldData(foldNum)
    clf = RandomForestClassifier(**paramCombo)
    clf.fit(trainData['data'], trainData['target'])
    return predict(clf, paramCombo, foldNum, startTime, testData)

##########################################
# Gaussian Naive Bayes 
##########################################
def crossValidateGaussianBayes():
    print("Using: Gaussian Naive Bayes")
    sys.stdout.flush()
    valueLabelPairs = None
    crossValidate(GaussianBayesFold, valueLabelPairs)

def GaussianBayesFold(foldNum, paramCombo):
    startTime = time()
    trainData,testData = getFoldData(foldNum)
    clf = naive_bayes.MultinomialNB()
    clf.fit(trainData['data'], trainData['target'])
    return predict(clf, paramCombo, foldNum, startTime, testData)

##########################################
# Decision Tree 
##########################################
def crossValidateDecisionTree():
    print("Using: Decision Tree")
    sys.stdout.flush()
    valueLabelPairs = None
    crossValidate(DecisionTreeFold, valueLabelPairs)

def DecisionTreeFold(foldNum, paramCombo):
    startTime = time()
    trainData,testData = getFoldData(foldNum)
    clf = tree.DecisionTreeClassifier() 
    clf.fit(trainData['data'], trainData['target'])
    return predict(clf, paramCombo, foldNum, startTime, testData)

##########################################
# Decision Tree 
##########################################
def crossValidateAdaboost():
    print("Using: Adaboost")
    sys.stdout.flush()
    valueLabelPairs = [([svm.SVC(C=10., gamma=0.1)], 'base_estimator'),
                       ([5,10],                      'n_estimators'),
                       (['SAMME'],                   'algorithm'),
                       ([1., 10.],                   'learning_rate')]
    crossValidate(adaboostFold, valueLabelPairs)

def adaboostFold(foldNum, paramCombo):
    startTime = time()
    trainData,testData = getFoldData(foldNum)
    clf = AdaBoostClassifier(**paramCombo)
    clf.fit(trainData['data'], trainData['target'])
    return predict(clf, paramCombo, foldNum, startTime, testData)

##########################################
# KNN 
##########################################
def crossValidateKNN():
    print("Using: KNN")
    sys.stdout.flush()
    valueLabelPairs = [(list(range(1, 20)), 'n_neighbors'),
                       (['uniform', 'distance'], 'weights'),
    		       ([2], 'p')]
    return crossValidate(KNNFold, valueLabelPairs)

def KNNFold(foldNum, paramCombo):
    startTime = time()
    trainData,testData = getFoldData(foldNum)
    clf = neighbors.KNeighborsClassifier(**paramCombo)
    clf.fit(trainData['data'], trainData['target'])
    return predict(clf, paramCombo, foldNum, startTime, testData)
