import sklearn
import vectorizeRecipes
from sklearn import svm
from sklearn import cross_validation
from pprint import pprint


#--------------Begin Skeleton--------------------

def serialize(dataset):
    #Turn dataset into serialized string
    return serializedDataset

def unserialize(serializedDataset):
    #Turn serialized dataset into dataset
    return dataset

def writeKfoldSets(fullTrainingSet, foldNum):
    #Use kfold to split into foldNum (train,test) sets
    count = 0
    for train,test in folds:
        stest = serialize(test)
        #write stest to test_<count>.dat
        strain = serialize(train)
        #write strain to train_<count>.dat
        count += 1


#  pythonScript - one python script per ML algo we use
#  foldNum - the current fold number
def crossValidateSubprocess(pythonScript, foldNum, hyperparams):
    #Recreate input filenames
    stestFile = test_<foldNum>.dat
    strainFile = train_<foldNum>.dat
    #Use subprocess to call pythonScript, passing stestFile, strainFile, and hyperparam list
    #pythonScript should write a result file (see below for details)
    

def collectResults(expectedResultFiles):
    resultList = list()
    algoParamCombos = set()
    avgComboResults = list()
    #Loop through each expected result file
        #Add (algo, foldNum, hyperparams, accuracy) to resultList
        #Add (algo, hyperparams) to alogParamCombos
    #For curCombo in algoParamCombos:
        curComboCount = 0
        curComboAcc = 0.
        #for curResult in resultList:
            #if curCombo matches curResult (matching means curResult is one of the folds for curCombo)
                curComboCount += 1
                curComboAcc += curCombo.Accuracy
        avgComboResults.append([curCombo.algo, curCombo.hyperparams, curComboAcc/curComboCount])
    return avgComboResults

#pythonScript algo requirements
#  Params:
#    first command line arg is fold number
#    remaining command line args are hyperparameters
#  Calculates:
#    accuracy when training with train_<foldNum>.dat
#  Output:
#    Written to <algoName>_<foldNum>_<param1>_<param2>_<param...>.res
#    Contents of this file should simply be testCorrect/testTotal
#    for example, svm.SVC_3_0.001_100.res would have the single line "0.75332" in it
    
#---------------End Skeleton---------------------




trainFile = 'srcData/train.json'
testFile = 'srcData/test.json'

trainSize = 1500
predictSize = 1000

#Get dataset object in sklearn format
recData = vectorizeRecipes.toSklearnFormat(trainFile=trainFile, testFile=testFile)
#Print statistics
vectorizeRecipes.printSklearnDatasetStats(recData)
print()
print("Training Statistics:")
print("    Training Examples Used For Training: " + str(trainSize))
print("    Training Examples Used For Testing:  " + str(predictSize))
#Ensure we have enough training examples
assert(trainSize+predictSize <= len(recData['data']))
#Instantiate the svm classifier
clf = svm.SVC(gamma=0.001, C=100.)
#Train the svm classifier
clf.fit(recData['data'][:trainSize], recData['target'][:trainSize])
#Predict the labels of the last predictSize training examples
pred = clf.predict(recData['data'][-1*predictSize:])
#Compare predicted labels with actual labels, maintaining a count of matches
count = 0
for x in range(predictSize):
    idx = -1*x - 1
    if(pred[idx] == recData['target'][idx]):
        count += 1

print()
print("Accurately predicted:")
print(str(count) + "/" + str(predictSize))
