import sklearn
import vectorizeRecipes
from sklearn import svm
from sklearn import cross_validation
from pprint import pprint

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
