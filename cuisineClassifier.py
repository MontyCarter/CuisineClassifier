import sklearn
import vectorizeRecipes
from sklearn import svm
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from pprint import pprint

trainFile = 'srcData/train.json'
testFile = 'srcData/test.json'


#Get dataset object in sklearn format
recData = vectorizeRecipes.toSklearnFormat(trainFile=trainFile, testFile=testFile)
#Print statistics
vectorizeRecipes.printSklearnDatasetStats(recData)
trainSize = int(len(recData['data'])/10) 
predictSize = 500 
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
print()
print("Training Statistics:")
print("    Training Examples Used For Training: " + str(trainSize))
print("    Training Examples Used For Testing:  " + str(predictSize))
#Ensure we have enough training examples
assert(trainSize+predictSize <= len(recData['data']))
#Instantiate the svm classifier
#clf = svm.SVC(gamma=0.001, C=100.)
clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5)
#Train the svm classifier
clf.fit(recData['data'][:trainSize], recData['target'][:trainSize])
print()
print("Best parameters set found on development set:")
print(clf.best_params_)
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
