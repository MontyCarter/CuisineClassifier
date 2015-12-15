import sklearn
from vectorizeRecipes import toSklearnFormat
from sklearn import svm
from pprint import pprint

trainFile = 'srcData/train.json'
testFile = 'srcData/test.json'

#Get dataset object in sklearn format
recData = toSklearnFormat(trainFile=trainFile, testFile=testFile)
#Instantiate the svm classifier
clf = svm.SVC(gamma=0.001, C=100.)
#Train the svm classifier (saving 5 as test data)
clf.fit(recData['data'][:500], recData['target'][:500])
#Predict the last 5 labels
pred = clf.predict(recData['data'][-5:])
#Print the last 5 predicted labels, and the last 5 real labels
print(pred)
print(recData['target'][-5:])
