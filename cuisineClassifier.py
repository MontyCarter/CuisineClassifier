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
clf.fit(recData['data'][:1500], recData['target'][:1500])
#Predict the last x labels
predictSize = 50
pred = clf.predict(recData['data'][-1*predictSize:])
#Print the last 5 predicted labels, and the last 5 real labels
count = 0
for x in range(predictSize):
    idx = -1*x - 1
    if(pred[idx] == recData['target'][idx]):
        count += 1

print("Accurately predicted:")
print(str(count) + "/" + str(predictSize))
