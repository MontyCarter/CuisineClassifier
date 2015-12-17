import json
import pickle
import multiprocessing
import mlAlgos
import sys
from multiprocessing.pool import ThreadPool
from numpy import array
from pprint import pprint
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from time import time
from time import sleep



##########################################
# Original data format functions
##########################################
# Reads a json file into python data structures
def readJson(filename):
    with open(filename) as f:
        return json.load(f)

# Data looks like
#  {
# "id": 24717,
# "cuisine": "indian",
# "ingredients": [
#     "tumeric",
#     "vegetable stock",
#     "tomatoes",
#     "garam masala",
#     "naan",
#     "red lentils",
#     "red chili peppers",
#     "onions",
#     "spinach",
#     "sweet potatoes"
# ]
# },

# Takes recipes json data, gets set of all possible ingredients,
#  converts this set to a sorted list
#  and assigns each a number (representing a dimension) to each ingredient
#
# Returns a dictionary where first dim is indexed by ingredient, and second
#  dimension is indexed by either 'dim' or 'usageCount'.
#  dim is the feature vector dimension representing inclusion of the ingredient
#  usageCount is the number of recipes using this ingredient
def genIngredMap(recipes):
    ingreds = set()
    # Get a set of all ingredients in use
    for recipe in recipes:
        for ingred in recipe['ingredients']:
            ingreds.add(ingred)
    # Convert set to sorted list
    ingreds = sorted(list(ingreds))
    # Create a dict, indexed by ingredient name
    # This allows us to look up the vector dimension for a given
    # ingredient
    ingredMap = dict()
    for x in range(len(ingreds)):
        # Create a dict - 'dim' is dimension number for ingredient
        # Later we'll add usage count 
        # (i just thought this info was interesting - not super useful)
        ingredMap[ingreds[x]] = {'dim':x, 'usageCount':0}
    # Add the usage counts to the map
    for recipe in recipes:
        for ingred in recipe['ingredients']:
            ingredMap[ingred]['usageCount'] += 1
    return ingredMap

# Conceptually identical to genIngredMap
def genCuisineMap(recipes):
    cuisines = set()
    for recipe in recipes:
        try:
            cuisines.add(recipe['cuisine'])
        except:
            print(recipe)
            exit()
    cuisines = sorted(list(cuisines))
    cuisineMap = dict()
    for x in range(len(cuisines)):
        cuisineMap[cuisines[x]] = {'dim':x, 'usageCount':0}
    for recipe in recipes:
        cuisineMap[recipe['cuisine']]['usageCount'] += 1
    return cuisineMap
    
def genVectorRepresentation(recipes,iMap,cMap):
    vectors = list()
    seenUnknownIngred = False
    unknownIngredCount = 0
    # Loop through each recipe, vectorize it, and add it to vectors
    for recipe in recipes:
        # Create a list of zeros of the same size as the number
        # of possible ingredients, plus 1 extra slot for label (if train data)
        if cMap:
            vector = [0.] * (1 + len(iMap))
        else:
            vector = [0.] * len(iMap)
        # Set the dimension to 1 for each present ingredient
        for ingred in recipe['ingredients']:
            if ingred  in iMap:
                vector[iMap[ingred]['dim']] = 1.
            else:
                if not seenUnknownIngred:
                    pass
                    #print("Unknown ingredient seen in test data:")
                #print("  " + ingred)
                seenUnknownIngred = True
                unknownIngredCount += 1
        # Add the label in the last slot
        if cMap:
            vector[-1] = cMap[recipe['cuisine']]['dim']
        vectors.append(vector)
    return vectors, unknownIngredCount

##########################################
# sklearn format functions
##########################################

def toSklearnFormat(trainFile, testFile):
    #Read in json files
    train = readJson(trainFile)
    test = readJson(testFile)
    # Grab the ingredient and cuisine maps 
    # (cuisine map does not exist if input file is test data)
    iMap = genIngredMap(train)
    cMap = genCuisineMap(train)
    #Convert json into a list of vector examples, and maps from ingredients
    #  to their indices and cuisines into their label numbers
    trainVectors, unknownTrain = genVectorRepresentation(train, iMap, cMap)
    testVectors, unknownTest = genVectorRepresentation(test, iMap, None)
    #Create an object to store everything
    dataset = dict()
    dataset['unknownTestIngredCount'] = unknownTest
    #Add the training data target names (label names)
    dataset['target_names'] = array(list(cMap.keys()))
    #Add the feature names (ingredient names) (applies to test & train)
    dataset['feature_names'] = array(list(iMap.keys()))
    target = list()
    data = list()
    test = list()
    #Split the features from the label, and add each to a separate list
    for traindatum in trainVectors:
        data.append(traindatum[:-1])
        target.append(traindatum[-1])
    for testdatum in testVectors:
        test.append(testdatum)
    #Add the examples to the dataset object
    dataset['data'] = csr_matrix(array(data))
    #Add the target labels to the dataset object
    dataset['target'] = array(target)
    #Add the test data to the dataset object
    dataset['test'] = csr_matrix(array(test))
    return dataset

def printSklearnDatasetStats(dataset):
    print()
    print("Total dataset statistics:")
    print("    Training cuisine count:              " + str(len(dataset['target_names'])))
    print("    Training ingredient count:           " + str(len(dataset['feature_names'])))
    print("    Training recipes:                    " + str(dataset['data'].shape[0]))
    print("    Test recipes:                        " + str(dataset['test'].shape[0]))
    print("    Unknown Ingredients in Test Recipes: " + str(dataset['unknownTestIngredCount']))

##########################################
# serialization functions
##########################################

def serialize(dataset, filename):
    pickle.dump(dataset, open(filename,'wb'), protocol=-1)

def unserialize(serializedFile):
    return pickle.load(open(serializedFile, 'rb'))

def writeKfoldSets(fullTrainingData, fullTrainingTarget, foldNum):
    #Use kfold to split into foldNum (train,test) sets
    count = 0
    kf = KFold(fullTrainingData.shape[0], n_folds=10)
    for train,test in kf:
        print(train,test)
        trainSet = dict()
        testSet = dict()
        trainSet['data'] = csr_matrix(fullTrainingData[train])
        testSet['data'] = csr_matrix(fullTrainingData[test])
        trainSet['target'] = fullTrainingTarget[train]
        testSet['target'] = fullTrainingTarget[test]
        trainFile = "train_" + str(count) + ".dat"
        testFile = "test_" + str(count) + ".dat"
        trainSet = serialize(trainSet, trainFile)
        testSet = serialize(testSet, testFile)
        count += 1
        # trainSet={'data':exampleVectors, 'target':listOflabels}

##########################################
# cross validation functions
##########################################

class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

def traverse(a, level, l, res):
    if level == len(a):
        return res.append(l)
    for x in a[level]:
        ll = list(l)
        ll.append(x)
        traverse(a, level+1, ll, res)

def genCombos(*valueNamePairs):
    combos = list()
    vals = [pair[0] for pair in valueNamePairs]
    labels = [pair[1] for pair in valueNamePairs]
    traverse(vals, 0, [], combos)
    dictCombos = list()
    for combo in combos:
        dictCombo = dict()
        for x in range(len(combo)):
            dictCombo[labels[x]] = combo[x]
        dictCombos.append(hashabledict(dictCombo))
    return dictCombos

# resList like [(paramCombo1, foldNum1, correctTestCount11, totalTestCount11),
#               (paramCombo1, foldNum2, correctTestCount12, totalTestCount12),
#               ...]
def averageFolds(resList):
    comboSets = set()
    results = list()
    for res in resList:
        comboSets.add(res[0])
    pprint(comboSets)
    for combo in comboSets:
        correct = total = totalTime = 0
        for res in resList:
            if res[0] == combo:
                totalTime += res[2]
                correct += res[3]
                total += res[4]
        results.append([combo, totalTime, correct, total, 100*correct/total])
    return results

def crossValidate(function, paramValuesLabelPairs):
    global results
    num_cpus = multiprocessing.cpu_count()
    #num_cpus=13
    p = ThreadPool(processes=num_cpus)
    paramCombos = genCombos(*paramValuesLabelPairs)
    rs = []
    results = []
    try:
        for paramCombo in paramCombos:
            for foldNum in range(10):
                r = p.apply_async(function,
                                  args=(foldNum, paramCombo))
                #sleep(10)
                rs.append(r)
        finishedCount = 0
        length = len(paramCombos)*10
        sys.stdout.write("  Validation Progress: " + str(round(100*finishedCount/length,2)) + "% \r")
        sys.stdout.flush()
        for r in rs:
            r.wait()
            finishedCount += 1
            sys.stdout.write("  Validation Progress: " + str(round(100*finishedCount/length,2)) + "% \r")
            sys.stdout.flush()
            output = r.get()
            results.append(output)
    except KeyboardInterrupt:
        p.terminate()
        p.join()
    else:
        p.close()
        p.join()
    #Results elements like: [combo, correct, total, 100*correct/total]
    avgResults = sorted(averageFolds(results), key=lambda x: x[-1])
    pprint(results)
    pprint(avgResults)

def getFoldData(foldNum):
    trainFile = "train_" + str(foldNum) + ".dat"
    testFile = "test_" + str(foldNum) + ".dat"
    trainData = unserialize(trainFile)
    testData = unserialize(testFile)
    return trainData,testData

def predict(classifier, paramCombo, foldNum, startTime, testData):
    pred = classifier.predict(testData['data'])
    #Compare predicted labels with actual labels, maintaining a count of matches
    correctCount = 0
    for x in range(len(testData['target'])):
        if(pred[x] == testData['target'][x]):
            correctCount += 1
    return (paramCombo, 
            foldNum, 
            time() - startTime, 
            correctCount, 
            len(testData['target']))

#  pythonScript - one python script per ML algo we use
#  foldNum - the current fold number
#  hyperparams - must be a list! (of hyper params)
def processFold(pythonScript, foldNum, hyperparams):
    trainFile = "train_" + str(foldNum) + ".dat"
    testFile = "test_" + str(foldNum) + ".dat"
    cmd = ['python3', pythonScript, foldNum] + hyperparams
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err  = p.communicate()
    return float(out)

def collectResults(expectedResultFiles):
    pass
    #resultList = list()
    #algoParamCombos = set()
    #avgComboResults = list()
    #Loop through each expected result file
        #Add (algo, foldNum, hyperparams, accuracy) to resultList
        #Add (algo, hyperparams) to alogParamCombos
    #For curCombo in algoParamCombos:
        #curComboCount = 0
        #curComboAcc = 0.
        #for curResult in resultList:
            #if curCombo matches curResult (matching means curResult is one of the folds for curCombo)
                #curComboCount += 1
                #curComboAcc += curCombo.Accuracy
        #avgComboResults.append([curCombo.algo, curCombo.hyperparams, curComboAcc/curComboCount])
    #return avgComboResults

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
