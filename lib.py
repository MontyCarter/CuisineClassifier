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

#This file provides the framework for easily running different algorithms
#  in scikit-learn in a parallel fashion.  Also includes functions for 
#  consuming and formatting original, raw data.

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

#Converts json recipes to vectors, according to ingredient map and cuisine map    
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

#Reads original data in, converts to earlier format (which is no longer in use)
# and finally converts it into a format suitable for scikit-learn
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

#Print statistics on dataset
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
#Writes python object to disk
def serialize(dataset, filename):
    pickle.dump(dataset, open(filename,'wb'), protocol=-1)

#Reads python object from disk
def unserialize(serializedFile):
    return pickle.load(open(serializedFile, 'rb'))

#Splits input data into folds, and writes folds to disk
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

# Makes dictionaries hashable (for use with set())
class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

# Generates combinations of possible parameter values
def traverse(a, level, l, res):
    if level == len(a):
        return res.append(l)
    for x in a[level]:
        ll = list(l)
        ll.append(x)
        traverse(a, level+1, ll, res)

# Driver for traverse - generates combinations of possible parameter values
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

# Collects all folds for each paramCombo and aggregates the results
#
# resList like:
#     [(paramCombo1, foldNum1, runTime1, correctTestCount11, totalTestCount11),
#      (paramCombo1, foldNum2, runTime2, correctTestCount12, totalTestCount12),
#      ...]
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

# Main cross validate function.  Should be called from a "crossValidate<Algo>()"
#   function.  This function provides the mechanics for parallelization and 
#   results aggregation/printing
#
# Return value indicates whether the process was aborted (True==aborted).
#   On abort, caller should call exit(-1) so outstanding threads terminate
def crossValidate(function, paramValuesLabelPairs):
    aborted = True
    finished = False
    #Set up threading
    num_cpus = multiprocessing.cpu_count()
    p = ThreadPool(processes=num_cpus)
    paramCombos = genCombos(*paramValuesLabelPairs)
    rs = []
    results = []
    # Setup progress "bar"
    length = len(paramCombos)*10
    print("Validating ", int(length/10), " parameter combinations...")
    sys.stdout.flush()
    try:
        for paramCombo in paramCombos:
            for foldNum in range(10):
                #Queue algo on each paramCombo/fold
                r = p.apply_async(function, args=(foldNum, paramCombo))
                rs.append(r)
        finishedCount = 0
        progressStr = "  Validation Progress: " 
        progressStr += str(round(100*finishedCount/length,2)) + "% \r"
        sys.stdout.write(progressStr)
        sys.stdout.flush()
        #Wait for queued tasks to finish, print progress along the way
        for r in rs:
            r.wait()
            finishedCount += 1
            progressStr = "  Validation Progress: " 
            progressStr += str(round(100*finishedCount/length,2)) + "% \r"
            sys.stdout.write(progressStr)
            sys.stdout.flush()
            output = r.get()
            results.append(output)
        print()
    except KeyboardInterrupt:
        #Print partial results and exit
        print("Aborted - printing individual fold results:")
        pprint(results)
        avgResults = sorted(averageFolds(results), key=lambda x: x[-1])
        pprint(avgResults)
        sys.stdout.flush()
        return aborted
    else:
        p.close()
        p.join()
    #print results
    avgResults = sorted(averageFolds(results), key=lambda x: x[-1])
    print("Individual fold results:")
    pprint(results)
    print("Aggregate Cross Validation results:")
    pprint(avgResults)
    return finished

#Loads train and test data for a given fold number
def getFoldData(foldNum):
    trainFile = "train_" + str(foldNum) + ".dat"
    testFile = "test_" + str(foldNum) + ".dat"
    trainData = unserialize(trainFile)
    testData = unserialize(testFile)
    return trainData,testData

#Runs prediction on the testData, and returns a tuple of all relevant results
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
