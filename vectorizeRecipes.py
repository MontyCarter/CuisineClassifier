import json
from numpy import array

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
    dataset['data'] = array(data)
    #Add the target labels to the dataset object
    dataset['target'] = array(target)
    #Add the test data to the dataset object
    dataset['test'] = array(test)
    return dataset

def printSklearnDatasetStats(dataset):
    print()
    print("Total dataset statistics:")
    print("    Training cuisine count:              " + str(len(dataset['target_names'])))
    print("    Training ingredient count:           " + str(len(dataset['feature_names'])))
    print("    Training recipes:                    " + str(len(dataset['data'])))
    print("    Test recipes:                        " + str(len(dataset['test'])))
    print("    Unknown Ingredients in Test Recipes: " + str(dataset['unknownTestIngredCount']))

if __name__ == '__main__':
  trainFile = 'srcData/train.json'
  testFile = 'srcData/test.json'
  #dataset = toSklearnFormat(trainFile, testFile)
  #print(dataset['data'])

