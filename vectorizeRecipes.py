#!/usr/bin/python

import json
from pprint import pprint

trainFile = 'srcData/train.json'

def readJson(filename):
    with open(filename) as f:
        return json.load(f)

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
    for recipe in recipes:
        for ingred in recipe['ingredients']:
            ingreds.add(ingred)
    ingreds = sorted(list(ingreds))
    ingredMap = dict()
    print("There are " + str(len(ingreds)) + " ingredients.")
    for x in range(len(ingreds)):
        # Create a dict - 'dim' is dimension number for ingredient
        # Later we'll add usage count
        ingredMap[ingreds[x]] = {'dim':x, 'usageCount':0}
    for recipe in recipes:
        for ingred in recipe['ingredients']:
            ingredMap[ingred]['usageCount'] += 1
    return ingredMap

# Conceptually identical to genIngredMap
def genCuisineMap(recipes):
    cuisines = set()
    for recipe in recipes:
        cuisines.add(recipe['cuisine'])
    cuisines = sorted(list(cuisines))
    cuisineMap = dict()
    print("There are " + str(len(cuisines)) + " cuisines.")
    for x in range(len(cuisines)):
        cuisineMap[cuisines[x]] = {'dim':x, 'usageCount':0}
    for recipe in recipes:
        cuisineMap[recipe['cuisine']]['usageCount'] += 1
    return cuisineMap
    
def genVectorRepresentation(recipes):
    iMap = genIngredMap(recipes)
    cMap = genCuisineMap(recipes)
    vectors = list()
    for recipe in recipes:
        # Create a list of zeros of the same size as the number
        # of possible ingredients, plus 1 extra slot for the label
        vector = [0] * (1 + len(iMap))
        # Add the label in the first slot
        vector[0] = cMap[recipe['cuisine']]['dim']
        # Set the dimension to 1 for each present ingredient
        for ingred in recipe['ingredients']:
            vector[iMap[ingred]['dim']] = 1
        vectors.append(vector)
    return vectors
        

recipes = readJson(trainFile)
print("There are " + str(len(recipes)) + " recipes.")
vectors = genVectorRepresentation(recipes)
print(vectors)
