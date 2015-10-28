#!/usr/bin/python

import json
from pprint import pprint

trainFile = 'srcData/train.json'

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
    print("There are " + str(len(ingreds)) + " ingredients.")
    for x in range(len(ingreds)):
        # Create a dict - 'dim' is dimension number for ingredient
        # Later we'll add usage count 
        # (i just thought this info was interesting - not super useful)
        ingredMap[ingreds[x]] = {'dim':x, 'usageCount':0}
    # Add the usage counts to the map
    for recipe in recipes:
        for ingred in recipe['ingredients']:
            ingredMap[ingred]['usageCount'] += 1
    pprint(ingredMap)
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
    pprint(cuisineMap)
    return cuisineMap
    
def genVectorRepresentation(recipes):
    # Grab the ingredient and cuisine maps
    iMap = genIngredMap(recipes)
    cMap = genCuisineMap(recipes)
    vectors = list()
    # Loop through each recipe, vectorize it, and add it to vectors
    for recipe in recipes:
        # Create a list of zeros of the same size as the number
        # of possible ingredients, plus 1 extra slot for the label
        vector = [0] * (1 + len(iMap))
        # Set the dimension to 1 for each present ingredient
        for ingred in recipe['ingredients']:
            vector[iMap[ingred]['dim']] = 1
        # Add the label in the last slot
        vector[-1] = cMap[recipe['cuisine']]['dim']
        vectors.append(vector)
    return vectors
        

recipes = readJson(trainFile)
print("There are " + str(len(recipes)) + " recipes.")
vectors = genVectorRepresentation(recipes)
print(vectors)
