from lib import *

train = readJson('srcData/train.json')

#Read in json files
train = readJson('srcData/train.json')
# Grab the ingredient and cuisine maps 
# (cuisine map does not exist if input file is test data)
iMap = genIngredMap(train)
cMap = genCuisineMap(train)
#Convert json into a list of vector examples, and maps from ingredients
#  to their indices and cuisines into their label numbers
trainVectors, unknownTrain = genVectorRepresentation(train, iMap, cMap)

cuisIngs = dict()
cuisCount = [0] * 20
#Loop through cuisines
for cuis in range(len(cMap)):
    #Set up empty ingred list
    ings = [0.] * len(iMap)
    #Loop through vectors in train data
    for vec in trainVectors:
        #If match cur cuisine
        if vec[-1] == cuis:
            #Sum ingreds with current total
            ings = [x + y for x, y in zip(ings, vec[0:-1])]
            cuisCount[cuis] += 1
    #Find label for cuisine label number
    label = ""
    for key,val in cMap.items():
        if val['dim']==cuis:
            label = key
    cuisIngs[label] = ings

#Loop through each cuis
for key in cuisIngs.keys():
    #Print out cuis label
    idx = 0
    # Get idx of cuisine
    for keyC,val in cMap.items():
        if keyC==key:
            idx = cMap[keyC]['dim']
    print(key + ": " + str(cuisCount[idx]))
    #Create list containing ingred dim number 
    cuisSum = list()
    #CuisIngs vals are summed ingredient counts
    #Loop through each ingredient
    for x in range(len(cuisIngs[key])):
        #Add pair (ingred count, ingred dimension number)
        cuisSum.append((cuisIngs[key][x], x))
    cuisSum = sorted(cuisSum, key=lambda x: x[0])
    #Loop through last 5 ingreds
    for x in range(10):
        #Loop through all ingredients in ing map
        for key,val in iMap.items():
            #If ingred map dim number matches cuisine sum dim num
            if val['dim']==cuisSum[-1*x - 1][1]:
                #Print out label and count
                print("  " + key + ":\t" + str(cuisSum[-1*x - 1][0]))
