import numpy as np
import pandas as pd
import statistics
import math
import seaborn as sns
import matplotlib.pyplot as plt

logTransposeConst = 0.00000000000000001


def zeroFlux(key, dictionary):
    fluxes = dictionary[key]
    for num in fluxes:
        if not num == 0:
            return False
    return True


def createTuples(dictionary):
    result = []
    keys = dictionary.keys()
    for key in keys:
        mean = statistics.mean(dictionary[key])
        result.append(tuple((key, mean)))
    return result


def createHeatMapFluxes(reactionDictionary, mapHeight, annot_grid):
    reactions = reactionDictionary.keys()
    zeroFluxReactions = []
    nonZeroFlux = {}

    for key in reactions:
        if zeroFlux(key, reactionDictionary):
            zeroFluxReactions.append(key)
        else:
            res = [abs(ele) for ele in reactionDictionary[key]]
            nonZeroFlux[key] = res

    transformedData = {}
    for key in nonZeroFlux.keys():
        new = []
        old = nonZeroFlux[key]
        for val in old:
            new.append(math.log(val + logTransposeConst))
        transformedData[key] = new

    tuples = createTuples(transformedData)
    tuples.sort(key=lambda x: x[1])

    sortedReactions = {}
    for elem in tuples:
        key = elem[0]
        sortedReactions[key] = transformedData[key]

    df = pd.DataFrame.from_dict(sortedReactions, orient='index')
    df.index.name = 'Reactions'
    df.reset_index(inplace=False)

    fig, ax = plt.subplots(figsize=(50, mapHeight))
    plt.xlabel("Seconds")
    if annot_grid:
        hmap = sns.heatmap(df, cmap="rocket_r", linewidths=.5, annot=True)
    else:
        hmap = sns.heatmap(df, cmap="rocket_r")

    result = [hmap, zeroFluxReactions]

    return result


def notPresent(key, complexes):
    amounts = complexes[key]
    for num in amounts:
        if not num == 0:
            return False
    return True


def createHeatMapComplexes(complexes, mapHeight, annot_grid):
    complexNames = complexes.keys()
    zeroQuantComplexes = []
    presentComplexes = {}

    for key in complexes:
        if notPresent(key, complexes):
            zeroQuantComplexes.append(key)
        else:
            presentComplexes[key] = complexes[key]

    transformedData = {}
    for key in presentComplexes.keys():
        new = []
        old = presentComplexes[key]
        for val in old:
            new.append(math.log(val + logTransposeConst))
        transformedData[key] = new

    tuples = createTuples(transformedData)
    tuples.sort(key=lambda x: x[1])

    sortedComplexes = {}
    for elem in tuples:
        key = elem[0]
        sortedComplexes[key] = transformedData[key]

    df = pd.DataFrame.from_dict(sortedComplexes, orient='index')
    df.index.name = 'Reactions'
    df.reset_index(inplace=False)

    fig, ax = plt.subplots(figsize=(50, mapHeight))
    plt.xlabel("Seconds")
    if annot_grid:
        hmap = sns.heatmap(df, cmap="rocket_r", linewidths=.5, annot=True)
    else:
        hmap = sns.heatmap(df, cmap="rocket_r")

    result = [hmap, zeroQuantComplexes]

    return result


simData = np.load(r"..\out\geneRxnVerifData\output.npy", allow_pickle=True, encoding='ASCII')
fluxesWithCaption = simData.tolist()['agents']['0']['listeners']['fba_results']['estimated_fluxes']
complexes = simData.tolist()['agents']['0']['bulk']

ecData = pd.read_csv(r"..\notebooks\new genes\geneData.txt")
ecReactions = ecData["reactionID"].values.tolist()

simReactions = createHeatMapFluxes(fluxesWithCaption, 650, True)
print("Overall Simulation Reaction Non-Fluxes: \n")
print(simReactions[0])
print("\n Zero Flux Reactions: ")
print(simReactions[1])
simComplexes = createHeatMapComplexes(complexes, 650, False)
print("\n Overall Simulation Complex Presence: \n")
print(simComplexes[0])
print("\n Zero presence complexes: ")
print(simComplexes[1])

temp = []
for word in ecReactions:
    if not word == "No Reaction Found":
        temp.append(word)
ecReactions = temp

reactionsSim = fluxesWithCaption.keys()

foundReactions = {}
unfoundReactions = []

for reaction in ecReactions:
    x = 0
    for reactionSim in reactionsSim:
        if reaction in reactionSim:
            index = reactionSim.find(reaction)
            indexOfNextChar = index + len(reaction)
            if indexOfNextChar >= len(reactionSim) or not (
                    reactionSim[indexOfNextChar].isdigit() or reactionSim[indexOfNextChar].isalpha()):
                x = 1
                key = "(" + reaction + ")" + reactionSim
                foundReactions[key] = fluxesWithCaption[reactionSim]
    if x == 0:
        unfoundReactions.append(reaction)
