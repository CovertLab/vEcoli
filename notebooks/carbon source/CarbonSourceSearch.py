import numpy as np
import pandas as pd
import statistics
import math
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import json
import escher
from escher import Builder
import cobra
from time import sleep

simData = np.load(r"../../out/geneRxnVerifData/output.npy", allow_pickle=True, encoding='ASCII')
fluxesWithCaption = simData.tolist()['agents']['0']['listeners']['fba_results']['estimated_fluxes']
complexes = simData.tolist()['agents']['0']['bulk']

metabolData = np.load(r"../../out/geneRxnVerifData/stoichiometry.npy", allow_pickle=True, encoding='ASCII')
rxn_metabolites = metabolData.tolist()

def zeroFlux(key1):
    fluxes1 = fluxesWithCaption[key1]
    for num in fluxes1:
        if not num == 0:
            return False
    return True

def filterNoFlux(reactions):
    nonZeroFlux = []
    for key in reactions:
        if not zeroFlux(key):
            res = [abs(ele) for ele in fluxesWithCaption[key]]
            nonZeroFlux.append((key, np.mean(res)))
    return nonZeroFlux


def filterTop5(flux):
    if len(flux) == 0:
        return []

    flux.sort(key=lambda x: x[1], reverse=True)

    if flux[0][1]/100000 < 1:
        return [flux[0][0]]

    result = []
    x = 0
    while len(result) < 5 and flux[x][1]/100000 >= 1:
        result.append(flux[x][0])
        x += 1
    return result


def getReactions(molecule):
    rxns = list(rxn_metabolites.keys())
    specificRxns = []
    for rxn in rxns:
        metabolites = rxn_metabolites[rxn].keys()
        for metabolite in metabolites:
            if metabolite == molecule and rxn_metabolites[rxn][molecule] == 1:
                specificRxns.append(rxn)

    flux = filterNoFlux(specificRxns)
    top5 = filterTop5(flux)
    return top5

x = getReactions('CARBON-DIOXIDE[c]')
print(x)