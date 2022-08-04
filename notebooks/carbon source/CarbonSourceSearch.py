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

rxns = list(rxn_metabolites.keys())
glucoseRxns = {}
fructoseRxns = {}
for rxn in rxns:
    metabolites = rxn_metabolites[rxn].keys()
    for metabolite in metabolites:
        if metabolite == 'CARBON-DIOXIDE[p]':
            glucoseRxns[rxn] = rxn_metabolites[rxn]
        if metabolite == 'CARBON-DIOXIDE[c]' and rxn_metabolites[rxn]['CARBON-DIOXIDE[c]'] == 1:
            fructoseRxns[rxn] = rxn_metabolites[rxn]

fluxes = {}
for rKeys in fructoseRxns.keys():
    fluxes[rKeys] = fluxesWithCaption[rKeys]
createHeatMapFluxes(fluxes, 20, True)

#helpful portion of createHeatMapFluxes
zeroFluxReactions = []
    nonZeroFlux = {}

    for key in reactions:
        if zeroFlux(key, reactionDictionary):
            zeroFluxReactions.append(key)
        else:
            res = [abs(ele) for ele in reactionDictionary[key]]
            nonZeroFlux[key] = res