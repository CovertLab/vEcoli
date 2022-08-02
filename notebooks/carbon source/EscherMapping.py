import requests
from xml.etree import ElementTree
from collections import defaultdict
import numpy as np
import pandas as pd
import statistics
import math
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import json
import urllib.request
import escher
from escher import Builder
import cobra
import time
from time import sleep

def etree_to_dict(t):
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v
                     for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(('@' + k, v)
                        for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]['#text'] = text
        else:
            d[t.tag] = text
    return d

def createTree(link):
    time.sleep(1)
    response = requests.get(link)
    tree = ElementTree.fromstring(response.content)
    return tree


def createDictFromName(complexList, errors):
    result = []

    for name in complexList:
        try:
            linkComplex = "https://websvc.biocyc.org/getxml?id=ECOLI:" + name + "&detail=low"
            treeComplex = createTree(linkComplex)
            treeDictComplex = etree_to_dict(treeComplex)
            result.append((name, treeDictComplex))
        except Exception as e:
            errors.append((name, e))

    return result

def checkAndReplaceCommonName(reaction, names):
    return

def checkAndReplaceSynonyms(reactionList, names):
    return

simData = np.load(r"../../out/geneRxnVerifData/output.npy", allow_pickle=True, encoding='ASCII')
fluxesWithCaption = simData.tolist()['agents']['0']['listeners']['fba_results']['estimated_fluxes']
complexes = simData.tolist()['agents']['0']['bulk']

metabolData = np.load(r"../../out/geneRxnVerifData/stoichiometry.npy", allow_pickle=True, encoding='ASCII')
rxn_metabolites = metabolData.tolist()

reactions = fluxesWithCaption.keys()
ecReactions = pd.read_csv('reactions.tsv', sep='\t', header=0)['Reaction'].tolist()

foundNames = []
notFound = []
for rxn in ecReactions:
    found = False
    for key in fluxesWithCaption:
        if rxn in key:
            found = True
            foundNames.append((rxn, key))
    if not found:
        notFound.append(rxn)

prev = ""
fluxes = {}
for names in foundNames:
    if prev == names[0]:
        for i in range(len(fluxes[prev])):
            if "(reverse)" in names[1]:
                fluxes[prev][i] -= fluxesWithCaption[names[1]][i]
            else:
                fluxes[prev][i] += fluxesWithCaption[names[1]][i]
    else:
        prev = names[0]
        fluxes[prev] = fluxesWithCaption[names[1]]

with open('../../iJO1366.Central metabolism.json') as json_file:
    data = json.load(json_file)

escherRxns = data[1]['reactions']
names = []

for number in escherRxns:
    reactionName = escherRxns[number]['name']
    names.append(reactionName)

errors = []
test = ['FADSYN-RXN']
reactionDict = createDictFromName(test, errors)
