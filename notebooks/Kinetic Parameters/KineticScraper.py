# Imports
import math
from collections import defaultdict
from typing import List, Any
from xml.etree import ElementTree
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import requests


# Helper Functions
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


def getEnzyme_Rxns(gene, res, no):
    response = s.get("https://websvc.biocyc.org/apixml?fn=enzymes-of-gene&id=ECOLI:" + gene + "&detail=low")
    tree = ElementTree.fromstring(response.content)
    dicte = etree_to_dict(tree)
    dicte = dicte['ptools-xml']['Protein']

    if type(dicte) is list:
        no.append(gene)
        return
    else:
        dicte = dicte['catalyzes']['Enzymatic-Reaction']

    if type(dicte) is dict:
        res.add(dicte['@frameid'])
    else:
        for val in dicte:
            res.add(val['@frameid'])

def getReaction(dicte):
    if 'Enzymatic-Reaction' in dicte['ptools-xml']:
        return dicte['ptools-xml']['Enzymatic-Reaction']['reaction']['Reaction']['@frameid']

def getEnzyme(dicte):
    if 'Enzymatic-Reaction' in dicte['ptools-xml']:
        return dicte['ptools-xml']['Enzymatic-Reaction']['enzyme']['Protein']['@frameid']

def getKms(enzyme_rxn_dict):
    return 0

def getSubstrates(enzyme_rxn_dict):
    return 0



def getKms(gene):
    response = s.get("https://websvc.biocyc.org/apixml?fn=reactions-of-gene&id=ECOLI:" + gene + "&detail=full")
    tree = ElementTree.fromstring(response.content)
    dicte = etree_to_dict(tree)
    enzymeRxn = dicte['ptools-xml']['Reaction']['enzymatic-reaction']['Enzymatic-Reaction']['@frameid']
    response = s.get("https://websvc.biocyc.org/getxml?id=ECOLI:" + enzymeRxn + "&detail=full")
    tree = ElementTree.fromstring(response.content)
    dicte = etree_to_dict(tree)

    new = dicte['ptools-xml']['Enzymatic-Reaction']
    res = []

    if 'km' in new:
        new = new['km']

        for dicte in new:
            val = dicte['value']['#text']
            res.append(float(val))

    return res


# Establish Session
s = requests.Session()

# Post login credentials to session:
s.post('https://websvc.biocyc.org/credentials/login/', data={'email': 'aniketh@stanford.edu', 'password': '1324Ai22@4'})

# Create Gene List:
genes = list(pd.read_csv("New Liste - Sheet.csv")["Gene ID (EcoCyc)"])

# Create Set of New Enzymes:
enzyme_rxns = set({})
not_found = []
for gene in genes:
    getEnzyme_Rxns(gene, enzyme_rxns, not_found)

reactions = []
enzymes = []

# Extract Information
info = {
    'genes': genes,

}

for rxn in enzyme_rxns:
    response = s.get("https://websvc.biocyc.org/getxml?id=ECOLI:" + rxn + "&detail=full")
    tree = ElementTree.fromstring(response.content)
    dicte = etree_to_dict(tree)
    reactions.append(getReaction(dicte))
    enzymes.append([getEnzyme(dicte)])

information = {
    'reactionID': reactions,
    'enzymeID': enzymes
}

df = pd.DataFrame.from_dict(information)

print(1)
