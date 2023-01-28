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


def updateKms(source, substrate, km, params):
    if source not in params:
        params[source] = {}

    if substrate in params[source]:
        params[source][substrate]['km'] = float(km)
    else:
        params[source][substrate] = {
            'km': float(km),
            'kcat': -1
        }


def updateKcats(source, substrate, kcat, params):
    if source not in params:
        params[source] = {}

    if substrate in params[source]:
        params[source][substrate]['kcat'] = float(kcat)
    else:
        params[source][substrate] = {
            'km': -1,
            'kcat': float(kcat)
        }


def findSources(val, sources):
    if 'citation' in val:
        val1 = val['citation']
    else:
        return tuple("None")

    if type(val1) is list:
        for pub in val1:
            sources.append(pub['Publication']['@frameid'])
    else:
        sources.append(val1['Publication']['@frameid'])

    sources = tuple(sorted(sources))

    return sources


def revaluateSize(dicte, reactions, rxn_index, enzymes, enz_index, params):
    if 'km' in dicte:
        dict_km = dicte['km']
        if type(dict_km) is list:
            for val in dict_km:
                sources = []
                sources = findSources(val, sources)

                substrate = val['substrate']
                if 'Compound' in substrate:
                    substrate = substrate['Compound']['@frameid']
                else:
                    substrate = substrate['Protein']['@frameid']

                km = val['value']['#text']
                updateKms(sources, substrate, km, params)
        else:
            substrate = dict_km['substrate']
            if 'Compound' in substrate:
                substrate = substrate['Compound']['@frameid']
            else:
                substrate = substrate['Protein']['@frameid']

            km = dict_km['value']['#text']
            sources = []
            sources = findSources(dict_km, sources)
            updateKms(sources, substrate, km, params)

    if 'kcat' in dicte:
        dict_kcat = dicte['kcat']
        if type(dict_kcat) is list:
            for val in dict_kcat:
                sources = []
                sources = findSources(val, sources)

                substrate = val['substrate']
                if 'Compound' in substrate:
                    substrate = substrate['Compound']['@frameid']
                else:
                    substrate = substrate['Protein']['@frameid']

                kcat = val['value']['#text']
                updateKcats(sources, substrate, kcat, params)
        else:
            substrate = dict_kcat['substrate']
            if 'Compound' in substrate:
                substrate = substrate['Compound']['@frameid']
            else:
                substrate = substrate['Protein']['@frameid']

            kcat = dict_kcat['value']['#text']
            sources = []
            sources = findSources(dict_kcat, sources)
            updateKcats(sources, substrate, kcat, params)

    if 'km' not in dicte and 'kcat' not in dicte:
        del reactions[rxn_index]
        del enzymes[enz_index]

    size = len(params)

    for x in range(size - 1):
        reactions.insert(rxn_index, reactions[rxn_index])
        enzymes.insert(enz_index, enzymes[enz_index])


def updateKinetics(substrates, kcats, kms, dicte, reactions, rxn_index, enzymes, enz_index):
    if 'Enzymatic-Reaction' in dicte['ptools-xml']:
        dicte = dicte['ptools-xml']['Enzymatic-Reaction']

    params = {}

    revaluateSize(dicte, reactions, rxn_index, enzymes, enz_index, params)

    for source in params:
        temp_substrates = list(params[source].keys())
        temp_km = []
        temp_kcat = []
        for substrate in temp_substrates:
            temp_km.append(params[source][substrate]['km'])
            temp_kcat.append(params[source][substrate]['kcat'])
        substrates.append(temp_substrates)
        kcats.append(temp_kcat)
        kms.append(temp_km)


# Establish Session
s = requests.Session()

# Post login credentials to session:
s.post('https://websvc.biocyc.org/credentials/login/', data={'email': 'aniketh@stanford.edu', 'password': '1324Ai22@4'})

# Create Gene List:
genes = list(pd.read_csv("New Liste - Sheet.csv")["Gene ID (EcoCyc)"])

# Create Set of New Enzymes:
enzyme_rxns = set({})
not_found_genes = []
for gene in genes:
    getEnzyme_Rxns(gene, enzyme_rxns, not_found_genes)

reactions, enzymes, kms, kcats, substrates, temperatures = [], [], [], [], [], []


# Extract Information
for rxn in enzyme_rxns:
    response = s.get("https://websvc.biocyc.org/getxml?id=ECOLI:" + rxn + "&detail=full")
    tree = ElementTree.fromstring(response.content)
    dicte = etree_to_dict(tree)
    reactions.append(getReaction(dicte))
    enzymes.append([getEnzyme(dicte)])
    updateKinetics(substrates, kcats, kms, dicte, reactions, len(reactions) - 1, enzymes, len(enzymes) - 1)
    temperatures.append([])

    if len(reactions) + len(enzymes) + len(kms) + len(kcats) != 4 * len(reactions):
        print(1)

information = {
    'reactionID': reactions,
    'enzymeID': enzymes,
    'substrateIDs': substrates,
    'kcat (1/s)': kcats,
    'kM (uM)': kms,
    'Temp': temperatures
}

df = pd.DataFrame.from_dict(information)

print(1)
