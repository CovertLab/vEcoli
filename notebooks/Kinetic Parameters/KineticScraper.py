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
        params[source] = {
            'substrates': {},
            'general': {}
        }

    if substrate in params[source]['substrates']:
        params[source]['substrates'][substrate]['km'] = float(km)
    else:
        params[source]['substrates'][substrate] = {
            'km': float(km),
            'kcat': -1,
            'vmax': -1,
        }

        params[source]['general'] = {
            'ph': [],
            'temperature': []
        }


def updateKcats(source, substrate, kcat, params):
    if source not in params:
        params[source] = {
            'substrates': {},
            'general': {}
        }

    if substrate in params[source]['substrates']:
        params[source]['substrates'][substrate]['kcat'] = float(kcat)
    else:
        params[source]['substrates'][substrate] = {
            'km': -1,
            'kcat': float(kcat),
            'vmax': -1,
        }

        params[source]['general'] = {
            'ph': [],
            'temperature': []
        }


def updateVmaxes(source, substrate, vmax, params):
    if source not in params:
        params[source] = {
            'substrates': {},
            'general': {}
        }

    if substrate in params[source]['substrates']:
        params[source]['substrates'][substrate]['vmax'] = float(vmax)
    else:
        params[source]['substrates'][substrate] = {
            'km': -1,
            'kcat': -1,
            'vmax': float(vmax),
        }

        params[source]['general'] = {
            'ph': [],
            'temperature': []
        }


def updatePhs(source, ph, params):
    if source not in params:
        params[source] = {
            'substrates': {},
            'general': {}
        }

    if 'ph' in params[source]['general']:
        params[source]['general']['ph'].append(ph)
    else:
        params[source]['general'] = {
            'ph': [ph],
            'temperature': []
        }


def updateTemps(source, temp, params):
    if source not in params:
        params[source] = {
            'substrates': {},
            'general': {}
        }

    if 'general' in params[source]['general']:
        params[source]['general']['temperature'].append(float(temp))
    else:
        params[source]['general'] = {
            'ph': [],
            'temperature': [temp]
        }


def findSources(val, sources):
    if 'citation' in val:
        val1 = val['citation']
    else:
        return "None", "None"

    if type(val1) is list:
        for pub in val1:
            if 'pubmed-id' in pub['Publication']:
                sources.append((pub['Publication']['@frameid'], pub['Publication']['pubmed-id']['#text']))
            else:
                sources.append((pub['Publication']['@frameid'], "None"))
    else:
        if 'pubmed-id' in val1['Publication']:
            sources.append((val1['Publication']['@frameid'], val1['Publication']['pubmed-id']['#text']))
        else:
            sources.append((val1['Publication']['@frameid'], "None"))

    sources = tuple(sorted(sources))

    return sources


def revaluateSize(dicte, reactions, rxn_index, enzymes, enz_index, params, no_params):
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

    if 'vmax' in dicte:
        dict_vmax = dicte['vmax']
        if type(dict_vmax) is list:
            for val in dict_vmax:
                sources = []
                sources = findSources(val, sources)

                substrate = val['substrate']
                if 'Compound' in substrate:
                    substrate = substrate['Compound']['@frameid']
                else:
                    substrate = substrate['Protein']['@frameid']

                vmax = val['value']['#text']
                updateVmaxes(sources, substrate, vmax, params)
        else:
            substrate = dict_vmax['substrate']
            if 'Compound' in substrate:
                substrate = substrate['Compound']['@frameid']
            else:
                substrate = substrate['Protein']['@frameid']

            vmax = dict_vmax['value']['#text']
            sources = []
            sources = findSources(dict_vmax, sources)
            updateVmaxes(sources, substrate, vmax, params)

    if 'ph-opt' in dicte:
        dict_ph = dicte['ph-opt']
        if type(dict_ph) is list:
            for val in dict_ph:
                sources = []
                sources = findSources(val, sources)

                if 'value' in val:
                    ph = val['value']['#text']
                else:
                    ph = val['#text']

                updatePhs(sources, ph, params)
        else:
            if 'value' in dict_ph:
                ph = dict_ph['value']['#text']
            else:
                ph = dict_ph['#text']

            sources = []
            sources = findSources(dict_ph, sources)
            updatePhs(sources, ph, params)

    if 'temperature-opt' in dicte:
        dict_temp = dicte['temperature-opt']
        if type(dict_temp) is list:
            for val in dict_temp:
                sources = []
                sources = findSources(val, sources)

                if 'value' in val:
                    temp = val['value']['#text']
                else:
                    temp = val['#text']

                updateTemps(sources, temp, params)
        else:
            if 'value' in dict_temp:
                temp = dict_temp['value']['#text']
            else:
                temp = dict_temp['#text']

            sources = []
            sources = findSources(dict_temp, sources)
            updateTemps(sources, temp, params)

    if ('km' not in dicte and 'kcat' not in dicte and
            'vmax' not in dicte and 'ph-opt' not in dicte and
            'temperature-opt' not in dicte):
        no_params.append([reactions[rxn_index], enzymes[enz_index]])
        del reactions[rxn_index]
        del enzymes[enz_index]

    size = len(params)

    for x in range(size - 1):
        reactions.insert(rxn_index, reactions[rxn_index])
        enzymes.insert(enz_index, enzymes[enz_index])


def updateKinetics(pubmeds, substrates, kcats, kms, vmaxes, phs, temperatures, dicte, reactions, rxn_index, enzymes,
                   enz_index, no_params):
    if 'Enzymatic-Reaction' in dicte['ptools-xml']:
        dicte = dicte['ptools-xml']['Enzymatic-Reaction']

    params = {}

    revaluateSize(dicte, reactions, rxn_index, enzymes, enz_index, params, no_params)

    for source in params:
        temp_substrates = list(params[source]['substrates'].keys())
        temp_km = []
        temp_kcat = []
        temp_vmax = []
        temp_ph = []
        temp_temperature = []
        for substrate in temp_substrates:
            temp_km.append(params[source]['substrates'][substrate]['km'])
            temp_kcat.append(params[source]['substrates'][substrate]['kcat'])
            temp_vmax.append(params[source]['substrates'][substrate]['vmax'])
        if params[source]['general']:
            temp_ph = params[source]['general']['ph']
            temp_temperature = params[source]['general']['temperature']
        pubmeds.append([x[1] for x in source])
        substrates.append(temp_substrates)
        kcats.append(temp_kcat)
        kms.append(temp_km)
        vmaxes.append(temp_vmax)
        phs.append(temp_ph)
        temperatures.append(temp_temperature)


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

pubmeds, reactions, enzymes, kms, kcats, substrates, phs, temperatures, vmaxes = [], [], [], [], [], [], [], [], []

# Extract Information
no_params = []
for rxn in enzyme_rxns:
    response = s.get("https://websvc.biocyc.org/getxml?id=ECOLI:" + rxn + "&detail=full")
    tree = ElementTree.fromstring(response.content)
    dicte = etree_to_dict(tree)
    reactions.append(getReaction(dicte))
    enzymes.append([getEnzyme(dicte)])
    updateKinetics(pubmeds, substrates, kcats, kms, vmaxes, phs, temperatures, dicte, reactions, len(reactions) - 1, enzymes,
                   len(enzymes) - 1, no_params)

information = {
    'pubmedID': pubmeds,
    'reactionID': reactions,
    'enzymeID': enzymes,
    'substrateIDs': substrates,
    'kcat (1/s)': kcats,
    'kM (uM)': kms,
    'vmax': vmaxes,
    'ph': phs,
    'temperature': temperatures
}

output = pd.DataFrame.from_dict(information)
output.to_csv("params.csv")

