import math
from collections import defaultdict
from typing import List, Any
from xml.etree import ElementTree
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import requests


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


def getnames(values):
    res = []
    for i in range(len(dicte)):
        temp = values[i]['CELL'][-4]

        if ('VALUE' in temp and temp['VALUE'] is None):
            continue

        temp = temp['FRAME']

        if type(temp) is list:
            for val in temp:
                res.append((val['@ID'], i))
        else:
            res.append((temp['@ID'], i))

    return res


response = requests.get("https://websvc.biocyc.org/st-service-get?id=biocyc17-56717-3854927444&format=xml")
tree = ElementTree.fromstring(response.content)
dicte = etree_to_dict(tree)['GROUP']['ROWS']['ROW']

genesList = list(pd.read_csv("New Liste - Sheet.csv", header=None).loc[1:, 0])
gene_to_Km_Kcat = {}

names = getnames(dicte)
notFound = []

for gene in genesList:
    index = -1
    for tup in names:
        if gene in tup:
            index = tup[1]
            break
    if index < 0:
        notFound.append(gene)
        continue

    Km = dicte[index]['CELL'][-1]
    if Km is not None:
        if type(Km['VALUE']) is list:
            Km = [eval(i) for i in Km['VALUE']]
        else:
            Km = [eval(Km['VALUE'])]

    Kcat = dicte[index]['CELL'][-2]
    if Kcat is not None:
        if type(Kcat['VALUE']) is list:
            Kcat = [float(i) for i in Kcat['VALUE']]
        else:
            Kcat = [float(Kcat['VALUE'])]

    gene_to_Km_Kcat[gene] = {}
    gene_to_Km_Kcat[gene]['Km'] = Km
    gene_to_Km_Kcat[gene]['Kcat'] = Kcat

for gene in notFound:
    gene_to_Km_Kcat[gene] = {
        'Km': None,
        'Kcat': None
    }

data_dict = {
    'Km_&_Kcat': {
        'count': 0,
        'genes': []
    },
    'Km_Only': {
        'count': 0,
        'genes': []
    },
    'Kcat_Only': {
        'count': 0,
        'genes': []
    },
    'None': {
        'count': 0,
        'genes': []
    }
}

for gene in gene_to_Km_Kcat:
    if (gene_to_Km_Kcat[gene]['Km'] is not None and
            gene_to_Km_Kcat[gene]['Kcat'] is not None):
        data_dict['Km_&_Kcat']['count'] += 1
        data_dict['Km_&_Kcat']['genes'].append(gene)
    elif gene_to_Km_Kcat[gene]['Km'] is not None:
        data_dict['Km_Only']['count'] += 1
        data_dict['Km_Only']['genes'].append(gene)
    elif gene_to_Km_Kcat[gene]['Kcat'] is not None:
        data_dict['Kcat_Only']['count'] += 1
        data_dict['Kcat_Only']['genes'].append(gene)
    else:
        data_dict['None']['count'] += 1
        data_dict['None']['genes'].append(gene)

counts = [data_dict[val]['count'] for val in data_dict]
labels = [val for val in data_dict]
df = pd.DataFrame(zip(labels, counts))
df = df.T
new_header = df.iloc[0]
df = df[1:]
df.columns = new_header

ax = sns.barplot(data=df)
plt.xlabel("Kinetic Data Available")
plt.ylabel("Number of Genes (From New Gene List)")
ax.bar_label(ax.containers[0])
plt.show()
print(1)
