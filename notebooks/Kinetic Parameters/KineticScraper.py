# Imports
import math
from collections import defaultdict
from typing import List, Any
from xml.etree import ElementTree
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import requests

# Establish Session
s = requests.Session()

# Post login credentials to session:
s.post('https://websvc.biocyc.org/credentials/login/', data={'email':'aniketh@stanford.edu', 'password':'1324Ai22@4'})

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


def getGenes(reaction):
    response = requests.get("https://websvc.biocyc.org/apixml?fn=genes-of-reaction&id=ECOLI:" + reaction + "detail=low")
    tree = ElementTree.fromstring(response.content)
    dicte = etree_to_dict(tree)
    return dicte['ptools-xml']['Gene']['@frameid']


def g
