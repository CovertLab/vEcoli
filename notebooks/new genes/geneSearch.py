# %%
import time
from collections import defaultdict
from typing import List, Any
from xml.etree import ElementTree

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


def createTree(link):
    time.sleep(1)
    response = requests.get(link)
    tree = ElementTree.fromstring(response.content)
    return tree


def getReactionID(tree):
    return tree["ptools-xml"]["Reaction"]["@frameid"]


def getProteinComplex(tree):
    return tree["ptools-xml"]["Protein"]["@frameid"]


def isReversible(tree):
    try:
        return "REVERSIBLE" == tree["ptools-xml"]["Reaction"]["reaction-direction"]
    except KeyError:
        return "Not found"

def getStoichiometrySubstrate(tree, geneReactions, stoichDict):
    coefficients = []
    substrates = tree["ptools-xml"]["Reaction"]["left"]

    if type(substrates) is dict:
        try:
            name = substrates["Compound"]["@frameid"]
        except KeyError:
            name = substrates["Protein"]["@frameid"]

        if "coefficient" in substrates:
            coef = substrates["coefficient"]["#text"]
            stoichDict[name] = int(coef) * -1
        else:
            stoichDict[name] = -1
    else:
        for compound in substrates:
            try:
                name = compound["Compound"]["@frameid"]
            except KeyError:
                name = compound["Protein"]["@frameid"]

            if "coefficient" in compound:
                coef = compound["coefficient"]["#text"]
                stoichDict[name] = int(coef) * -1
            else:
                stoichDict[name] = -1


def getStoichiometryProduct(tree, geneReactions, stoichDict):
    coefficients = []
    products = tree["ptools-xml"]["Reaction"]["right"]

    if type(products) is dict:
        try:
            name = products["Compound"]["@frameid"]
        except KeyError:
            name = products["Protein"]["@frameid"]

        if name not in stoichDict:
            if "coefficient" in products:
                coef = products["coefficient"]["#text"]
                stoichDict[name] = int(coef)
            else:
                stoichDict[name] = 1
        else:
            if "coefficient" in products:
                coef = products["coefficient"]["#text"]
                stoichDict[name] = [stoichDict[name], int(coef)]
            else:
                stoichDict[name] = [stoichDict[name], 1]
    else:
        for compound in products:
            try:
                name = compound["Compound"]["@frameid"]
            except KeyError:
                name = compound["Protein"]["@frameid"]

            if name not in stoichDict:
                if "coefficient" in products:
                    coef = products["coefficient"]["#text"]
                    stoichDict[name] = int(coef)
                else:
                    stoichDict[name] = 1
            else:
                if "coefficient" in products:
                    coef = products["coefficient"]["#text"]
                    stoichDict[name] = [stoichDict[name], int(coef)]
                else:
                    stoichDict[name] = [stoichDict[name], 1]

    geneReactions["stoichiometry"] = stoichDict


def createDictFromName(complexList):
    result = []

    for name in complexList:
        linkComplex = "https://websvc.biocyc.org/getxml?id=ECOLI:" + name + "&detail=low"
        treeComplex = createTree(linkComplex)
        treeDictComplex = etree_to_dict(treeComplex)
        result.append(treeDictComplex)

    return result


def allMonomers(currentDicts):
    for dictionary in currentDicts:
        if "gene" not in dictionary["ptools-xml"]["Protein"]:
            return False

    return True


def getMonomersHelper(dictionary):
    result = []
    components = dictionary["ptools-xml"]["Protein"]["component"]

    if type(components) is dict:
        result.append(components["Protein"]["@frameid"])
    else:
        for protein in components:
            result.append(protein["Protein"]["@frameid"])

    return result


def getMonomers(currentList, monomerDicts):
    currentDicts = createDictFromName(currentList)

    for dictionary in currentDicts:
        if "gene" in dictionary["ptools-xml"]["Protein"]:
            monomerDicts.append(dictionary)
        else:
            monomers = getMonomersHelper(dictionary)
            getMonomers(monomers, monomerDicts)

    return


def getGenes(currentList):
    result: List[Any] = []
    monomerDicts = []
    getMonomers(currentList, monomerDicts)

    for dictionary in monomerDicts:
        genes = dictionary["ptools-xml"]["Protein"]["gene"]
        if type(genes) is dict:
            result.append(genes["Gene"]["@frameid"])
        else:
            for gene in genes:
                result.append(gene["@frameid"])

    return result


def getMonomersCountsGenes(easy):
    complexDicts = createDictFromName(easy)

    monomers = []
    inAddition = []
    count = []
    genes = []

    for dictionary in complexDicts:

        components = dictionary["ptools-xml"]["Protein"]

        if "component" in components:
            components = components["component"]

            if type(components) is dict:
                monomers.append(components["Protein"]["@frameid"])
                inAddition.append(components["Protein"]["@frameid"])
                if "coefficient" in components:
                    coef = components["coefficient"]["#text"]
                    count.append(int(coef))
                else:
                    count.append(1)
            else:
                for protein in components:
                    monomers.append(protein["Protein"]["@frameid"])
                    inAddition.append(protein["Protein"]["@frameid"])
                    if "coefficient" in protein:
                        coef = protein["coefficient"]["#text"]
                        count.append(int(coef))
                    else:
                        count.append(1)
        else:
            inAddition.append(components["@frameid"])

    genes = getGenes(inAddition)

    data = [monomers, count, genes]

    return data


def getProteinComplexes(tree):
    result = []

    complexes = tree["ptools-xml"]["Reaction"]["enzymatic-reaction"]["Enzymatic-Reaction"]

    if type(complexes) is dict:
        result.append(complexes["enzyme"]["Protein"]["@frameid"])
    else:
        for complexR in complexes:
            result.append(complexR["enzyme"]["Protein"]["@frameid"])

    return result


# %%
def getGeneProductAddresses(products):
    result = []

    if type(products) is dict:
        link = products["@resource"]
        result.append(link)
        return result
    else:
        for product in products:
            result.append(product["@resource"])

    return result


def getComplexDictionaries(monomerDicts, complexDicts):
    for dictionary in monomerDicts:
        stem = dictionary["ptools-xml"]["Protein"]

        if "component-of" not in stem:
            complexDicts.append(dictionary)
        else:
            complexes = stem["component-of"]
            temp = []
            if type(complexes) is dict and type(complexes["Protein"]) is dict:
                complexAddress = complexes["Protein"]["@resource"]
                link = "https://websvc.biocyc.org/" + complexAddress + "&detail=low"
                treeComplex = createTree(link)
                treeDictComplex = etree_to_dict(treeComplex)
                temp.append(treeDictComplex)
            else:
                if type(complexes["Protein"]) is not dict:
                    complexes = complexes["Protein"]
                for complexR in complexes:
                    complexAddress = complexR["@resource"]
                    link = "https://websvc.biocyc.org/" + complexAddress + "&detail=low"
                    treeComplex = createTree(link)
                    treeDictComplex = etree_to_dict(treeComplex)
                    temp.append(treeDictComplex)
            getComplexDictionaries(temp, complexDicts)

    return


def geneSearch(gene, reactionStore):
    geneID = gene

    # Getting gene dict
    linkGene = "https://websvc.biocyc.org/getxml?id=ECOLI:" + geneID + "&detail=low"
    treeGene = createTree(linkGene)
    treeDictGene = etree_to_dict(treeGene)

    # Getting addresses of all gene products
    products = treeDictGene["ptools-xml"]["Gene"]["product"]["Protein"]

    productAddresses = getGeneProductAddresses(products)

    # Convert Gene Product Addresses into Dictionaries
    monomerDicts = []

    for address in productAddresses:
        linkGeneProduct = "https://websvc.biocyc.org/" + address + "&detail=low"
        treeGeneProduct = createTree(linkGeneProduct)
        treeDictGeneProduct = etree_to_dict(treeGeneProduct)
        monomerDicts.append(treeDictGeneProduct)

    complexDicts = []

    getComplexDictionaries(monomerDicts, complexDicts)
    reactionAddresses = []

    for dictionary in complexDicts:
        stem = dictionary["ptools-xml"]["Protein"]
        if "catalyzes" not in stem:
            reactionAddresses.append("NONE")
        else:
            reactions = stem["catalyzes"]["Enzymatic-Reaction"]
            if type(reactions) is dict:
                respectiveReactions = []
                respectiveReactions.append(reactions["reaction"]["Reaction"]["@resource"])
                reactionAddresses.append(respectiveReactions)
            else:
                respectiveReactions = []
                for reaction in reactions:
                    respectiveReactions.append(reaction["reaction"]["Reaction"]["@resource"])
                reactionAddresses.append(respectiveReactions)

    # Get reaction dictionaries

    reactionDicts = []

    for addressList in reactionAddresses:
        if addressList == "NONE":
            reactionDicts.append("NONE")
        else:
            temp = []
            for address in addressList:
                linkReaction = "https://websvc.biocyc.org/" + address
                treeReaction = createTree(linkReaction)
                treeDictReaction = etree_to_dict(treeReaction)
                temp.append(treeDictReaction)

            reactionDicts.append(temp)
    geneReactionInfo = []

    for x in range(len(complexDicts)):

        dictionary1 = complexDicts[x]

        geneReactions = {
            "reactionID": "",
            "stoichiometry": [],
            "reversible": "",
            "protein_complexes": "",
            "protein_monomers": [],
            "monomer_counts": [],
            "genes": [],
        }

        dictionary2 = reactionDicts[x]

        if dictionary2 == "NONE":
            geneReactions["reactionID"] = "No Reaction Found"
            geneReactions["protein_complexes"] = getProteinComplex(dictionary1)
            geneReactions["genes"].append(geneID)

            components = dictionary1["ptools-xml"]["Protein"]

            if "component" in dictionary1:
                components = components["component"]
                if type(components) is dict:
                    geneReactions["protein_monomers"].append(components["Protein"]["@frameid"])
                else:
                    for protein in components:
                        geneReactions["protein_monomers"].append(protein["Protein"]["@frameid"])

            copy = geneReactions.copy()
            geneReactionInfo.append(copy)

        else:
            for dictionary2nest in dictionary2:
                geneReactions["reactionID"] = getReactionID(dictionary2nest)
                rxnName = geneReactions["reactionID"]
                if rxnName not in reactionStore:
                    reactionStore.add(rxnName)
                    geneReactions["protein_complexes"] = getProteinComplexes(dictionary2nest)
                    easy = geneReactions["protein_complexes"]
                    data = getMonomersCountsGenes(easy)
                    geneReactions["protein_monomers"] = data[0]
                    geneReactions["monomer_counts"] = data[1]
                    geneReactions["genes"] = data[2]
                    stoichiometryDict = {}
                    getStoichiometrySubstrate(dictionary2nest, geneReactions, stoichiometryDict)
                    getStoichiometryProduct(dictionary2nest, geneReactions, stoichiometryDict)
                    geneReactions["reversible"] = isReversible(dictionary2nest)
                    copy = geneReactions.copy()
                    geneReactionInfo.append(copy)

    df = pd.DataFrame(geneReactionInfo)
    return df


# %%
geneData = pd.DataFrame()
genesList = list(pd.read_csv("Genes.txt", header=None).loc[:, 0])
reactionStore = set()
genesThatDontWork = []
done = 0

for gene in genesList:
    try:
        temp = geneSearch(gene, reactionStore)
        geneData = geneData.append(temp, ignore_index=True)
    except Exception as e:
        genesThatDontWork.append({gene, e})
    done += 1

geneData.to_csv("geneData.txt", header=True, index=True)

