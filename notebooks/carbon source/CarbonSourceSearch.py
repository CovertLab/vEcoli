import numpy as np

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

    if flux[0][1] / 100000 < 1:
        return [flux[0][0]]

    result = []
    x = 0
    while x < len(flux) and len(result) < 3 and flux[x][1] / 100000 >= 1:
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

    if len(flux) == 0:
        return flux

    top5 = filterTop5(flux)
    return top5


def getSubstrates(rxnName):
    metabols = rxn_metabolites[rxnName]
    substrates = []

    for key in metabols:
        if "water" not in key.lower() and "proton" not in key.lower():
            if metabols[key] == -1:
                substrates.append(key)

    return substrates


def searchForCarbon(molecule, finalList, searchedMetabols, level):
    molrxns = getReactions(molecule)

    if len(molrxns) == 0:
        return

    if level >= 20:
        return

    for molreaction in molrxns:
        subs = getSubstrates(molreaction)
        for mol in subs:
            if mol not in searchedMetabols:
                searchedMetabols.add(mol)
                searchForCarbon(mol, finalList, searchedMetabols, level + 1)
                if level == 19:
                    finalList.update(molrxns)
                    return

startingMolecule = 'CARBON-DIOXIDE[c]'
finalList = set()
searchedMetabols = set()
searchedMetabols.add(startingMolecule)

searchForCarbon(startingMolecule, finalList, searchedMetabols, 0)

np.save("10thLevelCarbonReactions", finalList)
