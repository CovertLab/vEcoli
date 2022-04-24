import pandas as pd
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

rnas = pd.read_table("reconstruction/ecoli/flat/rnas.tsv", comment='#')
model_degenes = []


# Use fold change from direct exposure to tetracycline instead
tet_FC = pd.read_table("ecoli/experiments/marA_binding/tet_FC.tsv")
tet_10_FC = tet_FC.loc[:,["Gene Name ", "10 mg/L tet."]]
tet_10_FC.rename(columns={
    "Gene Name ": "Gene name", "10 mg/L tet.": "Fold change"}, inplace=True)
degenes = tet_10_FC.sort_values(
    by="Fold change", ascending=False, ignore_index=True)

for i, gene in enumerate(degenes["Gene name"]):
    # Cycle through gene synonyms in rnas.tsv to find EcoCyc name for DE genes
    found = False
    for j, synonyms in enumerate(rnas["synonyms"]):
        if gene in synonyms:
            model_degenes.append(
                degenes.iloc[i,].append(rnas.iloc[j,]).to_frame().T)
            found = True
    if not found:
        print(gene)

# Concatenating at end is supposed to be more efficient than row-wise append
model_degenes = pd.concat(model_degenes, ignore_index=True)
# Delete these two duplicates that are the incorrect genes
model_degenes = model_degenes.loc[~((model_degenes['Gene name']=="acrE") & (model_degenes['common_name']=='acrB'))]
model_degenes = model_degenes.loc[~((model_degenes['Gene name']=="acrB") & (model_degenes['common_name']=='gyrB'))]
with open("ecoli/experiments/marA_binding/TU_id_to_index.json") as f:
    TU_id_to_index = json.load(f)

# Get model RNAs names by appending the "[c]" suffix, then get TU index for RNA
TU_idx = [TU_id_to_index[rna_id + "[c]"] for rna_id in model_degenes["id"]]
model_degenes["TU_idx"] = TU_idx
TU_idx_to_FC = {}

with open("data/wcecoli_t0.json") as f:
    initial_state = json.load(f)
bulk_names = list(initial_state["bulk"].keys())

# Include complexes
comp_rxns = pd.read_table(
    "reconstruction/ecoli/flat/complexation_reactions.tsv", comment = "#")
removed_comp_rxns = pd.read_table(
    "reconstruction/ecoli/flat/complexation_reactions_removed.tsv")
removed_rxn_ids = removed_comp_rxns["id"].to_list()
filter = [id not in removed_rxn_ids for id in comp_rxns["id"]]
comp_rxns = comp_rxns.loc[filter,]
comp_rxns["common_name"][comp_rxns["common_name"].isna()] = "No name"

def get_IDs(monomer_id):
    monomer_id = json.loads(monomer_id)
    # Noncoding RNAs
    if len(monomer_id) == 0:
        return [[], [], [], []]
    else:
        monomer_id = monomer_id[0]
    degene_bulk_ids = []
    for bulk_name in bulk_names:
        if monomer_id in bulk_name:
            degene_bulk_ids.append(bulk_name)

    monomers_used = []
    complex_names = []
    common_names = []
    for i in range(comp_rxns.shape[0]):
        stoich = json.loads(comp_rxns.iloc[i, 1])
        if monomer_id in stoich:
            monomers_used.append(stoich[monomer_id])
            complex_names.append(list(stoich.keys())[0])
            common_names.append(comp_rxns.iloc[i, 2])
    
    degene_complex_ids = []
    degene_monomers_used = []
    degene_common_names = []
    for bulk_name in bulk_names:
        for monomer_used, complex_name, common_name in zip(
            monomers_used, complex_names, common_names):
            if complex_name in bulk_name:
                degene_complex_ids.append(bulk_name)
                degene_monomers_used.append(monomer_used)
                degene_common_names.append(common_name)
    
    return [degene_bulk_ids, degene_monomers_used, 
            degene_complex_ids, degene_common_names]

# Protein IDs have varied suffixes: brute force search
with ProcessPoolExecutor(multiprocessing.cpu_count()) as executor:
    all_ids = executor.map(get_IDs, model_degenes["monomer_ids"])

(model_degenes["bulk_ids"], model_degenes["monomers_used"],
    model_degenes["complex_ids"], model_degenes["complex_common_names"]) = zip(
        *[ids for ids in all_ids])
    
model_degenes.to_csv(
    "ecoli/experiments/marA_binding/model_degenes.csv", index=False)
