"""
====================================
Compile marA-Regulated Gene Metadata
====================================

This file contains functions to compile important metadata for genes that
are designed to be regulated by marA when the `mar_regulon` option is enabled.
For each gene, the output `model_degenes.csv` file contains the fold change,
monomer ID, gene ID, TU index, bulk ID, and IDs of complexes containing that
monomer as well as the number of monomers incorporated into each complex.

Required files:
- tetFC.tsv: fold change for each gene
- complexation_stoich.npy: complexation stoichiometric matrix
- complexation_molecules.npy: list of molecules used by complexation process
- TU_id_to_index.json: dictionary mapping RNA names to TU indexes (Example: {"EG10001_RNA[c]": 0})
"""

import pandas as pd
import json
import numpy as np

def main():
    rnas = pd.read_table("reconstruction/ecoli/flat/rnas.tsv", comment='#')
    model_degenes = []


    # Use fold change from exposure to 1.5 mg/L tetracycline
    tet_FC = pd.read_table("ecoli/experiments/marA_binding/tet_FC.tsv")
    tet_FC = tet_FC.loc[:,["Gene Name ", "1.5 mg/L tet."]]
    tet_FC.rename(columns={
        "Gene Name ": "Gene name", "1.5 mg/L tet.": "Fold change"}, inplace=True)
    degenes = tet_FC.sort_values(
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

    with open("data/wcecoli_t0.json") as f:
        initial_state = json.load(f)
    bulk_names = list(initial_state["bulk"].keys())

    # Include complexes
    # These numpy arrays were saved directly from complexation sim data
    comp_stoich = np.load('ecoli/experiments/marA_binding/complexation_stoich.npy')
    comp_molecules = np.load('ecoli/experiments/marA_binding/complexation_molecules.npy')
    comp_rxns = pd.DataFrame(comp_stoich, columns=comp_molecules)

    def recursive_search(complex_name, monomers_used):
        add_monomers_used = []
        add_complex_names = []
        if complex_name in comp_rxns.columns:
            stoich = comp_rxns.loc[comp_rxns[complex_name]<0, :]
            for i in range(stoich.shape[0]):
                curr_stoich = stoich.iloc[i, :]
                product = curr_stoich.loc[curr_stoich>0]
                add_monomers_used.append(int(curr_stoich[complex_name]*np.abs(monomers_used)/product[0]))
                add_complex_names.append(product.index.array[0])
        
        if len(add_complex_names) > 0:
            for i, add_complex_name in enumerate(add_complex_names):
                more_monomers_used, more_complex_names = recursive_search(add_complex_name, add_monomers_used[i])
                add_monomers_used += more_monomers_used
                add_complex_names += more_complex_names
        
        return add_monomers_used, add_complex_names
        

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
        for bulk_id in degene_bulk_ids:
            if bulk_id in comp_rxns.columns:
                stoich = comp_rxns.loc[comp_rxns[bulk_id]<0, :]
                for i in range(len(stoich)):
                    curr_stoich = stoich.iloc[i,:]
                    monomers_used.append(curr_stoich[bulk_id])
                    complex_names.append(curr_stoich.loc[curr_stoich>0].index.array[0])
                
        for i, complex_name in enumerate(complex_names):
            add_monomers_used, add_complex_names = recursive_search(complex_name, monomers_used[i])
            monomers_used += add_monomers_used
            complex_names += add_complex_names
        
        degene_complex_ids = []
        degene_monomers_used = []
        for bulk_name in bulk_names:
            for monomer_used, complex_name in zip(
                monomers_used, complex_names):
                if complex_name in bulk_name:
                    degene_complex_ids.append(bulk_name)
                    degene_monomers_used.append(monomer_used)
        
        return [degene_bulk_ids, degene_monomers_used, degene_complex_ids]

    # Protein IDs have varied suffixes: brute force search
    all_ids = []
    for monomer_id in model_degenes["monomer_ids"]:
        all_ids.append(get_IDs(monomer_id))

    (model_degenes["bulk_ids"], model_degenes["monomers_used"], 
        model_degenes["complex_ids"]) = zip(
            *[ids for ids in all_ids])
        
    # Add marR-tet complex
    marR_complex_ids = model_degenes.loc[model_degenes['Gene name'] == 'marR', 'complex_ids'].to_numpy()[0]
    marR_complex_ids.append('marR-tet[c]')
    marR_monomers_used = model_degenes.loc[model_degenes['Gene name'] == 'marR', 'monomers_used'].to_numpy()[0]
    marR_monomers_used.append(-2)
        
    model_degenes.to_csv(
        "ecoli/experiments/marA_binding/model_degenes.csv", index=False)

if __name__=="__main__":
    main()
