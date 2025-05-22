"""
====================================
Compile marA-Regulated Gene Metadata
====================================

This file contains functions to compile important metadata for genes that
are designed to be regulated by marA when the `mar_regulon` option is enabled.
For each gene, the output `gene_fc.csv` file contains the fold change,
monomer ID, gene ID, TU index, bulk ID, and IDs of complexes containing that
monomer as well as the number of monomers incorporated into each complex.

Required files:
- tetFC.tsv: fold change for each gene, extracted from Figure 1 of
 Viveiros M, Dupont M, Rodrigues L, Couto I, Davin-Regli A, et al. (2007)
 Antibiotic Stress, Genetic Response and Altered Permeability of E. coli.
 PLOS ONE 2(4): e365. https://doi.org/10.1371/journal.pone.0000365
"""

import argparse
import pandas as pd
import pickle
import json
import numpy as np
from ast import literal_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--sim_data_path",
        action="store",
        help="Path to the simulation data file from ParCa",
    )
    args = parser.parse_args()
    # Load complexation and TU index data from sim_data
    sim_data = pickle.load(open(args.sim_data_path, "rb"))
    bulk_names = sim_data.internal_state.bulk_molecules.bulk_data["id"].tolist()
    cistron_id_to_index = {
        cistron: idx
        for idx, cistron in enumerate(sim_data.process.transcription.cistron_data["id"])
    }
    cistron_tu_mapping = sim_data.process.transcription.cistron_tu_mapping_matrix
    comp_stoich = sim_data.process.complexation.stoich_matrix().astype(np.int64).T
    comp_molecules = [str(i) for i in sim_data.process.complexation.molecule_names]

    rnas = pd.read_table("reconstruction/ecoli/flat/rnas.tsv", comment="#")
    rnas["synonyms"] = rnas["synonyms"].apply(literal_eval)
    rnas = rnas.explode("synonyms")

    # Use fold change from exposure to 1.5 mg/L tetracycline
    tet_FC = pd.read_table("data/marA_binding/tet_FC.tsv")
    tet_FC = tet_FC.loc[:, ["Gene Name ", "1.5 mg/L tet."]]
    tet_FC.rename(
        columns={"Gene Name ": "Gene name", "1.5 mg/L tet.": "Fold change"},
        inplace=True,
    )
    de_genes = tet_FC.sort_values(by="Fold change", ascending=False, ignore_index=True)
    de_genes = de_genes.merge(rnas, left_on="Gene name", right_on="synonyms")

    # Delete these two duplicates that are the incorrect genes
    de_genes = de_genes.loc[
        ~((de_genes["Gene name"] == "acrE") & (de_genes["common_name"] == "acrB"))
    ]
    de_genes = de_genes.loc[
        ~((de_genes["Gene name"] == "acrB") & (de_genes["common_name"] == "gyrB"))
    ]

    # Get get TU index for genes
    # For genes in multiple TUs, take the first one
    cistron_idx = [cistron_id_to_index[rna_id] for rna_id in de_genes["id"]]
    row_id, tu_idx = cistron_tu_mapping[cistron_idx, :].nonzero()
    _, unique_idx = np.unique(row_id, return_index=True)
    de_genes["TU_idx"] = tu_idx[unique_idx]

    # Include complexes
    comp_rxns = pd.DataFrame(comp_stoich, columns=comp_molecules)

    def recursive_search(complex_name, monomers_used):
        add_monomers_used = []
        add_complex_names = []
        if complex_name in comp_rxns.columns:
            stoich = comp_rxns.loc[comp_rxns[complex_name] < 0, :]
            for i in range(stoich.shape[0]):
                curr_stoich = stoich.iloc[i, :]
                product = curr_stoich.loc[curr_stoich > 0]
                add_monomers_used.append(
                    int(curr_stoich[complex_name] * np.abs(monomers_used) / product[0])
                )
                add_complex_names.append(product.index.array[0])

        if len(add_complex_names) > 0:
            for i, add_complex_name in enumerate(add_complex_names):
                more_monomers_used, more_complex_names = recursive_search(
                    add_complex_name, add_monomers_used[i]
                )
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
                stoich = comp_rxns.loc[comp_rxns[bulk_id] < 0, :]
                for i in range(len(stoich)):
                    curr_stoich = stoich.iloc[i, :]
                    monomers_used.append(int(curr_stoich[bulk_id]))
                    complex_names.append(
                        curr_stoich.loc[curr_stoich > 0].index.array[0]
                    )

        for i, complex_name in enumerate(complex_names):
            add_monomers_used, add_complex_names = recursive_search(
                complex_name, monomers_used[i]
            )
            monomers_used += add_monomers_used
            complex_names += add_complex_names

        degene_complex_ids = []
        degene_monomers_used = []
        for bulk_name in bulk_names:
            for monomer_used, complex_name in zip(monomers_used, complex_names):
                if complex_name in bulk_name:
                    degene_complex_ids.append(bulk_name)
                    degene_monomers_used.append(monomer_used)

        return [degene_bulk_ids, degene_monomers_used, degene_complex_ids]

    # Protein IDs have varied suffixes: brute force search
    all_ids = []
    for monomer_id in de_genes["monomer_ids"]:
        all_ids.append(get_IDs(monomer_id))

    (de_genes["bulk_ids"], de_genes["monomers_used"], de_genes["complex_ids"]) = zip(
        *[ids for ids in all_ids]
    )

    # Add marR-tet complex
    marR_complex_ids = de_genes.loc[
        de_genes["Gene name"] == "marR", "complex_ids"
    ].to_numpy()[0]
    marR_complex_ids.append("marR-tet[c]")
    marR_monomers_used = de_genes.loc[
        de_genes["Gene name"] == "marR", "monomers_used"
    ].to_numpy()[0]
    marR_monomers_used.append(-2)

    de_genes.to_csv("data/marA_binding/gene_fc.csv", index=False)


if __name__ == "__main__":
    main()
