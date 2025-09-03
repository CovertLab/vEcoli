import os
from typing import Any

from duckdb import DuckDBPyConnection
import numpy as np
import pandas as pd

from ecoli.library.sim_data import LoadSimData


def build_query(
    columns, history_sql
):  # generates sql query for user specified parquet columns
    query_sql = f"""
        SELECT {",".join(columns)}, time FROM ({history_sql})
        ORDER BY time
    """

    return query_sql


def read_outputs(
    history_sql: str,
    conn: DuckDBPyConnection,
    columns=["bulk", "listeners__rna_counts__full_mRNA_counts"],
):
    # retrieves specifc columns from parquet outputs and returns a dataframe
    query_sql = build_query(columns, history_sql)

    outputs_df = conn.sql(query_sql).df()

    outputs_df = outputs_df.groupby("time", as_index=False).sum()

    return outputs_df


def get_bulk_ids(sim_data):
    bulk_ids = sim_data.internal_state.bulk_molecules.bulk_data["id"].tolist()
    return bulk_ids


def build_bulk2monomers_matrix(sim_data):
    # decomplexes bulk species into monomers

    bulk_ids = sim_data.internal_state.bulk_molecules.bulk_data["id"].tolist()
    get_monomers = sim_data.process.complexation.get_monomers
    all_monomers = [list(get_monomers(bulk_id)["subunitIds"]) for bulk_id in bulk_ids]
    all_monomers = [item for sublist in all_monomers for item in sublist]
    all_monomers = list(np.unique(all_monomers))

    bulk2monomers = np.zeros((len(bulk_ids), len(all_monomers)))

    for idx, bulk_id in enumerate(bulk_ids):
        monomer_mapping = get_monomers(bulk_id)
        subunits = monomer_mapping["subunitIds"]
        stoich_coeffs = monomer_mapping["subunitStoich"]

        for j in range(len(subunits)):
            subunit = subunits[j]
            monomer_idx = all_monomers.index(subunit)
            bulk2monomers[idx, monomer_idx] = stoich_coeffs[j]

    return bulk2monomers, all_monomers


def consolidate_timepoints(state_mtx, n_tp, normalized=False):
    # generate consolidated relative time points
    checkpoints = np.linspace(0, np.shape(state_mtx)[0], n_tp, dtype=int)

    if normalized:
        denom = [
            len(state_mtx[checkpoints[i] : checkpoints[i + 1]])
            for i in range(len(checkpoints) - 1)
        ]

        block_sums = [
            state_mtx[checkpoints[i] : checkpoints[i + 1]].sum(axis=0) / denom[i]
            for i in range(len(checkpoints) - 1)
        ]

    else:
        block_sums = [
            state_mtx[checkpoints[i] : checkpoints[i + 1]].sum(axis=0)
            for i in range(len(checkpoints) - 1)
        ]

    block_sums = np.stack(block_sums, axis=0)
    block_sums_final = np.insert(block_sums, 0, state_mtx[0], axis=0)

    return block_sums_final


def plot(
    params: dict[str, Any],
    conn: DuckDBPyConnection,
    history_sql: str,
    config_sql: str,
    success_sql: str,
    sim_data_paths: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_names: dict[str, str],
):
    exp_id = list(sim_data_paths.keys())[0]

    wd_top = os.getcwd().split("/out/")[0]

    wd_raw = os.path.join(wd_top, "reconstruction", "ecoli", "flat")

    sim_data_path = list(sim_data_paths[exp_id].values())[0]

    sim_data = LoadSimData(sim_data_path).sim_data

    bulk_ids = get_bulk_ids(sim_data)

    genes_input_raw = pd.read_csv(
        os.path.join(wd_raw, "genes.tsv"), sep="\t", header=5, index_col=0
    )

    output_columns = [
        "bulk",
        "listeners__unique_molecule_counts__oriC",
        "listeners__unique_molecule_counts__active_RNAP",
        "listeners__unique_molecule_counts__active_ribosome",
    ]
    translation_module = sim_data.process.translation.monomer_data.fullArray()

    replisome_monomer_subunits = sim_data.molecule_groups.replisome_monomer_subunits
    replisome_trimer_subunits = sim_data.molecule_groups.replisome_trimer_subunits
    riboproteins = [
        sim_data.molecule_ids.s30_full_complex,
        sim_data.molecule_ids.s50_full_complex,
    ]
    rnap_id = sim_data.molecule_ids.full_RNAP

    output_df = read_outputs(history_sql, conn, output_columns)

    bulk_mtx = np.stack(output_df["bulk"].values)

    for bulk_id in replisome_monomer_subunits:
        unique_complex = output_df["listeners__unique_molecule_counts__oriC"].values
        add_bulk = unique_complex * 2
        bulk_idx = bulk_ids.index(bulk_id)
        bulk_mtx[:, bulk_idx] = bulk_mtx[:, bulk_idx] + add_bulk

    for bulk_id in replisome_trimer_subunits:
        unique_complex = output_df["listeners__unique_molecule_counts__oriC"].values
        add_bulk = unique_complex * 6
        bulk_idx = bulk_ids.index(bulk_id)
        bulk_mtx[:, bulk_idx] = bulk_mtx[:, bulk_idx] + add_bulk

    for bulk_id in riboproteins:
        unique_complex = output_df[
            "listeners__unique_molecule_counts__active_ribosome"
        ].values
        add_bulk = unique_complex
        bulk_idx = bulk_ids.index(bulk_id)
        bulk_mtx[:, bulk_idx] = bulk_mtx[:, bulk_idx] + add_bulk

    rnap_counts = output_df["listeners__unique_molecule_counts__active_RNAP"].values
    rnap_idx = bulk_ids.index(rnap_id)
    bulk_mtx[:, rnap_idx] = bulk_mtx[:, rnap_idx] + rnap_counts

    bulk2monomers, all_monomers = build_bulk2monomers_matrix(sim_data)

    rna2genes = {}

    for gene_id in genes_input_raw.index:
        rna_id = genes_input_raw.loc[gene_id, "rna_ids"][2:-2]
        rna2genes[rna_id] = gene_id

    protein_monomers = translation_module["id"]

    protein_monomer_idxs = np.array(
        [all_monomers.index(protein) for protein in protein_monomers]
    )

    bulk2protein_monomers = bulk2monomers[:, protein_monomer_idxs]

    cistrons = list(translation_module["cistron_id"])

    gene_labels = []

    for protein_idx, protein in enumerate(protein_monomers):
        cistron = cistrons[protein_idx]
        gene = rna2genes[cistron]
        gene_labels.append(gene)

    proteomics = np.matmul(bulk_mtx, bulk2protein_monomers)

    n_tp = int(params["n_tp"])

    tp_columns = ["t" + str(i) for i in range(n_tp)]

    proteomics_bulksum = consolidate_timepoints(proteomics, n_tp, normalized=True)

    ptools_proteins = pd.DataFrame(
        data=proteomics_bulksum.transpose(), index=gene_labels, columns=tp_columns
    )

    ptools_proteins.index.name = "$"

    ptools_proteins.to_csv(
        os.path.join(outdir, "ptools_proteins_multigen.txt"),
        sep="\t",
        index=True,
        header=True,
        float_format="%.4f",
    )
