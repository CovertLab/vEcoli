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

    return outputs_df


def retrieve_tu_source(wd_raw):
    # reads and combines transcription units raw data

    tu_source_1 = pd.read_csv(
        os.path.join(wd_raw, "transcription_units.tsv"), sep="\t", header=5, index_col=0
    )
    tu_source_2 = pd.read_csv(
        os.path.join(wd_raw, "transcription_units_added.tsv"),
        sep="\t",
        header=0,
        index_col=0,
    )
    tu_source_2 = tu_source_2.drop("_comments", axis=1)

    tu_source = pd.concat([tu_source_1, tu_source_2], axis=0)
    return tu_source


def tu2gene_mapping(tu_ids, tu_source):
    # creates a mapping between transcription units and individual genes

    tu_ids_model = tu_ids

    tu_ids_source = [id[:-3] for id in tu_ids]

    tu_id2genes = []
    tu_id_missing = []

    for i in range(len(tu_ids_source)):
        try:
            tu_id_genes = tu_source["genes"][tu_ids_source[i]]
        except KeyError:
            tu_id_genes = f"[{tu_ids_source[i].replace('_RNA', '')}]"
            tu_id_missing.append(tu_ids_source[i])

        tu_id_genes = tu_id_genes[1:-1].replace('"', "").split(", ")
        tu_id2genes.append(tu_id_genes)

    tu_id2genes = dict(zip(tu_ids_model, tu_id2genes))
    genes_tu_all = np.unique(
        [gene for genes in list(tu_id2genes.values()) for gene in genes]
    ).tolist()

    return tu_id2genes, genes_tu_all


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

    rna_data = sim_data.process.transcription.rna_data

    tu_source = retrieve_tu_source(wd_raw)

    bulk_ids = get_bulk_ids(sim_data)

    # specify parquet columns
    output_columns = [
        "bulk",
        "listeners__rna_counts__full_mRNA_counts",
        "listeners__unique_molecule_counts__active_ribosome",
    ]

    output_df = read_outputs(history_sql, conn, output_columns)

    bulk_mtx = np.stack(output_df["bulk"].values)

    # retrieve mrnas
    mrna_mtx = np.stack(
        output_df["listeners__rna_counts__full_mRNA_counts"].values
    ).astype(int)

    mrna_tu_ids = rna_data["id"][rna_data["is_mRNA"]].tolist()

    tu2gene_mapping_mrna, genes_tu_mrna = tu2gene_mapping(
        tu_ids=mrna_tu_ids, tu_source=tu_source
    )

    tu_mrna_dict = {}

    for idx, mrna_tu_id in enumerate(mrna_tu_ids):
        tu_mrna_dict[mrna_tu_id] = mrna_mtx[:, idx]

    # retrieve processed rnas (trnas, rrnas)

    rna_ids_unprocessed = rna_data["id"][rna_data["is_unprocessed"]]
    rna_ids_mature = sim_data.process.transcription.mature_rna_data["id"]

    # retrieve processed trnas

    uncharged_trna_ids = sim_data.process.transcription.uncharged_trna_names
    charged_trna_ids = sim_data.process.transcription.charged_trna_names

    uncharged_trna_bulk_idxs = [bulk_ids.index(i) for i in uncharged_trna_ids]
    charged_trna_bulk_idxs = [bulk_ids.index(i) for i in charged_trna_ids]

    trna_total = (
        bulk_mtx[:, charged_trna_bulk_idxs] + bulk_mtx[:, uncharged_trna_bulk_idxs]
    )

    trna_processed_ids = list(filter(lambda x: x in rna_ids_mature, uncharged_trna_ids))

    trna_processed_idx = [uncharged_trna_ids.index(i) for i in trna_processed_ids]

    trna_processed_total = trna_total[:, trna_processed_idx]

    rna_processed_total = {}

    for trna_idx, trna_id in enumerate(trna_processed_ids):
        rna_processed_total[trna_id] = trna_processed_total[:, trna_idx]

    # add rrna to rna_processed total

    active_ribosome = output_df[
        "listeners__unique_molecule_counts__active_ribosome"
    ].values

    processed_rrna_ids = [
        sim_data.molecule_groups.s50_23s_rRNA,
        sim_data.molecule_groups.s30_16s_rRNA,
        sim_data.molecule_groups.s50_5s_rRNA,
    ]
    processed_rrna_ids = [item for sublist in processed_rrna_ids for item in sublist]

    processed_rrna_idxs = [bulk_ids.index(i) for i in processed_rrna_ids]

    bulk2monomers, all_monomers = build_bulk2monomers_matrix(sim_data)

    riboprotein_cplxs_ids = ["CPLX0-3953[c]", "CPLX0-3962[c]"]
    riboprotein_cplxs_idxs = [bulk_ids.index(i) for i in riboprotein_cplxs_ids]

    bulk_mtx_riboprotein_cplx = bulk_mtx[:, riboprotein_cplxs_idxs]

    bulk_total_riboprotein_cplx = np.array(
        [
            bulk_mtx_riboprotein_cplx[tp] + active_ribosome[tp]
            for tp in range(len(active_ribosome))
        ]
    )

    bulk_total_riboprotein_monomers = np.matmul(
        bulk_total_riboprotein_cplx, bulk2monomers[riboprotein_cplxs_idxs]
    )

    riboprotein_monomers_idx_rrna = [
        list(all_monomers).index(i) for i in processed_rrna_ids
    ]

    bulk_total_riboprotein_rrna = bulk_total_riboprotein_monomers[
        :, riboprotein_monomers_idx_rrna
    ]

    bulk_total_rrna = bulk_total_riboprotein_rrna + bulk_mtx[:, processed_rrna_idxs]

    for rrna_idx, rrna_id in enumerate(processed_rrna_ids):
        rna_processed_total[rrna_id] = bulk_total_rrna[:, rrna_idx]

    # reorder processed rrnas for rna maturaiton mtx
    rna_processed = {}

    for rna_id in rna_ids_mature.tolist():
        rna_processed[rna_id] = rna_processed_total[rna_id]

    rna_processed = np.stack(list(rna_processed.values())).transpose()

    rna_maturation_stoich_mtx = (
        sim_data.process.transcription.rna_maturation_stoich_matrix.toarray()
    )

    rna_processed_tu = np.matmul(rna_processed, rna_maturation_stoich_mtx)

    rna_processed_tu_dict = {}

    for rna_tu_idx, rna_tu in enumerate(rna_ids_unprocessed.tolist()):
        rna_processed_tu_dict[rna_tu] = rna_processed_tu[:, rna_tu_idx]

    rna_processed_tu_ids = list(rna_processed_tu_dict.keys())

    tu2gene_mapping_processed, genes_processed = tu2gene_mapping(
        rna_processed_tu_ids, tu_source
    )

    # add missing trna

    tu_idx_trna = np.where(rna_data.fullArray()["is_tRNA"])[0]

    tu_idx_not_unprocessed = np.where(~rna_data.fullArray()["is_unprocessed"])[0]

    trna_not_unprocessed_idx = np.intersect1d(tu_idx_trna, tu_idx_not_unprocessed)

    tu_id_trna_missing = rna_data["id"][trna_not_unprocessed_idx].tolist()

    missing_trna_genes = [trna_tu[:4] for trna_tu in tu_id_trna_missing]

    genes_input_raw = pd.read_csv(
        os.path.join(wd_raw, "genes.tsv"), sep="\t", header=5, index_col=0
    )

    missing_trna_gene_ids = {}

    missing_trna_genes_biocyc = []

    for idx, trna_gene in enumerate(missing_trna_genes):
        gene_id = genes_input_raw.index[genes_input_raw["symbol"] == trna_gene][0]
        missing_trna_gene_ids[tu_id_trna_missing[idx]] = [gene_id]
        missing_trna_genes_biocyc.append(gene_id)

    trna_missing_idx = [uncharged_trna_ids.index(i) for i in tu_id_trna_missing]

    trna_missing_counts = trna_total[:, trna_missing_idx]

    trna_missing_tu = {}

    for idx, trna_id in enumerate(tu_id_trna_missing):
        trna_missing_tu[trna_id] = trna_missing_counts[:, idx]

    tu_dict_full = {}
    tu_gene_mapping_full = {}
    for key in tu_mrna_dict.keys():
        tu_dict_full[key] = tu_mrna_dict[key]
        tu_gene_mapping_full[key] = tu2gene_mapping_mrna[key]

    for key in rna_processed_tu_dict.keys():
        tu_dict_full[key] = rna_processed_tu_dict[key]
        tu_gene_mapping_full[key] = tu2gene_mapping_processed[key]

    for key in trna_missing_tu.keys():
        tu_dict_full[key] = trna_missing_tu[key]
        tu_gene_mapping_full[key] = missing_trna_gene_ids[key]

    tu_genes_all = np.unique(
        genes_tu_mrna + genes_processed + missing_trna_genes_biocyc
    ).tolist()

    tu_gene_mtx = np.zeros([len(tu_dict_full), len(tu_genes_all)])

    for tu_idx, key in enumerate(tu_gene_mapping_full.keys()):
        genes_tu = tu_gene_mapping_full[key]
        genes_tu_idx = [tu_genes_all.index(g) for g in genes_tu]
        tu_gene_mtx[tu_idx, genes_tu_idx] = 1

    tu_counts_mtx = np.stack(list(tu_dict_full.values())).transpose()

    rna_counts_gene = np.matmul(tu_counts_mtx, tu_gene_mtx)

    n_tp = int(params["n_tp"])

    rna_counts_gene_blocksum = consolidate_timepoints(
        rna_counts_gene, n_tp, normalized=True
    )

    tp_columns = ["t" + str(i) for i in range(n_tp)]

    ptools_rna = pd.DataFrame(
        data=rna_counts_gene_blocksum.transpose(),
        columns=tp_columns,
        index=tu_genes_all,
    )

    ptools_rna.index.name = "$"

    ptools_rna.to_csv(
        os.path.join(outdir, "ptools_rna.txt"),
        sep="\t",
        index=True,
        header=True,
        float_format="%.4f",
    )
