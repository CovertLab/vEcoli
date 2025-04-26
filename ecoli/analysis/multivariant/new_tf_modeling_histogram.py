"""
Anaylsis Script for saving all protein counts impacted by new genes. Note that
this file only works for saving data for simulations that contain at least two
variants (one control and one experimental variant).
"""

import os

# noinspection PyUnresolvedReferences
from duckdb import DuckDBPyConnection
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, cast

from ecoli.library.parquet_emitter import (
    read_stacked_columns,
    get_field_metadata,
    named_idx
)


IGNORE_FIRST_N_GENS = 4
"""
Indicate which generation the data should start being collected from (sometimes 
this number should be greater than 0 because the first few generations may not 
be representative of the true dynamics occuring in the cell).
"""

COLORS = ["b", "r", "k", "g", 'y']


def plot(
    params: dict[str, Any],
    conn: DuckDBPyConnection,
    history_sql: str,
    config_sql: str,
    success_sql: str,
    sim_data_dict: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_names: dict[str, str],
):
    """
    Calculates average monomer counts per variant and saves them as separate
    files under ``{outdir}/saved_data/unfiltered_data``. Filters data to only
    include monomers whose counts are greater than ``params["filter_num"]``
    (default 0) and saves under ``{outdir}/saved_data/filtered_data``.
    """

    # with open_arbitrary_sim_data(sim_data_dict) as f:
    #    sim_data: "SimulationDataEcoli" = pickle.load(f)

    # So what I need is: for each cell, get the average of the total mRNA in the cell,
    # the fraction which purC occupies. This makes sense for a single time-point. For the
    # entire generation, we could either average this fraction across all time-points;
    # or take the sum of purC over the sum of total mRNA. The idea of the plot is,
    # here is purC expression levels when purR is on or off. Let's just do the averaged
    # fraction over time as of now.
    # And need to get it for each individual cell.

    # TODO: could do it for some monomer metric as well?

    gene_names = ["purC", "prmA", "lolB"]
    gene_to_id = {
        "purC": "TU00055[c]",
        "prmA": "TU00314[c]",
        "lolB": "TU0-8323[c]",
    }
    mRNA_idxs = cast(
        list[list[int]],
        get_indexes(
            conn, config_sql, "mRNA", [[gene_to_id[gene]] for gene in gene_names]
        ),
    )


    # purC_protein_id = "SAICARSYN-MONOMER[c]"
    # prmA_protein_id = "EG11497-MONOMER[c]"
    # lolB_protein_id = "EG11293-MONOMER[c]"
    # protein_idxs = cast(
    #     list[list[int]],
    #     get_indexes(
    #         conn,
    #         config_sql,
    #         "monomer",
    #         [purC_protein_id],
    #     ),
    # )

    mRNA_sql = get_per_cell_gene_count_fraction_sql(
        mRNA_idxs, "listeners__rna_counts__mRNA_counts", "mRNA"
    )
    # protein_sql = get_per_cell_gene_count_fraction_sql(
    #     protein_idxs, "listeners__monomer_counts", "monomer"
    # )
    # mRNA_names = named_idx(
    #     "listeners__rna_counts__mRNA_counts", ['purC', 'prmA', 'lolB'], [mRNAs[0]]
    # )

    subquery = read_stacked_columns(
        history_sql=history_sql,
        columns=["listeners__rna_counts__mRNA_counts"],#, "listeners__monomer_counts"],
        #remove_first=True,
    )

    mRNA_data = conn.sql(mRNA_sql.format(subquery=subquery)).arrow().to_pylist()
    #protein_data = conn.sql(protein_sql.format(subquery=subquery)).arrow().to_pylist()

    exp_id = mRNA_data[0]["experiment_id"]
    variant_name_exp = variant_names[exp_id]
    variant_metadata_exp = variant_metadata[exp_id]
    mRNA_variant_to_name = {
        i: variant_metadata_exp[i][variant_name_exp]
        for i in variant_metadata_exp.keys()
    }
    avg_fracs = {
        gene: {
            var: []
            for var in mRNA_variant_to_name.values()
        } for gene in gene_names
    }
    for data in mRNA_data:
        if data['generation'] > IGNORE_FIRST_N_GENS:
            if ((data["variant"] == 1) & (data["lineage_seed"] == 20)) | (
                    ((data["variant"] == 2) & (data["lineage_seed"] == 20)) | (
                    (data["variant"] == 3) & (data["lineage_seed"] == 20)
                )
            ):
                gene_idx = data["gene_idx"]
                variant_key = data["variant"]
                avg_fracs[gene_names[gene_idx]][mRNA_variant_to_name[variant_key]].append(data["avg_frac"])

    #protein_avg_fracs = np.array([x["avg_frac"] for x in protein_data])

    # Get variant names
    #protein_variants = np.array([x["variant"] for x in protein_data])
    fig, axs = plt.subplots(len(gene_names), figsize=(10, 5*len(gene_names)))

    var_to_plot_name = {
        "basal": "minimal",
        "with_aa": "minimal + aa",
        "with_aa_nucl": "minimal + aa + nucls",
        "acetate": "minimal acetate",
        "succinate": "minimal succinate"
    }

    for i, gene in enumerate(gene_names):
        for j, var in enumerate(list(mRNA_variant_to_name.values())):
            if len(avg_fracs[gene][var]) > 0:
                fracs = avg_fracs[gene][var]
                axs[i].hist(
                    fracs,
                    bins=np.linspace(0, 1.5*np.max(fracs), num=20),
                    alpha=0.5,
                    lw=3,
                    color=COLORS[j],
                    label=var_to_plot_name[var]
                )
            axs[i].set_title(f"{gene} fractional expression for different variants")
            axs[i].set_ylabel("Number of simulations")
            axs[i].set_xlabel(f"Time-average fraction of mRNA counts that contain {gene}")
            axs[i].legend()

            # protein_var_fracs = protein_avg_fracs[protein_variants == var]

        # axs[1].hist(
        #     protein_var_fracs,
        #     bins=20,
        #     alpha=0.5,
        #     lw=3,
        #     color=COLORS[i],
        #     label=str(var),
        # )

    # axs[1].set_title(
    #     "Histogram of time-average fraction of protein counts that are PurC, for different variants"
    # )
    # axs[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "new_tf_modeling_histogram.pdf"))


def get_per_cell_gene_count_fraction_sql(
    gene_indices: list[int] | list[list[int]], column: str, index_type: str
) -> str:
    """
    Construct generic SQL query that gets the average per cell of a select
    set of indices from a 1D list column divided by the total of all elements
    per row of that list column.

    Args:
        gene_indices: Indices to extract from 1D list column to get ratios for
        column: Name of 1D list column
        index_type: Can either be ``monomer`` or ``mRNA``. For ``monomer``,
            function works exactly as described above. For ``mRNA``,
            ``gene_indices`` will be a list of lists of mRNA indices. This is
            because one gene can have to multiple mRNAs (transcription units).
            Therefore, we sum the elements corresponding to each gene before
            proceeding (see :py:func:`~.get_rnas_combined_as_genes_projection`).
    """
    if index_type == "monomer":
        list_to_unnest = f"list_select({column}, {gene_indices})"
    else:
        list_to_unnest = (
            "["
            + ", ".join(
                [
                    f"list_sum(list_select({column}, {idx_for_one_gene}))"
                    for idx_for_one_gene in gene_indices
                ]
            )
            + "]"
        )
    return f"""
        WITH list_counts AS (
            SELECT {list_to_unnest} AS selected_counts, list_sum({column})
                AS total_counts, experiment_id, variant, lineage_seed,
                generation, agent_id
            FROM ({{subquery}})
        ),
        unnested_fracs AS (
            SELECT unnest(selected_counts) / total_counts AS gene_fracs,
                generate_subscripts(selected_counts, 1)-1 AS gene_idx,
                experiment_id, variant, lineage_seed, generation, agent_id
            FROM list_counts
        )
        SELECT avg(gene_fracs) AS 'avg_frac',
            experiment_id, variant, lineage_seed, generation, agent_id, gene_idx
        FROM unnested_fracs
        GROUP BY experiment_id, variant, lineage_seed,
            generation, agent_id, gene_idx
        """


def get_indexes(
    conn: DuckDBPyConnection,
    config_sql: str,
    index_type: str,
    ids: list[str] | list[list[str]],
) -> list[int | None] | list[list[int | None]]:
    """
    Retrieve DuckDB indices of a given type for a set of IDs. Note that
    DuckDB lists are 1-indexed.

    Args:
        conn: DuckDB database connection
        config_sql: DuckDB SQL query for sim config data (see
            :py:func:`~ecoli.library.parquet_emitter.get_dataset_sql`)
        index_type: Type of indices to return (one of ``cistron``,
            ``RNA``, ``mRNA``, or ``monomer``)
        ids: List of IDs to get indices for (must be monomer IDs
            if ``index_type`` is ``monomer``, else mRNA IDs)

    Returns:
        List of requested indexes
    """
    if index_type == "cistron":
        # Extract cistron indexes for each new gene
        cistron_idx_dict = {
            cis: i + 1
            for i, cis in enumerate(
                get_field_metadata(
                    conn, config_sql, "listeners__rnap_data__rna_init_event_per_cistron"
                )
            )
        }
        return [cistron_idx_dict.get(cistron) for cistron in ids]
    elif index_type == "RNA":
        # Extract RNA indexes for each new gene
        RNA_idx_dict = {
            rna: i + 1
            for i, rna in enumerate(
                get_field_metadata(
                    conn, config_sql, "listeners__rna_synth_prob__target_rna_synth_prob"
                )
            )
        }
        return [[RNA_idx_dict.get(rna_id) for rna_id in rna_ids] for rna_ids in ids]
    elif index_type == "mRNA":
        # Extract mRNA indexes for each new gene
        mRNA_idx_dict = {
            rna: i + 1
            for i, rna in enumerate(
                get_field_metadata(
                    conn, config_sql, "listeners__rna_counts__mRNA_counts"
                )
            )
        }
        return [[mRNA_idx_dict.get(rna_id) for rna_id in rna_ids] for rna_ids in ids]
    elif index_type == "monomer":
        # Extract protein indexes for each new gene
        monomer_idx_dict = {
            monomer: i + 1
            for i, monomer in enumerate(
                get_field_metadata(conn, config_sql, "listeners__monomer_counts")
            )
        }
        return [monomer_idx_dict.get(monomer_id) for monomer_id in ids]
    else:
        raise Exception(
            "Index type " + index_type + " has no instructions for data extraction."
        )
