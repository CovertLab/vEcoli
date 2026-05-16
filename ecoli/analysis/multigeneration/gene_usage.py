import altair as alt
import os
from typing import Any, cast

from duckdb import DuckDBPyConnection
import pickle
import polars as pl

from ecoli.library.parquet_emitter import (
    field_metadata,
    open_arbitrary_sim_data,
    named_idx,
    read_stacked_columns,
)


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
    # Determine new gene ids
    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)
    mRNA_sim_data = sim_data.process.transcription.cistron_data.struct_array
    monomer_sim_data = sim_data.process.translation.monomer_data.struct_array
    new_gene_mRNA_ids = mRNA_sim_data[mRNA_sim_data["is_new_gene"]]["id"].tolist()
    mRNA_monomer_id_dict = dict(
        zip(monomer_sim_data["cistron_id"], monomer_sim_data["id"])
    )
    new_gene_monomer_ids = [
        cast(str, mRNA_monomer_id_dict.get(mRNA_id)) for mRNA_id in new_gene_mRNA_ids
    ]

    if len(new_gene_mRNA_ids) == 0:
        print(
            "This plot requires simulations where the new gene option was "
            "enabled, but no new gene mRNAs were found."
        )
        return
    if len(new_gene_monomer_ids) == 0:
        print(
            "This plot requires simulations where the new gene option was "
            "enabled, but no new gene proteins were found."
        )
        return
    if len(new_gene_mRNA_ids) != len(new_gene_monomer_ids):
        print("The number of new gene monomers and mRNAs should be equal.")

    # Extract mRNA indexes for each new gene
    mRNA_idx_dict = {
        rna[:-3]: i
        for i, rna in enumerate(
            field_metadata(conn, config_sql, "listeners__rna_counts__mRNA_counts")
        )
    }
    new_gene_mRNA_indexes = [
        cast(int, mRNA_idx_dict.get(mRNA_id)) for mRNA_id in new_gene_mRNA_ids
    ]

    # Extract protein indexes for each new gene
    monomer_idx_dict = {
        monomer: i
        for i, monomer in enumerate(
            field_metadata(conn, config_sql, "listeners__monomer_counts")
        )
    }
    new_gene_monomer_indexes = [
        cast(int, monomer_idx_dict.get(monomer_id))
        for monomer_id in new_gene_monomer_ids
    ]

    # Load data
    new_monomers = named_idx(
        "listeners__monomer_counts", new_gene_monomer_ids, [new_gene_monomer_indexes]
    )
    new_mRNAs = named_idx(
        "listeners__rna_counts__mRNA_counts", new_gene_mRNA_ids, [new_gene_mRNA_indexes]
    )
    new_gene_data = read_stacked_columns(
        history_sql,
        [new_monomers, new_mRNAs],
        conn=conn,
    )
    new_gene_data = pl.DataFrame(new_gene_data).with_columns(
        **{"Time (min)": pl.col("time") / 60}
    )

    # mRNA counts
    mrna_plot = new_gene_data.plot.line(
        x="Time (min)",
        y=alt.Y(new_gene_mRNA_ids).title("mRNA Counts"),
    ).properties(title="New Gene mRNA Counts")

    # Protein counts
    protein_plot = new_gene_data.plot.line(
        x="Time (min)",
        y=alt.Y(new_gene_monomer_ids).title("Protein Counts"),
    ).properties(title="New Gene Protein Counts")

    combined_plot = alt.vconcat(mrna_plot, protein_plot)
    combined_plot.save(os.path.join(outdir, "new_gene_counts.html"))
