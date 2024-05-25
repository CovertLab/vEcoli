import os
from typing import Any

import polars as pl
import pickle
import hvplot.polars

from ecoli.analysis.template import get_field_metadata, named_idx


def plot(
    params: dict[str, Any],
    config_lf: pl.LazyFrame,
    history_lf: pl.LazyFrame,
    sim_data_paths: list[str],
    validation_data_paths: list[str],
    outdir: str,
):
    # Determine new gene ids
    with open(sim_data_paths[0], "rb") as f:
        sim_data = pickle.load(f)
    mRNA_sim_data = sim_data.process.transcription.cistron_data.struct_array
    monomer_sim_data = sim_data.process.translation.monomer_data.struct_array
    new_gene_mRNA_ids = mRNA_sim_data[mRNA_sim_data["is_new_gene"]]["id"].tolist()
    mRNA_monomer_id_dict = dict(
        zip(monomer_sim_data["cistron_id"], monomer_sim_data["id"])
    )
    new_gene_monomer_ids = [
        mRNA_monomer_id_dict.get(mRNA_id) for mRNA_id in new_gene_mRNA_ids
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
            get_field_metadata(config_lf, "listeners__rna_counts__mRNA_counts")
        )
    }
    new_gene_mRNA_indexes = [
        mRNA_idx_dict.get(mRNA_id) for mRNA_id in new_gene_mRNA_ids
    ]

    # Extract proein indexes for each new gene
    monomer_idx_dict = {
        monomer: i
        for i, monomer in enumerate(
            get_field_metadata(config_lf, "listeners__monomer_counts")
        )
    }
    new_gene_monomer_indexes = [
        monomer_idx_dict.get(monomer_id) for monomer_id in new_gene_monomer_ids
    ]

    # Load data
    columns = {
        "Time (min)": pl.col("time") / 60,
        **named_idx(
            "listeners__monomer_counts", new_gene_monomer_ids, new_gene_monomer_indexes
        )
        ** named_idx(
            "listeners__rna_counts__mRNA_counts",
            new_gene_mRNA_ids,
            new_gene_mRNA_indexes,
        ),
    }
    new_gene_data = (
        history_lf.select(**columns).sort("Time (min)").collect(streaming=True)
    )

    # mRNA counts
    new_gene_data = new_gene_data.rename(
        {
            "listeners__rna_counts__mRNA_counts" + mRNA_id: mRNA_id
            for mRNA_id in new_gene_mRNA_ids
        }
    )
    mrna_plot = new_gene_data.plot.line(
        x="Time (min)",
        y=[new_gene_mRNA_ids],
        ylabel="mRNA Counts",
        title="New Gene mRNA Counts",
    )

    # Protein counts
    new_gene_data = new_gene_data.rename(
        {
            "listeners__monomer_counts" + monomer_id: monomer_id
            for monomer_id in new_gene_monomer_ids
        }
    )
    protein_plot = new_gene_data.plot.line(
        x="Time (min)",
        y=[new_gene_monomer_ids],
        ylabel="Protein Counts",
        title="New Gene Protein Counts",
    )

    combined_plot = (mrna_plot + protein_plot).cols(1)
    hvplot.save(combined_plot, os.path.join(outdir, "new_gene_counts.html"))
