"""
Multigeneration plot of mRNA cistron counts and protein monomer counts for a
user-supplied set of EcoCyc gene IDs. Useful for validating native-gene
perturbation variants (knockouts, knockdowns, overexpressions): the resulting
trajectories should track the multiplier applied at the translation level.

Two layouts are supported:

* ``per_gene: false`` (default) — one mRNA panel and one protein panel, with
  all requested genes overlaid as colored lines on each panel. Useful when
  comparing a small number of genes at similar expression levels.
* ``per_gene: true`` — one mRNA panel + one protein panel **per gene**,
  stacked vertically in a single long column. Each gene's y-axis scales
  independently, so a weakly-expressed gene isn't crushed by a strong one.

Config usage::

    "analysis_options": {
        "multigeneration": {
            "gene_counts": {
                "gene_ids": ["EG10527", "EG11015", "EG10001"],
                "per_gene": true
            }
        }
    }
"""

import altair as alt
import os
import pickle
from typing import Any, cast

from duckdb import DuckDBPyConnection
import polars as pl

from ecoli.library.parquet_emitter import (
    field_metadata,
    named_idx,
    open_arbitrary_sim_data,
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
    gene_ids = params.get("gene_ids")
    per_gene = bool(params.get("per_gene", False))
    if not gene_ids:
        print(
            "gene_counts analysis requires a non-empty 'gene_ids' list in "
            "analysis_options.multigeneration.gene_counts. Skipping."
        )
        return

    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)
    cistron_data = sim_data.process.transcription.cistron_data.struct_array
    monomer_data = sim_data.process.translation.monomer_data.struct_array

    gene_id_to_cistron_id = dict(zip(cistron_data["gene_id"], cistron_data["id"]))
    cistron_id_to_monomer_id = dict(zip(monomer_data["cistron_id"], monomer_data["id"]))

    requested_cistron_ids: list[str] = []
    requested_monomer_ids: list[str] = []
    requested_ecocyc_ids: list[str] = []
    unknown: list[str] = []
    non_coding: list[str] = []
    for ecocyc_id in gene_ids:
        cistron_id = gene_id_to_cistron_id.get(ecocyc_id)
        if cistron_id is None:
            unknown.append(ecocyc_id)
            continue
        monomer_id = cistron_id_to_monomer_id.get(cistron_id)
        if monomer_id is None:
            non_coding.append(ecocyc_id)
            continue
        requested_cistron_ids.append(cistron_id)
        requested_monomer_ids.append(monomer_id)
        requested_ecocyc_ids.append(ecocyc_id)

    if unknown:
        print(f"gene_counts: skipping unknown EcoCyc gene IDs: {unknown}")
    if non_coding:
        print(f"gene_counts: skipping non-coding cistrons (no monomer): {non_coding}")
    if not requested_cistron_ids:
        print("gene_counts: no resolvable coding gene IDs; nothing to plot.")
        return

    # mRNA cistron counts are indexed by cistron_id directly (one entry per
    # cistron, even on polycistronic mRNAs). The TU-level mRNA_counts listener
    # is indexed by transcription unit IDs and won't match cistron IDs for
    # polycistronic operons, so we use the cistron-level listener instead.
    mRNA_idx_dict = {
        cistron_id: i
        for i, cistron_id in enumerate(
            field_metadata(
                conn, config_sql, "listeners__rna_counts__mRNA_cistron_counts"
            )
        )
    }
    monomer_idx_dict = {
        monomer: i
        for i, monomer in enumerate(
            field_metadata(conn, config_sql, "listeners__monomer_counts")
        )
    }
    cistron_indexes = [
        cast(int, mRNA_idx_dict.get(cid)) for cid in requested_cistron_ids
    ]
    monomer_indexes = [
        cast(int, monomer_idx_dict.get(mid)) for mid in requested_monomer_ids
    ]

    missing_in_listeners = [
        cid for cid, idx in zip(requested_cistron_ids, cistron_indexes) if idx is None
    ] + [mid for mid, idx in zip(requested_monomer_ids, monomer_indexes) if idx is None]
    if missing_in_listeners:
        print(
            f"gene_counts: some IDs not present in listener field metadata: "
            f"{missing_in_listeners}. Skipping."
        )
        return

    mRNA_columns = named_idx(
        "listeners__rna_counts__mRNA_cistron_counts",
        requested_cistron_ids,
        [cistron_indexes],
    )
    monomer_columns = named_idx(
        "listeners__monomer_counts",
        requested_monomer_ids,
        [monomer_indexes],
    )
    data = read_stacked_columns(
        history_sql,
        [mRNA_columns, monomer_columns],
        conn=conn,
    )
    data = pl.DataFrame(data).with_columns(**{"Time (min)": pl.col("time") / 60})

    # Map listener field names back to the user's EcoCyc IDs for plot labels.
    cistron_to_ecocyc = dict(zip(requested_cistron_ids, requested_ecocyc_ids))
    monomer_to_ecocyc = dict(zip(requested_monomer_ids, requested_ecocyc_ids))

    mrna_long = (
        data.select(
            [
                "Time (min)",
                "lineage_seed",
                "generation",
                "agent_id",
                *requested_cistron_ids,
            ]
        )
        .unpivot(
            index=["Time (min)", "lineage_seed", "generation", "agent_id"],
            variable_name="cistron_id",
            value_name="mRNA count",
        )
        .with_columns(
            gene=pl.col("cistron_id").replace_strict(cistron_to_ecocyc),
        )
    )
    protein_long = (
        data.select(
            [
                "Time (min)",
                "lineage_seed",
                "generation",
                "agent_id",
                *requested_monomer_ids,
            ]
        )
        .unpivot(
            index=["Time (min)", "lineage_seed", "generation", "agent_id"],
            variable_name="monomer_id",
            value_name="protein count",
        )
        .with_columns(
            gene=pl.col("monomer_id").replace_strict(monomer_to_ecocyc),
        )
    )

    if per_gene:
        # One mRNA panel + one protein panel per gene, stacked into a single
        # long column. Independent y-scales mean weak and strong genes can be
        # read on the same page.
        sections: list[alt.Chart] = []
        for ecocyc_id in requested_ecocyc_ids:
            mrna_sub = mrna_long.filter(pl.col("gene") == ecocyc_id)
            protein_sub = protein_long.filter(pl.col("gene") == ecocyc_id)
            mrna_chart = (
                alt.Chart(mrna_sub)
                .mark_line()
                .encode(
                    x=alt.X("Time (min):Q"),
                    y=alt.Y("mRNA count:Q").title("mRNA cistron count"),
                    detail="agent_id:N",
                )
                .properties(
                    title=f"{ecocyc_id} — mRNA cistron counts",
                    width=600,
                    height=200,
                )
            )
            protein_chart = (
                alt.Chart(protein_sub)
                .mark_line()
                .encode(
                    x=alt.X("Time (min):Q"),
                    y=alt.Y("protein count:Q").title("protein monomer count"),
                    detail="agent_id:N",
                )
                .properties(
                    title=f"{ecocyc_id} — protein monomer counts",
                    width=600,
                    height=200,
                )
            )
            sections.append(
                alt.vconcat(mrna_chart, protein_chart).resolve_scale(
                    x="shared", y="independent"
                )
            )
        combined = alt.vconcat(*sections).resolve_scale(y="independent")
    else:
        mrna_plot = (
            alt.Chart(mrna_long)
            .mark_line()
            .encode(
                x=alt.X("Time (min):Q"),
                y=alt.Y("mRNA count:Q").title("mRNA Cistron Counts"),
                color=alt.Color("gene:N").title("Gene (EcoCyc ID)"),
            )
            .properties(title="mRNA Cistron Counts", width=600, height=250)
        )
        protein_plot = (
            alt.Chart(protein_long)
            .mark_line()
            .encode(
                x=alt.X("Time (min):Q"),
                y=alt.Y("protein count:Q").title("Protein Monomer Counts"),
                color=alt.Color("gene:N").title("Gene (EcoCyc ID)"),
            )
            .properties(title="Protein Monomer Counts", width=600, height=250)
        )
        combined = alt.vconcat(mrna_plot, protein_plot)
    combined.save(os.path.join(outdir, "gene_counts.html"))
