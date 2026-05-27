"""
Multivariant overlay of mRNA cistron counts and protein monomer counts
across variants for a user-supplied set of EcoCyc gene IDs.

For each gene, produces one mRNA panel and one protein panel with every
variant's cells overlaid (color = variant). One variant is drawn as the
visual "reference" (thicker stroke, neutral color) so that perturbed
variants stand out against it. The panel pairs are stacked vertically
into a single long HTML column with independent y-axes per gene.

Config usage::

    "analysis_options": {
        "multivariant": {
            "gene_counts": {
                "gene_ids": ["EG10544", "EG10669", "EG11015"],
                "reference_variant": 0
            }
        }
    }
"""

from __future__ import annotations

import os
import pickle
from typing import Any, cast

import altair as alt
import polars as pl
from duckdb import DuckDBPyConnection

from ecoli.library.parquet_emitter import (
    field_metadata,
    named_idx,
    open_arbitrary_sim_data,
    read_stacked_columns,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_gene_ids(
    sim_data, gene_ids: list[str]
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    """Return (cistron_ids, monomer_ids, ecocyc_ids, unknown, non_coding)."""
    cd = sim_data.process.transcription.cistron_data.struct_array
    md = sim_data.process.translation.monomer_data.struct_array
    gene_id_to_cistron_id = dict(zip(cd["gene_id"], cd["id"]))
    cistron_id_to_monomer_id = dict(zip(md["cistron_id"], md["id"]))
    cistron_ids: list[str] = []
    monomer_ids: list[str] = []
    ecocyc_ids: list[str] = []
    unknown: list[str] = []
    non_coding: list[str] = []
    for ecocyc in gene_ids:
        cid = gene_id_to_cistron_id.get(ecocyc)
        if cid is None:
            unknown.append(ecocyc)
            continue
        mid = cistron_id_to_monomer_id.get(cid)
        if mid is None:
            non_coding.append(ecocyc)
            continue
        cistron_ids.append(cid)
        monomer_ids.append(mid)
        ecocyc_ids.append(ecocyc)
    return cistron_ids, monomer_ids, ecocyc_ids, unknown, non_coding


def _gene_overlay_chart(
    df: pl.DataFrame,
    y_col: str,
    y_title: str,
    title: str,
    reference_variant: int,
) -> alt.Chart:
    """
    Build a single overlay line chart. The reference variant gets a thicker
    stroke and a neutral colour; all other variants are coloured by the
    tableau10 categorical scheme. One line is drawn per cell
    (variant × lineage × generation × agent).
    """
    if df.is_empty():
        return (
            alt.Chart(pl.DataFrame({"msg": ["no data"]}))
            .mark_text(size=14, color="#888")
            .encode(text="msg:N")
            .properties(title=title, width=700, height=80)
        )
    ref = df.filter(pl.col("variant") == reference_variant)
    other = df.filter(pl.col("variant") != reference_variant)
    layers: list[alt.Chart] = []
    if not other.is_empty():
        layers.append(
            alt.Chart(other)
            .mark_line(opacity=0.85)
            .encode(
                x=alt.X("time_min:Q").title("Time (min)"),
                y=alt.Y(f"{y_col}:Q").title(y_title),
                color=alt.Color(
                    "variant:N",
                    scale=alt.Scale(scheme="tableau10"),
                    legend=alt.Legend(title="Variant"),
                ),
                strokeDash=alt.StrokeDash(
                    "variant:N", legend=alt.Legend(title="Variant style")
                ),
                detail="cell_id:N",
                tooltip=[
                    alt.Tooltip("variant:N"),
                    alt.Tooltip("cell_id:N"),
                    alt.Tooltip("time_min:Q", format=".1f"),
                    alt.Tooltip(f"{y_col}:Q", format=".3f"),
                ],
            )
        )
    if not ref.is_empty():
        layers.append(
            alt.Chart(ref)
            .mark_line(color="#000000", strokeWidth=2.4, opacity=0.95)
            .encode(
                x=alt.X("time_min:Q"),
                y=alt.Y(f"{y_col}:Q"),
                detail="cell_id:N",
                tooltip=[
                    alt.Tooltip(
                        "variant:N", title=f"variant (reference = {reference_variant})"
                    ),
                    alt.Tooltip("cell_id:N"),
                    alt.Tooltip("time_min:Q", format=".1f"),
                    alt.Tooltip(f"{y_col}:Q", format=".3f"),
                ],
            )
        )
    return alt.layer(*layers).properties(title=title, width=700, height=240)


# ---------------------------------------------------------------------------
# Main plot entry point
# ---------------------------------------------------------------------------


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
    if not gene_ids:
        print(
            "multivariant gene_counts requires a non-empty 'gene_ids' list in "
            "analysis_options.multivariant.gene_counts. Skipping."
        )
        return
    reference_variant = int(params.get("reference_variant", 0))

    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)
    cistron_ids, monomer_ids, ecocyc_ids, unknown, non_coding = _resolve_gene_ids(
        sim_data, gene_ids
    )
    if unknown:
        print(f"multivariant gene_counts: skipping unknown EcoCyc IDs: {unknown}")
    if non_coding:
        print(
            f"multivariant gene_counts: skipping non-coding cistrons (no monomer): "
            f"{non_coding}"
        )
    if not cistron_ids:
        print(
            "multivariant gene_counts: no resolvable coding gene IDs; nothing to plot."
        )
        return

    # Listener indices (cistron-level for mRNA, monomer-level for protein).
    mRNA_idx_dict = {
        c: i
        for i, c in enumerate(
            field_metadata(
                conn, config_sql, "listeners__rna_counts__mRNA_cistron_counts"
            )
        )
    }
    monomer_idx_dict = {
        m: i
        for i, m in enumerate(
            field_metadata(conn, config_sql, "listeners__monomer_counts")
        )
    }
    cistron_indexes = [mRNA_idx_dict.get(c) for c in cistron_ids]
    monomer_indexes = [monomer_idx_dict.get(m) for m in monomer_ids]
    missing = [c for c, i in zip(cistron_ids, cistron_indexes) if i is None] + [
        m for m, i in zip(monomer_ids, monomer_indexes) if i is None
    ]
    if missing:
        print(
            f"multivariant gene_counts: some IDs not in listener field metadata: "
            f"{missing}. Skipping."
        )
        return

    mRNA_cols = named_idx(
        "listeners__rna_counts__mRNA_cistron_counts",
        cistron_ids,
        [cast(list, cistron_indexes)],
    )
    monomer_cols = named_idx(
        "listeners__monomer_counts",
        monomer_ids,
        [cast(list, monomer_indexes)],
    )
    data = read_stacked_columns(history_sql, [mRNA_cols, monomer_cols], conn=conn)
    data = (
        pl.DataFrame(data)
        .with_columns(time_min=pl.col("time") / 60)
        .with_columns(
            cell_id=pl.format(
                "{}_{}_{}_{}",
                pl.col("variant"),
                pl.col("lineage_seed"),
                pl.col("generation"),
                pl.col("agent_id"),
            )
        )
    )
    if data.is_empty():
        print("multivariant gene_counts: empty history; nothing to plot.")
        return

    cistron_to_ecocyc = dict(zip(cistron_ids, ecocyc_ids))
    monomer_to_ecocyc = dict(zip(monomer_ids, ecocyc_ids))

    index_cols = [
        "variant",
        "lineage_seed",
        "generation",
        "agent_id",
        "cell_id",
        "time_min",
    ]
    mrna_long = (
        data.select(index_cols + cistron_ids)
        .unpivot(
            index=index_cols,
            variable_name="cistron_id",
            value_name="mRNA count",
        )
        .with_columns(gene=pl.col("cistron_id").replace_strict(cistron_to_ecocyc))
    )
    protein_long = (
        data.select(index_cols + monomer_ids)
        .unpivot(
            index=index_cols,
            variable_name="monomer_id",
            value_name="protein count",
        )
        .with_columns(gene=pl.col("monomer_id").replace_strict(monomer_to_ecocyc))
    )

    # Build per-gene panel pairs and stack vertically.
    sections: list[alt.Chart] = []
    for ecocyc in ecocyc_ids:
        mrna_sub = mrna_long.filter(pl.col("gene") == ecocyc)
        protein_sub = protein_long.filter(pl.col("gene") == ecocyc)
        sections.append(
            alt.vconcat(
                _gene_overlay_chart(
                    mrna_sub,
                    "mRNA count",
                    "mRNA cistron count",
                    f"{ecocyc} — mRNA cistron counts",
                    reference_variant,
                ),
                _gene_overlay_chart(
                    protein_sub,
                    "protein count",
                    "protein monomer count",
                    f"{ecocyc} — protein monomer counts",
                    reference_variant,
                ),
            ).resolve_scale(x="shared", y="independent")
        )
    combined = alt.vconcat(*sections).resolve_scale(y="independent")
    output_path = os.path.join(outdir, "gene_counts.html")
    combined.save(output_path)
    print(
        f"multivariant gene_counts: wrote {output_path} "
        f"({os.path.getsize(output_path) / 1e6:.1f} MB) — "
        f"reference variant = {reference_variant}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _build_stub_sim_data():
    """Minimal sim_data stub for testing _resolve_gene_ids."""
    import numpy as np

    class Stub:
        pass

    cistron_dtype = [("id", "U16"), ("gene_id", "U16")]
    cistrons = np.array(
        [
            ("lacZ", "EG10001"),
            ("rrsA", "EG10002"),  # non-coding (no monomer)
            ("rpoB", "EG10003"),
        ],
        dtype=cistron_dtype,
    )
    monomer_dtype = [("id", "U24"), ("cistron_id", "U16")]
    monomers = np.array(
        [
            ("lacZ[c]", "lacZ"),
            ("rpoB[c]", "rpoB"),
        ],
        dtype=monomer_dtype,
    )
    sd = Stub()
    sd.process = Stub()
    sd.process.transcription = Stub()
    sd.process.transcription.cistron_data = Stub()
    sd.process.transcription.cistron_data.struct_array = cistrons
    sd.process.translation = Stub()
    sd.process.translation.monomer_data = Stub()
    sd.process.translation.monomer_data.struct_array = monomers
    return sd


def test_resolve_gene_ids_partitions():
    sd = _build_stub_sim_data()
    cids, mids, eids, unk, nc = _resolve_gene_ids(
        sd, ["EG10001", "EG10003", "EG10002", "EG_NOPE"]
    )
    assert cids == ["lacZ", "rpoB"]
    assert mids == ["lacZ[c]", "rpoB[c]"]
    assert eids == ["EG10001", "EG10003"]
    assert nc == ["EG10002"]
    assert unk == ["EG_NOPE"]


def test_gene_overlay_chart_empty_returns_placeholder():
    chart = _gene_overlay_chart(
        pl.DataFrame(
            schema={
                "variant": pl.Int64,
                "time_min": pl.Float64,
                "mRNA count": pl.Float64,
                "cell_id": pl.Utf8,
            }
        ),
        "mRNA count",
        "mRNA cistron count",
        "EG10001 — mRNA cistron counts",
        reference_variant=0,
    )
    assert chart is not None
