"""
Flux through reactions grouped into curated core metabolic pathways, and
percent flux utilization across pathway-branch-point nodes, for multivariant
simulations.

Pathway/reaction groupings (5 pilot pathways: Glycolysis, PP Pathway, ED
Pathway, TCA Cycle, Anaplerosis/Acetate) are loaded from
``validation/ecoli/flat/core_pathway_reactions.tsv``.

Two figures are saved:

- A box plot of base reaction flux per reaction, faceted by pathway and
  colored by variant.
- A stacked bar of % flux utilization across the immediate downstream fates
  of two metabolic hub nodes (G6P, and PEP/AcCoA), one bar per variant.
"""

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

import altair as alt
import plotly.express as px
import polars as pl

from ecoli.analysis.multivariant import _variant_label
from ecoli.library.parquet_emitter import (
    field_metadata,
    named_idx,
    read_stacked_columns,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

alt.data_transformers.enable("vegafusion")

PATHWAY_TSV = "validation/ecoli/flat/core_pathway_reactions.tsv"
PASTEL = px.colors.qualitative.Pastel

DEFAULT_FACET_COLUMNS = 3
DEFAULT_SUBPLOT_WIDTH = 350
DEFAULT_SUBPLOT_HEIGHT = 300

BRANCH_POINT_LABELS = {
    "G6P": "G6P node",
    "PEP_AcCoA": "PEP / AcCoA node",
}


def plot(
    params: dict[str, Any],
    conn: "DuckDBPyConnection",
    history_sql: str,
    config_sql: str,
    success_sql: str,
    sim_data_dict: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_names: dict[str, str],
) -> None:
    """
    All options have default values (do not need to be explicitly provided).

    Args:
        params: Dictionary of parameters given under analysis
            name in configuration JSON. Config options look like this:

            .. code-block:: json

                {
                    // Whether base reaction fluxes are raw counts requiring
                    // conversion to molar via counts_to_molar (true for
                    // MetabolismReduxClassic).
                    "is_reduxclassic": true,

                    // Facet grid columns for the pathway box plot.
                    "facet_columns": 3,

                    // Width/height of each subplot, in pixels.
                    "subplot_width": 350,
                    "subplot_height": 300,

                    // Which pathway_df column labels the box plot's x-axis:
                    // human-readable arrow notation ("reaction_label", e.g.
                    // "G6P -> F6P") or the raw base reaction ID
                    // ("reaction_id", e.g. "PGLUCISOM-RXN").
                    "label_by": "reaction_label"
                }
    """
    experiment_id = next(iter(variant_metadata.keys()), None)
    per_variant_params: dict[int, Any] = (
        variant_metadata[experiment_id] if experiment_id else {}
    )

    is_reduxclassic = params.get("is_reduxclassic", True)
    facet_columns = int(params.get("facet_columns", DEFAULT_FACET_COLUMNS))
    subplot_width = int(params.get("subplot_width", DEFAULT_SUBPLOT_WIDTH))
    subplot_height = int(params.get("subplot_height", DEFAULT_SUBPLOT_HEIGHT))
    label_by = params.get("label_by", "reaction_label")

    pathway_df = pl.read_csv(
        PATHWAY_TSV, separator="\t", comment_prefix="#", quote_char='"'
    )
    # A reaction_id can appear under more than one pathway (e.g. a shared
    # step like PEP -> Pyr); dedup here for the query/column-building steps,
    # then let the join against pathway_df below fan a shared reaction's
    # flux back out to each pathway it belongs to.
    all_reaction_ids = sorted(set(pathway_df["reaction_id"].to_list()))

    base_reaction_ids = field_metadata(
        conn, config_sql, "listeners__fba_results__base_reaction_fluxes"
    )
    reaction_idx = [base_reaction_ids.index(r) for r in all_reaction_ids]

    columns = [
        named_idx(
            "listeners__fba_results__base_reaction_fluxes",
            all_reaction_ids,
            [reaction_idx],
        ),
        "listeners__enzyme_kinetics__counts_to_molar AS counts_to_molar",
    ]
    raw = pl.DataFrame(
        read_stacked_columns(
            history_sql,
            columns,
            conn=conn,
            order_results=True,
            success_sql=success_sql,
            remove_first=is_reduxclassic,
        )
    )

    if raw.is_empty():
        print("pathway_flux_distribution: no rows returned; skipping.")
        return

    # Convert raw per-reaction values to molar flux (mmol/L/s) and take the
    # magnitude, since several Toya-convention reaction directions are
    # negative and both figures care about flux magnitude, not direction.
    if is_reduxclassic:
        raw = raw.with_columns(
            [
                (pl.col(r) * pl.col("counts_to_molar")).abs().alias(r)
                for r in all_reaction_ids
            ]
        )
    else:
        raw = raw.with_columns([pl.col(r).abs().alias(r) for r in all_reaction_ids])

    # Collapse from one row per timestep to one row per cell (mean flux over
    # its trajectory) before building per-reaction distributions, mirroring
    # mean_flux_distribution.py's per-cell averaging.
    id_vars = ["variant", "lineage_seed", "generation", "agent_id"]
    raw = raw.group_by(id_vars).agg(
        [pl.col(r).mean().alias(r) for r in all_reaction_ids]
    )

    unique_variants = sorted(raw["variant"].unique().to_list())
    variant_labels = {}
    for variant_val in unique_variants:
        label_l = _variant_label(variant_val, per_variant_params)
        variant_labels[variant_val] = (
            " ".join(label_l) if isinstance(label_l, list) else label_l
        )

    flux_long = raw.select(id_vars + all_reaction_ids).melt(
        id_vars=id_vars,
        value_vars=all_reaction_ids,
        variable_name="reaction_id",
        value_name="flux",
    )
    flux_long = flux_long.join(
        pathway_df.select(
            ["pathway", "reaction_id", "reaction_label", "order", "branch_point"]
        ),
        on="reaction_id",
    )
    flux_long = flux_long.with_columns(
        pl.col("variant")
        .replace_strict(variant_labels, return_dtype=pl.Utf8)
        .alias("variant_label")
    )

    # ── Figure 1: box plot of flux per reaction, faceted by pathway ────────
    box_pdf = flux_long.to_pandas()
    label_sort_order = pathway_df.sort(["pathway", "order"])[label_by].to_list()
    variant_label_list = [variant_labels[v] for v in unique_variants]
    variant_color_scale = alt.Scale(
        domain=variant_label_list, range=PASTEL[: len(variant_label_list)]
    )

    box_chart = (
        alt.Chart(box_pdf)
        .mark_boxplot(size=14)
        .encode(
            x=alt.X(
                f"{label_by}:N",
                sort=label_sort_order,
                title=None,
                axis=alt.Axis(labelAngle=-40, labelFontSize=10),
            ),
            y=alt.Y("flux:Q", title="Base reaction flux (mmol/L/s)"),
            color=alt.Color(
                "variant_label:N", title="Variant", scale=variant_color_scale
            ),
            tooltip=[
                alt.Tooltip("reaction_id:N", title="Reaction ID"),
                alt.Tooltip("reaction_label:N", title="Reaction"),
                alt.Tooltip("variant_label:N", title="Variant"),
                alt.Tooltip("flux:Q", title="Flux"),
            ],
        )
        .properties(width=subplot_width, height=subplot_height)
        .facet(facet=alt.Facet("pathway:N", title=None), columns=facet_columns)
        .resolve_scale(x="independent", y="independent")
        .properties(title="Distribution of Base Reaction Flux by Pathway")
    )

    box_out_path = os.path.join(outdir, "pathway_flux_boxplot.html")
    box_chart.save(box_out_path)
    print(f"Saved pathway flux box plot to {box_out_path}")

    # ── Figure 2: stacked % utilization across branch-point nodes ──────────
    util_long = flux_long.filter(pl.col("branch_point") != "")
    util_means = util_long.group_by(["variant_label", "branch_point", "pathway"]).agg(
        pl.col("flux").mean().alias("mean_flux")
    )
    util_means = util_means.with_columns(
        (
            pl.col("mean_flux")
            / pl.col("mean_flux").sum().over(["variant_label", "branch_point"])
            * 100
        ).alias("percent")
    )
    util_means = util_means.with_columns(
        pl.col("branch_point")
        .replace_strict(BRANCH_POINT_LABELS, return_dtype=pl.Utf8)
        .alias("branch_point_label")
    )
    util_pdf = util_means.to_pandas()
    pathway_domain = sorted(util_means["pathway"].unique().to_list())
    pathway_color_scale = alt.Scale(
        domain=pathway_domain, range=PASTEL[: len(pathway_domain)]
    )

    util_chart = (
        alt.Chart(util_pdf)
        .mark_bar()
        .encode(
            x=alt.X("variant_label:N", title="Variant"),
            y=alt.Y("percent:Q", title="% of branch-point flux"),
            color=alt.Color(
                "pathway:N", title="Downstream pathway", scale=pathway_color_scale
            ),
            tooltip=[
                alt.Tooltip("variant_label:N", title="Variant"),
                alt.Tooltip("pathway:N", title="Downstream pathway"),
                alt.Tooltip("percent:Q", title="Percent", format=".1f"),
            ],
        )
        .properties(width=subplot_width, height=subplot_height)
        .facet(facet=alt.Facet("branch_point_label:N", title=None), columns=2)
        .properties(title="Branch-Point Flux Utilization by Variant")
    )

    util_out_path = os.path.join(outdir, "pathway_flux_branch_utilization.html")
    util_chart.save(util_out_path)
    print(f"Saved pathway flux branch utilization to {util_out_path}")
