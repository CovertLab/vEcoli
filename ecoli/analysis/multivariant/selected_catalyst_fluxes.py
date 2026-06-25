"""
Plot catalyst (enzyme) counts and the fluxes of the base reactions they
catalyze, across variants of a multivariant simulation.

This is the inverse of ``ecoli.analysis.single.selected_fluxes``: instead of
taking reaction IDs and (optionally) showing their catalysts, this takes
catalyst IDs and shows their counts alongside the fluxes of the base
reactions they catalyze.

One block per variant, stacked vertically. Each block is a 1x2 row: left
subplot shows catalyst counts, right subplot shows associated base reaction
fluxes. Within a block, color identifies the catalyst (shared between both
subplots); line style (dash) distinguishes between multiple reactions
catalyzed by the same catalyst. Lines represent the mean over all cells on a
continuous time axis (relative to the first timestep of each lineage seed);
generation is used as the detail encoding to draw separate line segments at
each cell division without resetting the time axis.
"""

from __future__ import annotations

import os
import pickle
from collections import defaultdict
from typing import Any, TYPE_CHECKING

import altair as alt
import plotly.express as px
import polars as pl

from ecoli.analysis.multivariant import _variant_label
from ecoli.library.parquet_emitter import (
    field_metadata,
    named_idx,
    open_arbitrary_sim_data,
    read_stacked_columns,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

alt.data_transformers.enable("vegafusion")

DEFAULT_SUBPLOT_WIDTH = 350
DEFAULT_SUBPLOT_HEIGHT = 300
PASTEL = px.colors.qualitative.Pastel
DASH_STYLES = [
    [1, 0],
    [6, 3],
    [2, 2],
    [4, 2, 1, 2],
    [1, 1],
    [8, 3, 1, 3],
]


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
                    // plot_catalysts: catalyst (enzyme) IDs to plot.
                    // Can be a list of string IDs...
                    "plot_catalysts": ["UDP-NACMURALA-GLU-LIG-MONOMER[c]"],

                    // ...or a dict of string IDs paired with human-readable
                    // labels.
                    "plot_catalysts": {
                        "UDP-NACMURALA-GLU-LIG-MONOMER[c]": "MurD"
                    },

                    // Width/height of each subplot, in pixels.
                    "subplot_width": 350,
                    "subplot_height": 300
                }
    """
    experiment_id = next(iter(variant_metadata.keys()), None)
    per_variant_params: dict[int, Any] = (
        variant_metadata[experiment_id] if experiment_id else {}
    )

    plot_catalysts = params.get("plot_catalysts", [])
    if not plot_catalysts:
        print("selected_catalyst_fluxes: no plot_catalysts given; skipping.")
        return
    if isinstance(plot_catalysts, dict):
        catalyst_labels = dict(plot_catalysts)
    else:
        catalyst_labels = dict(zip(plot_catalysts, plot_catalysts))
    requested_catalysts = list(catalyst_labels.keys())

    subplot_width = int(params.get("subplot_width", DEFAULT_SUBPLOT_WIDTH))
    subplot_height = int(params.get("subplot_height", DEFAULT_SUBPLOT_HEIGHT))

    # Get mappings from sim_data: reactions to catalysts, and fba reaction
    # ids to base reaction ids, then invert to get catalyst -> base reaction
    # ids (the mirror image of selected_fluxes.py's reaction -> catalysts).
    #
    # NOTE: Will NOT work in multivariant sims where the mapping from
    # reactions to catalysts changes between variants (currently, this is
    # not something we've had to model: the metabolism_redux_classic
    # variants only change lambda_* objective weights, not the
    # catalyst/reaction structure).
    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)
    reaction_catalysts = sim_data.process.metabolism.reaction_catalysts
    reaction_id_to_base_reaction_id = (
        sim_data.process.metabolism.reaction_id_to_base_reaction_id
    )

    catalyst_to_base_reactions: defaultdict[str, set[str]] = defaultdict(set)
    for rxn_id, catalysts in reaction_catalysts.items():
        base_id = reaction_id_to_base_reaction_id.get(rxn_id, rxn_id)
        for catalyst in catalysts:
            catalyst_to_base_reactions[catalyst].add(base_id)

    # Validate requested catalysts against listener metadata
    catalyst_ids = field_metadata(
        conn, config_sql, "listeners__fba_results__catalyst_counts"
    )
    missing_catalysts = [c for c in requested_catalysts if c not in catalyst_ids]
    if missing_catalysts:
        print(
            "selected_catalyst_fluxes: catalysts not found in catalyst_counts "
            f"metadata, skipping: {missing_catalysts}"
        )
        return

    base_reaction_ids = field_metadata(
        conn, config_sql, "listeners__fba_results__base_reaction_fluxes"
    )

    # Build catalyst -> ordered list of associated base reactions. Every
    # catalyst in catalyst_ids is guaranteed to catalyze at least one
    # reaction present in base_reaction_ids, since both are derived from the
    # same reaction_catalysts structure in sim_data.
    catalyst_to_reactions: dict[str, list[str]] = {
        catalyst: sorted(catalyst_to_base_reactions.get(catalyst, set()))
        for catalyst in requested_catalysts
    }

    all_reactions = sorted(
        {r for reactions in catalyst_to_reactions.values() for r in reactions}
    )

    # Pull catalyst counts and reaction fluxes in a single query
    catalyst_idx = [catalyst_ids.index(c) for c in requested_catalysts]
    reaction_idx = [base_reaction_ids.index(r) for r in all_reactions]
    columns = [
        named_idx(
            "listeners__fba_results__catalyst_counts",
            requested_catalysts,
            [catalyst_idx],
        ),
        named_idx(
            "listeners__fba_results__base_reaction_fluxes",
            all_reactions,
            [reaction_idx],
        ),
    ]

    raw = pl.DataFrame(
        read_stacked_columns(
            history_sql,
            columns,
            conn=conn,
            order_results=True,
            remove_first=True,
        )
    )

    if raw.is_empty():
        print("selected_catalyst_fluxes: no rows returned; skipping.")
        return

    # Continuous relative time per lineage_seed: subtract the global minimum
    # time for each seed so the x-axis starts at 0 and runs unbroken across
    # generations.
    min_t = raw.group_by(["lineage_seed"]).agg(pl.col("time").min().alias("t_min"))
    raw = raw.join(min_t, on=["lineage_seed"])
    raw = raw.with_columns(
        ((pl.col("time") - pl.col("t_min")) / 60).alias("Time (min)")
    )

    id_vars = ["Time (min)", "variant", "generation", "lineage_seed"]

    # catalyst columns are renamed raw ID -> display label after melting, so
    # both panels show the human-readable label (defaults to the ID itself)
    # while all internal lookups above stay keyed on the raw catalyst ID.
    count_long = (
        raw.select(id_vars + requested_catalysts)
        .melt(
            id_vars=id_vars,
            value_vars=requested_catalysts,
            variable_name="catalyst",
            value_name="count",
        )
        .with_columns(pl.col("catalyst").replace(catalyst_labels))
    )

    # Build the catalyst -> reaction -> dash_key mapping table, then join
    # against the melted flux data. A reaction shared by multiple requested
    # catalysts intentionally appears once per catalyst (mirrors how
    # selected_fluxes.py lets one catalyst be shared by multiple reactions).
    map_rows = []
    for catalyst in requested_catalysts:
        for dash_key, reaction in enumerate(catalyst_to_reactions[catalyst]):
            map_rows.append(
                {
                    "catalyst": catalyst_labels[catalyst],
                    "reaction": reaction,
                    "dash_key": str(dash_key),
                }
            )

    flux_long = raw.select(id_vars + all_reactions).melt(
        id_vars=id_vars,
        value_vars=all_reactions,
        variable_name="reaction",
        value_name="flux",
    )
    map_df = pl.DataFrame(map_rows)
    flux_long = flux_long.join(map_df, on="reaction")

    color_domain = [catalyst_labels[c] for c in requested_catalysts]
    color_range = [PASTEL[i % len(PASTEL)] for i in range(len(color_domain))]
    color_scale = alt.Scale(domain=color_domain, range=color_range)

    n_dash = max((len(r) for r in catalyst_to_reactions.values()), default=1)
    dash_domain = [str(i) for i in range(max(n_dash, 1))]
    dash_range = [DASH_STYLES[i % len(DASH_STYLES)] for i in range(len(dash_domain))]
    dash_scale = alt.Scale(domain=dash_domain, range=dash_range)

    variants = sorted(raw["variant"].unique().to_list())
    blocks = []
    for variant_val in variants:
        variant_label = _variant_label(variant_val, per_variant_params)
        if isinstance(variant_label, list):
            variant_label = " ".join(variant_label)

        count_pdf = count_long.filter(pl.col("variant") == variant_val).to_pandas()
        flux_pdf = flux_long.filter(pl.col("variant") == variant_val).to_pandas()

        count_chart = (
            alt.Chart(count_pdf)
            .mark_line(strokeWidth=1.5)
            .encode(
                x=alt.X("Time (min):Q", title="Time (min)"),
                y=alt.Y("mean(count):Q", title="Catalyst count"),
                color=alt.Color(
                    "catalyst:N", scale=color_scale, legend=alt.Legend(title="Catalyst")
                ),
                detail=alt.Detail("generation:N"),
                tooltip=[
                    alt.Tooltip("catalyst:N"),
                    alt.Tooltip("mean(count):Q", title="Mean count"),
                ],
            )
            .properties(
                width=subplot_width, height=subplot_height, title="Catalyst counts"
            )
        )

        flux_chart = (
            alt.Chart(flux_pdf)
            .mark_line(strokeWidth=1.5)
            .encode(
                x=alt.X("Time (min):Q", title="Time (min)"),
                y=alt.Y("mean(flux):Q", title="Base reaction flux (Count)"),
                color=alt.Color(
                    "catalyst:N", scale=color_scale, legend=alt.Legend(title="Catalyst")
                ),
                strokeDash=alt.StrokeDash("dash_key:N", scale=dash_scale, legend=None),
                detail=alt.Detail("generation:N"),
                tooltip=[
                    alt.Tooltip("reaction:N"),
                    alt.Tooltip("catalyst:N"),
                    alt.Tooltip("mean(flux):Q", title="Mean flux"),
                ],
            )
            .properties(
                width=subplot_width,
                height=subplot_height,
                title="Associated reaction fluxes",
            )
        )

        block = alt.hconcat(count_chart, flux_chart).properties(title=variant_label)
        blocks.append(block)

    final = alt.vconcat(*blocks).properties(
        title="Catalyst counts and associated reaction fluxes by variant"
    )

    out_path = os.path.join(outdir, "selected_catalyst_fluxes.html")
    final.save(out_path)
    print(f"Saved selected catalyst fluxes (multivariant) to {out_path}")
