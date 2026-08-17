"""
Plot how much of WATER[c]'s dm/dt comes from ordinary metabolism vs. from
the WATER[p]<->WATER[c] diffusion pathway, for multivariant simulation.

WATER[c]'s total achieved dm/dt (``estimated_homeostatic_dmdt``) splits
cleanly into two additive components:

- **Water from diffusion (import/export)**: the net flux through the
  WATER[p]<->WATER[c] diffusion reaction (forward minus reverse) -- the
  passive-membrane-diffusion pathway modeled directly in
  ``ecoli/processes/metabolism_redux.py`` (see ``WATER_CORRECTION_MODE``).
  Positive means net import into the cytosol.
- **Water from metabolism**: everything else -- the incidental byproduct of
  ordinary biosynthesis/hydrolysis/maintenance reactions that happen to
  touch WATER[c], computed as the remainder
  (total achieved dm/dt - diffusion's contribution).

These two series sum exactly to the total achieved dm/dt by construction,
regardless of which WATER_CORRECTION_MODE produced the data.

One line subplot per variant, averaged across all cells in that variant
(aligned by time since birth).
"""

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

import altair as alt
import polars as pl

from ecoli.analysis.multivariant.utils import create_variant_label
from ecoli.library.parquet_emitter import field_metadata, read_stacked_columns

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

alt.data_transformers.enable("vegafusion")

WATER_ID = "WATER[c]"
WATER_DIFFUSION_RXN_ID = "TRANS-RXN0-547[CCO-PM-BAC-NEG]-WATER//WATER.29."
REVERSE_TAG = " (reverse)"
DEFAULT_SUBPLOT_WIDTH = 600


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
    """One line subplot per variant, aggregated across all cells in that variant."""
    experiment_id = next(iter(variant_metadata.keys()), None)
    per_variant_params: dict[int, Any] = (
        variant_metadata[experiment_id] if experiment_id else {}
    )

    subplot_width = int(params.get("subplot_width", DEFAULT_SUBPLOT_WIDTH))

    rxn_ids = field_metadata(
        conn, config_sql, "listeners__fba_results__estimated_fluxes"
    )
    diffusion_idx = []
    for rxn_id in (WATER_DIFFUSION_RXN_ID, WATER_DIFFUSION_RXN_ID + REVERSE_TAG):
        if rxn_id in rxn_ids:
            diffusion_idx.append(rxn_ids.index(rxn_id) + 1)
    if not diffusion_idx:
        print(
            "water_flux_impact: water diffusion reaction not found in "
            "estimated_fluxes metadata; skipping."
        )
        return
    # Net flux through the diffusion reaction: forward minus reverse.
    if len(diffusion_idx) == 2:
        diffusion_expr = (
            f"listeners__fba_results__estimated_fluxes[{diffusion_idx[0]}] - "
            f"listeners__fba_results__estimated_fluxes[{diffusion_idx[1]}]"
        )
    else:
        diffusion_expr = f"listeners__fba_results__estimated_fluxes[{diffusion_idx[0]}]"

    homeostatic_ids = field_metadata(
        conn, config_sql, "listeners__fba_results__homeostatic_metabolite_counts"
    )
    if WATER_ID not in homeostatic_ids:
        print(
            f"water_flux_impact: {WATER_ID} not in homeostatic metabolites; skipping."
        )
        return
    water_idx = homeostatic_ids.index(WATER_ID) + 1

    query_cols = [
        "time",
        "generation",
        "lineage_seed",
        "agent_id",
        f"({diffusion_expr}) AS water_from_diffusion",
        f"listeners__fba_results__estimated_homeostatic_dmdt[{water_idx}] AS water_total_dmdt",
    ]

    raw = pl.DataFrame(
        read_stacked_columns(
            history_sql,
            query_cols,
            conn=conn,
            order_results=True,
            success_sql=success_sql,
            remove_first=True,
        )
    )

    if raw.is_empty():
        print("water_flux_impact: no rows returned; skipping.")
        return

    raw = raw.with_columns(
        (pl.col("water_total_dmdt") - pl.col("water_from_diffusion")).alias(
            "water_from_metabolism"
        )
    )

    # Relative time per (variant, generation, lineage_seed, agent_id)
    t_min = raw.group_by(["variant", "generation", "lineage_seed", "agent_id"]).agg(
        pl.col("time").min().alias("t_min")
    )
    raw = raw.join(t_min, on=["variant", "generation", "lineage_seed", "agent_id"])
    raw = raw.with_columns(
        ((pl.col("time") - pl.col("t_min")) / 60.0).alias("Time_min")
    )

    agg = (
        raw.group_by("variant", "Time_min")
        .agg(
            pl.col("water_from_diffusion").mean().alias("water_from_diffusion"),
            pl.col("water_from_metabolism").mean().alias("water_from_metabolism"),
            pl.col("water_total_dmdt").mean().alias("water_total_dmdt"),
        )
        .sort("variant", "Time_min")
    )

    variants = agg["variant"].unique().sort()
    w = subplot_width

    series_domain = [
        "Water from metabolism",
        "Water from diffusion (import/export)",
        "Total achieved dm/dt",
    ]
    series_range = ["#fdb462", "#80b1d3", "#b3b3b3"]

    subplot_charts = []
    for variant_val in variants:
        sub = agg.filter(pl.col("variant") == variant_val)
        if sub.is_empty():
            continue
        label = create_variant_label(variant_val, per_variant_params)

        long = sub.select(
            [
                "Time_min",
                "water_from_metabolism",
                "water_from_diffusion",
                "water_total_dmdt",
            ]
        ).melt(
            id_vars=["Time_min"],
            value_vars=[
                "water_from_metabolism",
                "water_from_diffusion",
                "water_total_dmdt",
            ],
            variable_name="series",
            value_name="dmdt",
        )
        long = long.with_columns(
            pl.col("series")
            .replace(
                {
                    "water_from_metabolism": "Water from metabolism",
                    "water_from_diffusion": "Water from diffusion (import/export)",
                    "water_total_dmdt": "Total achieved dm/dt",
                }
            )
            .alias("series")
        )

        chart = (
            alt.Chart(long.to_pandas())
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("Time_min:Q", title="Time (min)"),
                y=alt.Y("dmdt:Q", title="WATER[c] dm/dt (counts/step)"),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(domain=series_domain, range=series_range),
                    legend=alt.Legend(title="Source"),
                ),
                strokeDash=alt.condition(
                    alt.datum.series == "Total achieved dm/dt",
                    alt.value([4, 4]),
                    alt.value([1, 0]),
                ),
                tooltip=["Time_min:Q", "series:N", "dmdt:Q"],
            )
            .properties(
                height=300,
                width=w,
                title=f"WATER[c] dm/dt: metabolism vs. diffusion - {label}",
            )
        )

        subplot_charts.append(chart)

    if not subplot_charts:
        print("water_flux_impact: no per-variant data after aggregation; skipping.")
        return

    combined = alt.vconcat(*subplot_charts).properties(
        title="WATER[c] dm/dt source (metabolism vs. diffusion), by variant"
    )

    out_path = os.path.join(outdir, "water_flux_impact.html")
    combined.save(out_path)
    print(f"Saved water flux impact (multivariant) to {out_path}")
