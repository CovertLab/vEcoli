"""
Plot unmet homeostatic need for metabolites for multivariant simulation.

For each variant, shows a bar chart of the top-N metabolites by mean |unmet need|
and a timeseries of unmet need, aggregated across all cells in that variant.
One bar+line subplot per variant, stacked vertically.

Configure the number of initial generations to exclude from the averages using
the "skip_n_gens" key in the analysis params (defaults to 0).

To reduce emit and process data size, the per-metabolite unmet-need ratio,
time binning (defaults to 1 minute), and averaging are all computed in SQL
(DuckDB) via `UNNEST` + `GROUP BY` (same pattern as `average_monomer_counts.py`)
"""

from __future__ import annotations

import json
import os
from typing import Any, TYPE_CHECKING, cast

import altair as alt
import polars as pl

from ecoli.analysis.multivariant.utils import compute_variant_grid, create_variant_label
from ecoli.library.parquet_emitter import (
    field_metadata,
    read_stacked_columns,
    skip_n_gens,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

alt.data_transformers.enable("vegafusion")

DEFAULT_TOP_N = 8
DEFAULT_SUBPLOT_WIDTH = 600
DEFAULT_TIME_BIN_MIN = 1.0
DEFAULT_KD1_UNMET_NEED_THRESHOLD = 100.0
PASTEL = [
    "#8dd3c7",
    "#EECE9D",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
]


def _format_int_list(values: list[int]) -> str:
    """
    Format a sorted list of ints compactly, collapsing contiguous runs for
    making the seed and generation labels in the title.

    e.g. ``[0, 1, 2, 3]`` -> ``"0-3"``, ``[3, 5, 6, 7]`` -> ``"3, 5-7"``.
    """
    if not values:
        return "none"
    runs: list[tuple[int, int]] = []
    start = prev = values[0]
    for v in values[1:]:
        if v == prev + 1:
            prev = v
            continue
        runs.append((start, prev))
        start = prev = v
    runs.append((start, prev))
    return ", ".join(str(a) if a == b else f"{a}–{b}" for a, b in runs)


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
    """One bar+line subplot per variant, aggregated across all cells in that variant."""
    experiment_id = next(iter(variant_metadata.keys()), None)
    per_variant_params: dict[int, Any] = (
        variant_metadata[experiment_id] if experiment_id else {}
    )

    top_n = params.get("top_n", DEFAULT_TOP_N)
    metabolites_of_interest = params.get("metabolites_of_interest")
    subplot_width = int(params.get("subplot_width", DEFAULT_SUBPLOT_WIDTH))
    time_bin_min = float(params.get("time_bin_min", DEFAULT_TIME_BIN_MIN))

    # Number of initial generations to exclude before computing averages
    # (defaults to 0 if not provided in the config):
    skip_n_gens_val = int(params.get("skip_n_gens", 0))
    if skip_n_gens_val > 0:
        history_sql = skip_n_gens(history_sql, skip_n_gens_val)

    try:
        homeostatic_ids = field_metadata(
            conn, config_sql, "listeners__fba_results__homeostatic_metabolite_counts"
        )
    except Exception:
        print(
            "metabolite_unmet_need: listeners__fba_results__homeostatic_metabolite_counts "
            "not in config (e.g. non-metabolism_redux); skipping."
        )
        return

    bulk_ids = field_metadata(conn, config_sql, "bulk")
    try:
        homeostatic_bulk_idx_1based = [
            bulk_ids.index(met_id) + 1 for met_id in homeostatic_ids
        ]
    except ValueError as e:
        print(
            f"metabolite_unmet_need: homeostatic metabolite not in bulk: {e}; skipping."
        )
        return

    n_met = len(homeostatic_ids)

    query_cols = [
        "time",
        "generation",
        "lineage_seed",
        "listeners__fba_results__estimated_homeostatic_dmdt AS estimated_dmdt",
        "listeners__fba_results__target_homeostatic_dmdt AS target_dmdt",
        f"list_select(bulk, {homeostatic_bulk_idx_1based}) AS homeostatic_counts",
        "listeners__enzyme_kinetics__counts_to_molar AS counts_to_molar",
    ]
    subquery = read_stacked_columns(
        history_sql,
        query_cols,
        order_results=False,
        success_sql=success_sql,
        remove_first=True,
    )

    # ── Per-(variant, time bin, metabolite) average unmet need ────────────────
    # Computed necessary average unmet need entirely in SQL via UNNEST + GROUP BY
    agg = conn.sql(f"""
        WITH t_min AS (
            SELECT variant, lineage_seed, min(time) AS t_min
            FROM ({subquery})
            GROUP BY variant, lineage_seed
        ),
        unnested AS (
            SELECT
                s.variant,
                floor((s.time - t.t_min) / 60.0 / {time_bin_min}) * {time_bin_min}
                    AS "Time_min",
                generate_subscripts(s.estimated_dmdt, 1) AS met_idx,
                unnest(s.estimated_dmdt) AS est,
                unnest(s.target_dmdt) AS tgt,
                unnest(s.homeostatic_counts) AS cnt,
                s.counts_to_molar
            FROM ({subquery}) s
            JOIN t_min t USING (variant, lineage_seed)
        ),
        ratios AS (
            SELECT variant, "Time_min", met_idx,
                CASE
                    WHEN cnt = 0 THEN NULL
                    ELSE (tgt - est) / cnt / counts_to_molar
                END AS ratio
            FROM unnested
        )
        SELECT variant, "Time_min", met_idx,
            avg(CASE WHEN isinf(ratio) THEN NULL ELSE ratio END) AS unmet_need
        FROM ratios
        GROUP BY variant, "Time_min", met_idx
        ORDER BY variant, "Time_min", met_idx
        """).pl()

    if agg.is_empty():
        print("metabolite_unmet_need: no rows returned; skipping.")
        return

    met_df = pl.DataFrame(
        {"met_idx": list(range(n_met)), "metabolite": homeostatic_ids}
    )
    agg = agg.join(met_df, on="met_idx").drop("met_idx")

    variants = agg["variant"].unique().sort()

    # Obtain which seeds and generations actually feed each variant's averages
    coverage = conn.sql(f"""
        SELECT DISTINCT variant, lineage_seed, generation FROM ({subquery})
        """).pl()
    variant_coverage: dict[int, tuple[list[int], list[int]]] = {}
    for variant_val in variants:
        cov = coverage.filter(pl.col("variant") == variant_val)
        seeds = sorted(int(s) for s in cov["lineage_seed"].unique().to_list())
        gens = sorted(int(g) for g in cov["generation"].unique().to_list())
        variant_coverage[int(variant_val)] = (seeds, gens)

    # Collect metabolites used across all variants for a shared color scale
    ordered_mets: list[str] = []
    data_by_variant: dict[int, tuple[pl.DataFrame, pl.DataFrame]] = {}
    total_unmet_by_variant: dict[int, float] = {}
    for variant_val in variants:
        sub = agg.filter(pl.col("variant") == variant_val)
        if sub.is_empty():
            continue
        met_score = (
            sub.group_by("metabolite")
            .agg(pl.col("unmet_need").abs().mean().alias("mean_abs_unmet"))
            .sort("mean_abs_unmet", descending=True)
        )
        total_unmet_by_variant[int(variant_val)] = met_score["mean_abs_unmet"].sum()
        top_mets = met_score.head(top_n)["metabolite"].to_list()
        line_mets = (
            metabolites_of_interest if metabolites_of_interest is not None else top_mets
        )
        line_mets = [m for m in line_mets if m in homeostatic_ids]
        if not line_mets:
            line_mets = top_mets
        top_bar = met_score.filter(pl.col("metabolite").is_in(top_mets))
        agg_line = sub.filter(pl.col("metabolite").is_in(line_mets))
        for m in list(dict.fromkeys(top_mets + line_mets)):
            if m not in ordered_mets:
                ordered_mets.append(m)
        data_by_variant[int(variant_val)] = (top_bar, agg_line)

    if not data_by_variant:
        print("metabolite_unmet_need: no per-variant data after aggregation; skipping.")
        return

    # Split variants into weight-combo groups (first of the two sweep params;
    # e.g. objective weights x fraction_kinetic_target -- see
    # compute_variant_grid) and route a whole group to a separate "_high"
    # plot if its KD=1.0 variant (the lowest variant id in the group) has
    # total unmet need above the threshold, instead of bloating the main plot.
    kd1_threshold = float(
        params.get("kd1_unmet_need_threshold", DEFAULT_KD1_UNMET_NEED_THRESHOLD)
    )
    weight_groups: dict[str, list[int]] = {}
    for vid, vparams in per_variant_params.items():
        if not isinstance(vparams, dict) or len(vparams) != 2:
            continue
        first_key = next(iter(vparams))
        weight_groups.setdefault(
            json.dumps(vparams[first_key], sort_keys=True), []
        ).append(vid)
    excluded_variants: set[int] = set()
    n_excluded_groups = 0
    for group_ids in weight_groups.values():
        kd1_variant = min(group_ids)
        if total_unmet_by_variant.get(kd1_variant, 0.0) > kd1_threshold:
            excluded_variants.update(group_ids)
            n_excluded_groups += 1
    if excluded_variants:
        print(
            f"metabolite_unmet_need: {n_excluded_groups} group(s) "
            f"({len(excluded_variants)} variant(s)) moved to "
            f"metabolite_unmet_need_high.html (KD=1.0 total unmet need > {kd1_threshold})."
        )

    color_domain = ordered_mets
    color_range = [PASTEL[i % len(PASTEL)] for i in range(len(color_domain))]
    w = subplot_width

    _, columns, ordered_variant_ids = compute_variant_grid(per_variant_params)

    def _build_and_save(
        ids_to_plot: list[int], out_name: str, title_suffix: str
    ) -> None:
        subplot_charts: list[alt.VConcatChart] = []
        for variant_val in ids_to_plot:
            entry = data_by_variant.get(variant_val)
            if entry is None:
                continue
            top_bar, agg_line = entry
            label = create_variant_label(variant_val, per_variant_params)
            seeds, gens = variant_coverage[int(variant_val)]
            subtitle = (
                f"seeds: {_format_int_list(seeds)}  |  "
                f"generations: {_format_int_list(gens)} | unmet needs averaged over "
                f"{len(seeds) * len(gens)} cells"
            )
            title = alt.TitleParams(text=label, subtitle=subtitle)
            df_bar = top_bar.to_pandas()
            df_line = agg_line.to_pandas()

            bar_base = alt.Chart(df_bar).encode(
                x=alt.X("metabolite:N", title="Metabolite", sort="-y"),
                color=alt.Color(
                    "metabolite:N",
                    scale=alt.Scale(domain=color_domain, range=color_range),
                    legend=None,
                ),
                tooltip=["metabolite:N", "mean_abs_unmet:Q"],
            )

            bars = bar_base.mark_bar(cornerRadiusEnd=8, size=28).encode(
                y=alt.Y(
                    "mean_abs_unmet:Q",
                    title="Unmet need (mean |L1 diff|)",
                    scale=alt.Scale(type="symlog"),
                ),
            )

            bar_labels = bar_base.mark_text(
                align="center",
                baseline="bottom",
                dy=-4,
                fontSize=12,
                fontWeight="bold",
            ).encode(
                y=alt.Y("mean_abs_unmet:Q", scale=alt.Scale(type="symlog")),
                text=alt.Text("mean_abs_unmet:Q", format=".2e"),
            )

            bar_chart = (bars + bar_labels).properties(height=220, width=w)

            line_chart = (
                alt.Chart(df_line)
                .mark_line(strokeWidth=2)
                .encode(
                    x=alt.X("Time_min:Q", title="Time (min)"),
                    y=alt.Y("unmet_need:Q", title="L1 |Target - Estimate|"),
                    color=alt.Color(
                        "metabolite:N",
                        scale=alt.Scale(domain=color_domain, range=color_range),
                        legend=alt.Legend(title="Metabolite"),
                    ),
                    tooltip=["Time_min:Q", "metabolite:N", "unmet_need:Q"],
                )
                .properties(height=300, width=w)
            )

            subplot_charts.append(
                cast(
                    alt.VConcatChart,
                    alt.vconcat(bar_chart, line_chart, spacing=50).properties(
                        title=title
                    ),
                )
            )

        if not subplot_charts:
            return
        combined = alt.concat(*subplot_charts, columns=columns).properties(
            title=f"Unmet homeostatic need by variant{title_suffix}"
        )
        out_path = os.path.join(outdir, out_name)
        combined.save(out_path)
        print(f"Saved metabolite unmet need (multivariant) to {out_path}")

    included_ids = [v for v in ordered_variant_ids if v not in excluded_variants]
    excluded_ids = [v for v in ordered_variant_ids if v in excluded_variants]
    _build_and_save(included_ids, "metabolite_unmet_need.html", "")
    _build_and_save(
        excluded_ids,
        "metabolite_unmet_need_high.html",
        f" (KD=1.0 total unmet need > {kd1_threshold})",
    )
