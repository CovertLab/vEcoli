"""
Plot unmet homeostatic need for metabolites, broken out per lineage_seed
instead of averaged across seeds.

This is a per-seed variant of metabolite_unmet_need.py, intended for
experiments that test a SINGLE weight combination across many seeds (e.g. a
multi-generation stability check) rather than a sweep of many weight
combinations -- for that use case, averaging away seed-to-seed variation (as
metabolite_unmet_need.py does) throws away exactly the thing being studied.

Produces two outputs:
- An Excel workbook (one row per lineage_seed, top1/2/3 metabolite + unmet
  need columns per fraction_kinetic_target level) covering ALL seeds present.
- A grid plot (rows = lineage_seed, capped via "max_seeds_to_plot" since a
  full ~300-seed grid isn't practically renderable; columns =
  fraction_kinetic_target level) of the same bar+line panel style as
  metabolite_unmet_need.py, with dashed vertical lines marking generation
  boundaries within each seed's continuous multi-generation timeline
  (structurally similar to multigeneration/selected_fluxes.py's
  _mark_generations(), adapted to Altair).

Known cost: grouping by lineage_seed (instead of collapsing across seeds)
multiplies the underlying aggregation's row count by roughly the seed count.
For large multi-seed experiments this can need substantially more memory
than the collapsed-across-seeds query in metabolite_unmet_need.py -- use the
optional "lineage_seeds" param to restrict to a handful of seeds for a cheap
correctness-testing run before attempting the full seed set.
"""

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING, cast

import altair as alt
import pandas as pd
import polars as pl

from ecoli.library.parquet_emitter import (
    field_metadata,
    read_stacked_columns,
    skip_n_gens,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

alt.data_transformers.enable("vegafusion")

DEFAULT_TOP_N = 8
DEFAULT_SUBPLOT_WIDTH = 220
DEFAULT_TIME_BIN_MIN = 1.0
DEFAULT_MAX_SEEDS_TO_PLOT = 20
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


def _build_excel_df(
    seed_met_score: pl.DataFrame,
    per_variant_params: dict[int, Any],
    variant_ids: list[int],
    seed_ids: list[int],
) -> pd.DataFrame:
    """
    One row per lineage_seed, with top{1,2,3}_metabolite_<fraction> /
    top{1,2,3}_unmet_<fraction> columns per variant -- same naming convention
    as build_whole_cell_unmet_need_excel.py's build_sheet(), just with
    lineage_seed as the row key instead of a weight combo.
    """
    pdf = seed_met_score.to_pandas()
    rows = []
    for seed_val in seed_ids:
        row: dict[str, Any] = {"lineage_seed": seed_val}
        for variant_val in variant_ids:
            frac = per_variant_params[variant_val]["fraction_kinetic_target"]
            sub = pdf[
                (pdf["variant"] == variant_val) & (pdf["lineage_seed"] == seed_val)
            ].sort_values("rank")
            for i in (1, 2, 3):
                r = sub[sub["rank"] == i]
                row[f"top{i}_metabolite_{frac}"] = (
                    r["metabolite"].iloc[0] if not r.empty else None
                )
                row[f"top{i}_unmet_{frac}"] = (
                    float(r["mean_abs_unmet"].iloc[0]) if not r.empty else None
                )
        rows.append(row)
    return pd.DataFrame(rows)


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
    """One bar+line panel per (lineage_seed, variant), gridded by seed (row)
    and variant/fraction_kinetic_target (column)."""
    experiment_id = next(iter(variant_metadata.keys()), None)
    per_variant_params: dict[int, Any] = (
        {
            vid: p
            for vid, p in variant_metadata[experiment_id].items()
            if p != "baseline"
        }
        if experiment_id
        else {}
    )

    top_n = params.get("top_n", DEFAULT_TOP_N)
    metabolites_of_interest = params.get("metabolites_of_interest")
    subplot_width = int(params.get("subplot_width", DEFAULT_SUBPLOT_WIDTH))
    time_bin_min = float(params.get("time_bin_min", DEFAULT_TIME_BIN_MIN))
    max_seeds_to_plot = int(params.get("max_seeds_to_plot", DEFAULT_MAX_SEEDS_TO_PLOT))
    lineage_seeds = params.get("lineage_seeds")

    skip_n_gens_val = int(params.get("skip_n_gens", 0))
    if skip_n_gens_val > 0:
        history_sql = skip_n_gens(history_sql, skip_n_gens_val)

    try:
        homeostatic_ids = field_metadata(
            conn, config_sql, "listeners__fba_results__homeostatic_metabolite_counts"
        )
    except Exception:
        print(
            "metabolite_unmet_need_per_seed: "
            "listeners__fba_results__homeostatic_metabolite_counts "
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
            f"metabolite_unmet_need_per_seed: homeostatic metabolite not in bulk: "
            f"{e}; skipping."
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

    # Optional cheap-testing knob: restrict to a handful of seeds before
    # paying the full aggregation cost (see module docstring).
    if lineage_seeds:
        seed_list = ", ".join(str(int(s)) for s in lineage_seeds)
        subquery = f"SELECT * FROM ({subquery}) WHERE lineage_seed IN ({seed_list})"

    # ── Per-(variant, seed, time bin, metabolite) unmet need ──────────────
    # Same computation as metabolite_unmet_need.py, but lineage_seed is kept
    # in the final GROUP BY instead of being averaged away.
    agg = conn.sql(f"""
        WITH t_min AS (
            SELECT variant, lineage_seed, min(time) AS t_min
            FROM ({subquery})
            GROUP BY variant, lineage_seed
        ),
        unnested AS (
            SELECT
                s.variant,
                s.lineage_seed,
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
            SELECT variant, lineage_seed, "Time_min", met_idx,
                CASE
                    WHEN cnt = 0 THEN NULL
                    ELSE (tgt - est) / cnt / counts_to_molar
                END AS ratio
            FROM unnested
        )
        SELECT variant, lineage_seed, "Time_min", met_idx,
            avg(CASE WHEN isinf(ratio) THEN NULL ELSE ratio END) AS unmet_need
        FROM ratios
        GROUP BY variant, lineage_seed, "Time_min", met_idx
        ORDER BY variant, lineage_seed, "Time_min", met_idx
        """).pl()

    if agg.is_empty():
        print("metabolite_unmet_need_per_seed: no rows returned; skipping.")
        return

    met_df = pl.DataFrame(
        {"met_idx": list(range(1, n_met + 1)), "metabolite": homeostatic_ids}
    )
    agg = agg.join(met_df, on="met_idx").drop("met_idx")

    # Generation-boundary timestamps per (variant, seed), on the same
    # Time_min bin scale as the line charts, for the dashed vertical markers.
    gen_bounds = conn.sql(f"""
        WITH t_min AS (
            SELECT variant, lineage_seed, min(time) AS t_min
            FROM ({subquery})
            GROUP BY variant, lineage_seed
        )
        SELECT s.variant, s.lineage_seed, s.generation,
            min(floor((s.time - t.t_min) / 60.0 / {time_bin_min}) * {time_bin_min})
                AS gen_start_min
        FROM ({subquery}) s
        JOIN t_min t USING (variant, lineage_seed)
        GROUP BY s.variant, s.lineage_seed, s.generation
        ORDER BY s.variant, s.lineage_seed, s.generation
        """).pl()

    variant_ids = sorted(
        int(v)
        for v in agg["variant"].unique().to_list()
        if int(v) in per_variant_params
    )
    if not variant_ids:
        print(
            "metabolite_unmet_need_per_seed: no non-baseline variants found; skipping."
        )
        return
    variant_ids.sort(
        key=lambda v: -float(per_variant_params[v]["fraction_kinetic_target"])
    )
    seed_ids = sorted(int(s) for s in agg["lineage_seed"].unique().to_list())

    # ── Excel: top-3-per-seed, covering ALL seeds regardless of plot cap ──
    seed_met_score = (
        agg.group_by(["variant", "lineage_seed", "metabolite"])
        .agg(pl.col("unmet_need").abs().mean().alias("mean_abs_unmet"))
        .with_columns(
            pl.col("mean_abs_unmet")
            .rank(method="ordinal", descending=True)
            .over(["variant", "lineage_seed"])
            .alias("rank")
        )
        .filter(pl.col("rank") <= 3)
        .sort(["variant", "lineage_seed", "rank"])
    )
    excel_df = _build_excel_df(
        seed_met_score, per_variant_params, variant_ids, seed_ids
    )
    excel_path = os.path.join(outdir, "metabolite_unmet_need_per_seed.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        excel_df.to_excel(writer, sheet_name="per_seed_unmet_need", index=False)
    print(
        f"Saved metabolite unmet need per seed ({len(excel_df)} seeds) to {excel_path}"
    )

    # ── Plot: cap seeds for renderability, build per-(variant, seed) data ──
    plotted_seed_ids = seed_ids[:max_seeds_to_plot]
    if len(seed_ids) > max_seeds_to_plot:
        print(
            f"metabolite_unmet_need_per_seed: plotting first {max_seeds_to_plot} of "
            f"{len(seed_ids)} seeds (set params.max_seeds_to_plot to change)."
        )

    ordered_mets: list[str] = []
    data_by_pair: dict[tuple[int, int], tuple[pl.DataFrame, pl.DataFrame]] = {}
    for variant_val in variant_ids:
        for seed_val in plotted_seed_ids:
            sub = agg.filter(
                (pl.col("variant") == variant_val)
                & (pl.col("lineage_seed") == seed_val)
            )
            if sub.is_empty():
                continue
            met_score = (
                sub.group_by("metabolite")
                .agg(pl.col("unmet_need").abs().mean().alias("mean_abs_unmet"))
                .sort("mean_abs_unmet", descending=True)
            )
            top_mets = met_score.head(top_n)["metabolite"].to_list()
            line_mets = (
                metabolites_of_interest
                if metabolites_of_interest is not None
                else top_mets
            )
            line_mets = [m for m in line_mets if m in homeostatic_ids] or top_mets
            top_bar = met_score.filter(pl.col("metabolite").is_in(top_mets))
            agg_line = sub.filter(pl.col("metabolite").is_in(line_mets))
            for m in list(dict.fromkeys(top_mets + line_mets)):
                if m not in ordered_mets:
                    ordered_mets.append(m)
            data_by_pair[(variant_val, seed_val)] = (top_bar, agg_line)

    if not data_by_pair:
        print(
            "metabolite_unmet_need_per_seed: no per-seed data after aggregation; skipping plot."
        )
        return

    color_domain = ordered_mets
    color_range = [PASTEL[i % len(PASTEL)] for i in range(len(color_domain))]
    w = subplot_width

    def _build_cell(variant_val: int, seed_val: int) -> alt.VConcatChart | None:
        entry = data_by_pair.get((variant_val, seed_val))
        if entry is None:
            return None
        top_bar, agg_line = entry
        frac = per_variant_params[variant_val]["fraction_kinetic_target"]
        title = alt.TitleParams(
            text=f"seed {seed_val}",
            subtitle=f"fraction_kinetic_target={frac}",
        )
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
        bars = bar_base.mark_bar(cornerRadiusEnd=8, size=20).encode(
            y=alt.Y(
                "mean_abs_unmet:Q",
                title="Unmet need (mean |L1 diff|)",
                scale=alt.Scale(type="symlog"),
            ),
        )
        bar_labels = bar_base.mark_text(
            align="center", baseline="bottom", dy=-4, fontSize=10, fontWeight="bold"
        ).encode(
            y=alt.Y("mean_abs_unmet:Q", scale=alt.Scale(type="symlog")),
            text=alt.Text("mean_abs_unmet:Q", format=".2e"),
        )
        bar_chart = (bars + bar_labels).properties(height=180, width=w)

        line_base = alt.Chart(df_line).encode(
            x=alt.X("Time_min:Q", title="Time (min, seed-relative)"),
            y=alt.Y("unmet_need:Q", title="L1 |Target - Estimate|"),
            color=alt.Color(
                "metabolite:N",
                scale=alt.Scale(domain=color_domain, range=color_range),
                legend=alt.Legend(title="Metabolite"),
            ),
            tooltip=["Time_min:Q", "metabolite:N", "unmet_need:Q"],
        )
        line_chart = line_base.mark_line(strokeWidth=1.5).properties(
            height=180, width=w
        )

        rules_df = (
            gen_bounds.filter(
                (pl.col("variant") == variant_val)
                & (pl.col("lineage_seed") == seed_val)
            )
            .filter(pl.col("generation") > pl.col("generation").min())
            .to_pandas()
        )
        if not rules_df.empty:
            rule_layer = (
                alt.Chart(rules_df)
                .mark_rule(strokeDash=[4, 4], color="gray", opacity=0.6)
                .encode(x=alt.X("gen_start_min:Q"))
            )
            line_chart = alt.layer(line_chart, rule_layer).properties(
                height=180, width=w
            )

        return cast(
            alt.VConcatChart,
            alt.vconcat(bar_chart, line_chart, spacing=20).properties(title=title),
        )

    row_charts = []
    for seed_val in plotted_seed_ids:
        cells = [
            cell
            for variant_val in variant_ids
            if (cell := _build_cell(variant_val, seed_val)) is not None
        ]
        if cells:
            row_charts.append(alt.hconcat(*cells, spacing=30))

    if not row_charts:
        print("metabolite_unmet_need_per_seed: nothing to plot; skipping HTML output.")
        return

    grid = alt.vconcat(*row_charts, spacing=50).properties(
        title="Unmet homeostatic need by seed x fraction_kinetic_target"
    )
    html_path = os.path.join(outdir, "metabolite_unmet_need_per_seed.html")
    grid.save(html_path)
    print(
        f"Saved metabolite unmet need per seed grid "
        f"({len(plotted_seed_ids)} seeds x {len(variant_ids)} variants) to {html_path}"
    )
