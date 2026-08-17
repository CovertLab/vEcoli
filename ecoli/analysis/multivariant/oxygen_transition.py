"""
Plot the continuous aerobic-to-anaerobic oxygen transition produced by the
``oxygen_depletion`` variant (:py:mod:`ecoli.variants.oxygen_depletion`) and
:py:class:`~ecoli.processes.environment.oxygen_ramp.OxygenRamp`.

For each variant, plots (over absolute simulation time, not reset per
generation, since the ramp is scheduled in absolute time):

1. O2 exchange flux (mmol/g DCW/h), i.e. the actual FBA import rate that
   ``OXYGEN_SCALING_MODE == "continuous"`` caps -- NOT the periplasmic O2
   bulk count. The periplasmic pool is a small pass-through buffer: an
   internal transport reaction (``TRANS-RXN0-474``) moves O2 on to the
   cytoplasm almost as fast as it's imported, so the bulk count's own
   concentration target stays satisfiable with a tiny net flux regardless
   of how tightly the import *rate* is capped -- it does not decline even
   when the cap is working correctly, and is not a useful signal for this
   variant. Confirmed directly (2026-08) via a controlled diagnostic run:
   the LP's raw exchange-reaction flux matched the intended cap exactly,
   while the periplasmic bulk count stayed flat throughout.
2. Instantaneous growth rate, as the clearest downstream confirmation that
   the O2 restriction actually affects the cell (expect a modest decline,
   not dramatic, since carbon-source uptake is not correspondingly
   restricted -- the cell can partly compensate via anaerobic metabolism).
3. FNR active fraction (``FNR-4FE-4S-CPLX`` / all FNR species).
4. ArcA-P active fraction (``PHOSPHO-ARCA`` / (``PHOSPHO-ARCA`` +
   ``ARCA-MONOMER``)).

This is meant to confirm the transition is smooth end-to-end (media ->
O2 import rate -> TF activation / growth), not a discrete step, and to show
where the ramp's ``start_time``/``end_time`` window falls relative to the
downstream response. Panels 1-2 require ``ecoli-metabolism-redux`` (the
listeners they use aren't emitted by the older ``ecoli-metabolism``
process) -- skipped gracefully if unavailable.
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

N_AVOGADRO = 6.02214076e23
OXYGEN_EXCHANGE_DMDT_COL = (
    "listeners__fba_results__estimated_exchange_dmdt__OXYGEN-MOLECULE"
)
DRY_MASS_COL = "listeners__mass__dry_mass"
GROWTH_RATE_COL = "listeners__mass__instantaneous_growth_rate"
GROWTH_RATE_BIN_MIN = 5.0

# (display name, active id, [inactive ids to sum])
TFS = [
    (
        "FNR",
        "FNR-4FE-4S-CPLX[c]",
        ["FNR-4FE-4S-CPLX-OX[c]", "CPLX0-7797[c]"],
    ),
    ("ArcA", "PHOSPHO-ARCA[c]", ["ARCA-MONOMER[c]"]),
]
DEFAULT_SUBPLOT_WIDTH = 600


def _bulk_column_expr(bulk_ids: list[str], molecule_id: str, alias: str) -> str | None:
    """Returns a ``bulk[i] AS alias`` SQL expression, or None if missing."""
    try:
        idx = bulk_ids.index(molecule_id) + 1
    except ValueError:
        print(f"oxygen_transition: {molecule_id} not in bulk; skipping.")
        return None
    return f"bulk[{idx}] AS {alias}"


def _column_exists(
    conn: "DuckDBPyConnection", history_sql: str, column_name: str
) -> bool:
    described = conn.sql(f"DESCRIBE ({history_sql})").pl()
    return column_name in described["column_name"].to_list()


def _oxygen_flux_chart(
    raw: pl.DataFrame, per_variant_params: dict[int, Any], subplot_width: int
) -> list[alt.Chart]:
    """O2 exchange flux (mmol/g DCW/h) -- the actual capped import rate."""
    with_flux = raw.with_columns(
        (
            pl.col("o2_dmdt").abs()
            / N_AVOGADRO
            * 1000
            * 3600
            / (pl.col("dry_mass") * 1e-15)
        ).alias("o2_flux")
    )
    agg = (
        with_flux.group_by("variant", "Time_min")
        .agg(pl.col("o2_flux").mean().alias("o2_flux"))
        .sort("variant", "Time_min")
    )
    charts = []
    for variant_val in agg["variant"].unique().sort():
        sub = agg.filter(pl.col("variant") == variant_val)
        if sub.is_empty():
            continue
        variant_label = create_variant_label(variant_val, per_variant_params)
        chart = (
            alt.Chart(sub.to_pandas())
            .mark_line(strokeWidth=2, color="#4daf4a")
            .encode(
                x=alt.X("Time_min:Q", title="Time (min)"),
                y=alt.Y("o2_flux:Q", title="O2 exchange flux (mmol/gDCW/h)"),
                tooltip=["Time_min:Q", "o2_flux:Q"],
            )
            .properties(
                height=200,
                width=subplot_width,
                title=f"O2 import rate - {variant_label}",
            )
        )
        charts.append(chart)
    return charts


def _growth_rate_chart(
    raw: pl.DataFrame, per_variant_params: dict[int, Any], subplot_width: int
) -> list[alt.Chart]:
    """
    instantaneous_growth_rate is noisy at the single-timestep level (e.g.
    transient spikes right after division events), which swamps the
    modest, gradual decline this variant is expected to produce. Two
    passes to make the trend visible: (1) drop per-variant IQR outliers
    before aggregating, so isolated spikes don't pull bin means around;
    (2) aggregate into GROWTH_RATE_BIN_MIN-wide time bins (median, not
    mean, for additional robustness) instead of per-timestep points.
    """
    bounds = raw.group_by("variant").agg(
        pl.col("growth_rate").quantile(0.25).alias("q1"),
        pl.col("growth_rate").quantile(0.75).alias("q3"),
    )
    bounds = bounds.with_columns(
        (pl.col("q3") - pl.col("q1")).alias("iqr"),
    ).with_columns(
        (pl.col("q1") - 1.5 * pl.col("iqr")).alias("lo"),
        (pl.col("q3") + 1.5 * pl.col("iqr")).alias("hi"),
    )
    filtered = raw.join(bounds.select("variant", "lo", "hi"), on="variant").filter(
        pl.col("growth_rate").is_between(pl.col("lo"), pl.col("hi"))
    )
    agg = (
        filtered.with_columns(
            (pl.col("Time_min") // GROWTH_RATE_BIN_MIN * GROWTH_RATE_BIN_MIN).alias(
                "Time_min_bin"
            )
        )
        .group_by("variant", "Time_min_bin")
        .agg(pl.col("growth_rate").median().alias("growth_rate"))
        .rename({"Time_min_bin": "Time_min"})
        .sort("variant", "Time_min")
    )
    charts = []
    for variant_val in agg["variant"].unique().sort():
        sub = agg.filter(pl.col("variant") == variant_val)
        if sub.is_empty():
            continue
        variant_label = create_variant_label(variant_val, per_variant_params)
        chart = (
            alt.Chart(sub.to_pandas())
            .mark_line(strokeWidth=2, color="#377eb8")
            .encode(
                x=alt.X("Time_min:Q", title="Time (min)"),
                y=alt.Y(
                    "growth_rate:Q",
                    title=f"Growth rate (1/s), {GROWTH_RATE_BIN_MIN:g}-min median, outliers excluded",
                ),
                tooltip=["Time_min:Q", "growth_rate:Q"],
            )
            .properties(
                height=200,
                width=subplot_width,
                title=f"Growth rate - {variant_label}",
            )
        )
        charts.append(chart)
    return charts


def _tf_fraction_charts(
    tf_name: str,
    raw: pl.DataFrame,
    active_col: str,
    inactive_cols: list[str],
    per_variant_params: dict[int, Any],
    subplot_width: int,
) -> list[alt.Chart]:
    fraction_expr = pl.col(active_col) / (
        pl.col(active_col) + sum(pl.col(c) for c in inactive_cols)
    )
    with_fraction = raw.with_columns(fraction_expr.alias("active_fraction"))

    agg = (
        with_fraction.group_by("variant", "Time_min")
        .agg(pl.col("active_fraction").mean().alias("active_fraction"))
        .sort("variant", "Time_min")
    )

    charts = []
    for variant_val in agg["variant"].unique().sort():
        sub = agg.filter(pl.col("variant") == variant_val)
        if sub.is_empty():
            continue
        variant_label = create_variant_label(variant_val, per_variant_params)
        chart = (
            alt.Chart(sub.to_pandas())
            .mark_line(strokeWidth=2, color="#fb8072")
            .encode(
                x=alt.X("Time_min:Q", title="Time (min)"),
                y=alt.Y(
                    "active_fraction:Q",
                    title=f"{tf_name} active fraction",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                tooltip=["Time_min:Q", "active_fraction:Q"],
            )
            .properties(
                height=200,
                width=subplot_width,
                title=f"{tf_name} activity - {variant_label}",
            )
        )
        charts.append(chart)
    return charts


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
    """One subplot per variant for O2 import rate, growth rate, and each TF's active fraction."""
    experiment_id = next(iter(variant_metadata.keys()), None)
    per_variant_params: dict[int, Any] = (
        variant_metadata[experiment_id] if experiment_id else {}
    )
    subplot_width = int(params.get("subplot_width", DEFAULT_SUBPLOT_WIDTH))

    bulk_ids = field_metadata(conn, config_sql, "bulk")

    query_cols = ["time", "generation", "lineage_seed", "agent_id"]

    have_o2_flux = _column_exists(conn, history_sql, OXYGEN_EXCHANGE_DMDT_COL) and (
        _column_exists(conn, history_sql, DRY_MASS_COL)
    )
    if have_o2_flux:
        query_cols.append(f'"{OXYGEN_EXCHANGE_DMDT_COL}" AS o2_dmdt')
        query_cols.append(f'"{DRY_MASS_COL}" AS dry_mass')
    else:
        print(
            "oxygen_transition: O2 exchange flux listener not available "
            "(requires ecoli-metabolism-redux); skipping that panel."
        )

    have_growth_rate = _column_exists(conn, history_sql, GROWTH_RATE_COL)
    if have_growth_rate:
        query_cols.append(f'"{GROWTH_RATE_COL}" AS growth_rate')

    tf_columns: dict[str, tuple[str, list[str]]] = {}
    for tf_name, active_id, inactive_ids in TFS:
        active_alias = f"{tf_name.lower()}_active"
        active_expr = _bulk_column_expr(bulk_ids, active_id, active_alias)
        if active_expr is None:
            continue
        inactive_aliases = []
        skip_tf = False
        for i, inactive_id in enumerate(inactive_ids):
            inactive_alias = f"{tf_name.lower()}_inactive_{i}"
            inactive_expr = _bulk_column_expr(bulk_ids, inactive_id, inactive_alias)
            if inactive_expr is None:
                skip_tf = True
                break
            query_cols.append(inactive_expr)
            inactive_aliases.append(inactive_alias)
        if skip_tf:
            continue
        query_cols.append(active_expr)
        tf_columns[tf_name] = (active_alias, inactive_aliases)

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
        print("oxygen_transition: no rows returned; skipping.")
        return

    # Absolute time (not reset per generation), since the ramp is scheduled
    # in absolute simulation time and spans generations.
    raw = raw.with_columns((pl.col("time") / 60.0).alias("Time_min"))

    all_charts: list[alt.Chart] = []
    if have_o2_flux:
        all_charts.extend(_oxygen_flux_chart(raw, per_variant_params, subplot_width))
    if have_growth_rate:
        all_charts.extend(_growth_rate_chart(raw, per_variant_params, subplot_width))
    for tf_name, (active_alias, inactive_aliases) in tf_columns.items():
        all_charts.extend(
            _tf_fraction_charts(
                tf_name,
                raw,
                active_alias,
                inactive_aliases,
                per_variant_params,
                subplot_width,
            )
        )

    if not all_charts:
        print("oxygen_transition: no data to plot after aggregation; skipping.")
        return

    combined = alt.vconcat(*all_charts).properties(
        title="Aerobic-to-anaerobic oxygen transition"
    )

    out_path = os.path.join(outdir, "oxygen_transition.html")
    combined.save(out_path)
    print(f"Saved oxygen transition plot to {out_path}")
