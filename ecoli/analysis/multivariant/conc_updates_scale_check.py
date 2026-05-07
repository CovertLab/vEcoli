"""
Regression check: verify that the per-step ``conc_updates`` listener emitted
by metabolism-redux is actually scaled by ``homeostatic_target_scale``.

Background: ``conc_updates`` is the per-step dict of biomass / ppGpp / AA
target concentrations that gets merged into ``homeostatic_objective`` on
every update. If the variant scale is not applied to this dict, the merge
silently overwrites the scaled targets initialized in ``first_update`` and
the variant becomes a no-op for all metabolites it touches (~32 metabolites
including the biomass building blocks). See the historical bug fixed in
``ecoli/processes/metabolism_redux.py`` ``update``.

This analysis takes the mean of ``conc_updates`` over generations >= 6 for
each variant, ratios it against the implicit baseline (variant 0), and
expects the median ratio to match the configured ``homeostatic_target_scale``.
"""

import os
import pickle
from typing import Any

import altair as alt

# noinspection PyUnresolvedReferences
from duckdb import DuckDBPyConnection
import numpy as np
import polars as pl

from ecoli.library.parquet_emitter import field_metadata, read_stacked_columns


LATE_GEN_THRESHOLD = 6
LISTENER_COL = "listeners__fba_results__conc_updates"
TOL = 1e-3


def _scale_from_sim_data(sim_data_path: str) -> float | None:
    try:
        with open(sim_data_path, "rb") as f:
            sim_data = pickle.load(f)
    except Exception:
        return None
    return float(getattr(sim_data, "homeostatic_target_scale", 1.0))


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
    os.makedirs(outdir, exist_ok=True)

    metabolite_names = field_metadata(conn, config_sql, LISTENER_COL)
    n_mets = len(metabolite_names)

    sub = read_stacked_columns(history_sql, [LISTENER_COL], order_results=False)
    df = conn.sql(f"""
        SELECT variant, {LISTENER_COL} AS u
        FROM ({sub})
        WHERE generation >= {LATE_GEN_THRESHOLD}
          AND len({LISTENER_COL}) = {n_mets}
    """).pl()

    if df.height == 0:
        print(
            f"[conc_updates_scale_check] No rows with generation >= "
            f"{LATE_GEN_THRESHOLD} and populated arrays; aborting."
        )
        return

    # Per-variant element-wise mean over late generations.
    rows: list[dict[str, Any]] = []
    for variant in sorted(df["variant"].unique().to_list()):
        arr = np.asarray(
            df.filter(pl.col("variant") == variant)["u"].to_list(), dtype=float
        )
        means = arr.mean(axis=0)
        for i, met in enumerate(metabolite_names):
            rows.append(
                {
                    "variant": int(variant),
                    "metabolite": met,
                    "mean_conc_update": float(means[i]),
                }
            )
    means_df = pl.DataFrame(rows)

    # Ratio vs variant 0 (the implicit baseline; should equal scale=1.0).
    baseline_idx = int(means_df["variant"].min())
    baseline = (
        means_df.filter(pl.col("variant") == baseline_idx)
        .select(["metabolite", "mean_conc_update"])
        .rename({"mean_conc_update": "baseline_mean"})
    )
    ratio_df = means_df.join(baseline, on="metabolite", how="left").with_columns(
        pl.when(pl.col("baseline_mean").abs() < 1e-30)
        .then(None)
        .otherwise(pl.col("mean_conc_update") / pl.col("baseline_mean"))
        .alias("ratio")
    )

    # Pull configured scales from sim_data pickles (authoritative).
    scale_lookup: dict[int, float | None] = {}
    for experiment_id, per_variant in sim_data_dict.items():
        for variant_idx, path in per_variant.items():
            scale_lookup[int(variant_idx)] = _scale_from_sim_data(path)

    summary = (
        ratio_df.filter(pl.col("ratio").is_not_null())
        .group_by("variant")
        .agg(
            [
                pl.col("ratio").median().alias("median_ratio"),
                pl.col("ratio").mean().alias("mean_ratio"),
                pl.col("ratio").std().alias("std_ratio"),
                pl.col("ratio").min().alias("min_ratio"),
                pl.col("ratio").max().alias("max_ratio"),
                pl.len().alias("n_metabolites"),
            ]
        )
        .sort("variant")
        .with_columns(
            pl.col("variant")
            .map_elements(lambda v: scale_lookup.get(int(v)), return_dtype=pl.Float64)
            .alias("configured_scale"),
        )
        .with_columns(
            (pl.col("median_ratio") - pl.col("configured_scale"))
            .abs()
            .alias("median_abs_error")
        )
    )

    means_csv = os.path.join(outdir, "conc_updates_means.csv")
    means_df.write_csv(means_csv)
    print(f"Wrote {means_csv}")

    summary_csv = os.path.join(outdir, "conc_updates_scale_check.csv")
    summary.write_csv(summary_csv)
    print(f"Wrote {summary_csv}")

    print("[conc_updates_scale_check] Per-variant ratios vs variant 0:")
    fail_rows: list[dict[str, Any]] = []
    for row in summary.iter_rows(named=True):
        cfg = row["configured_scale"]
        med = row["median_ratio"]
        ok = (
            cfg is not None
            and med is not None
            and abs(med - cfg) < TOL
            and abs(row["min_ratio"] - cfg) < TOL
            and abs(row["max_ratio"] - cfg) < TOL
        )
        marker = "OK" if ok else "FAIL"
        print(
            f"  variant={row['variant']:>2}  configured={cfg}  "
            f"median={med:.4f}  min={row['min_ratio']:.4f}  "
            f"max={row['max_ratio']:.4f}  {marker}"
        )
        if not ok and cfg is not None:
            fail_rows.append(row)

    if not fail_rows:
        print(
            "[conc_updates_scale_check] PASS — every variant's emitted "
            "conc_updates are uniformly scaled by the configured "
            "homeostatic_target_scale."
        )
    else:
        print(
            "[conc_updates_scale_check] FAIL — at least one variant's "
            "conc_updates do not match the configured scale (regression in "
            "metabolism_redux variant plumbing). See CSV for details."
        )

    plot_df = summary.filter(pl.col("configured_scale").is_not_null()).with_columns(
        pl.col("variant").cast(pl.Utf8).alias("variant_str")
    )
    if plot_df.height > 0:
        configured = (
            alt.Chart(plot_df)
            .mark_point(shape="circle", size=120, color="steelblue", filled=True)
            .encode(
                x=alt.X("variant_str:N", title="Variant"),
                y=alt.Y("configured_scale:Q", title="Scale"),
                tooltip=["variant", "configured_scale"],
            )
        )
        observed = (
            alt.Chart(plot_df)
            .mark_point(shape="diamond", size=120, color="orange", filled=True)
            .encode(
                x="variant_str:N",
                y="median_ratio:Q",
                tooltip=["variant", "median_ratio", "min_ratio", "max_ratio"],
            )
        )
        chart = (configured + observed).properties(
            title=(
                "Configured scale (blue) vs median conc_updates ratio (orange) "
                f"over generations >= {LATE_GEN_THRESHOLD}"
            )
        )
        chart.save(f"{outdir}/conc_updates_scale_check.html")
