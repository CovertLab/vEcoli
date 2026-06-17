"""
Kinetic flux analysis for multivariant metabolism_redux_classic simulations.

Two rows of faceted subplots, one column per variant:
  Top row:    Scatter — per-reaction log10(avg kinetic target + ε) vs
              log10(avg estimated kinetic flux + ε) in mmol/(L·h), with a
              dashed y=x reference line and Pearson R² / R²-to-y=x annotation.
  Bottom row: Line — weighted kinetic objective term over continuous simulation
              time, broken at cell division.

Variant panels are labeled with the fraction_kinetic_target value when
available from variant_metadata.
"""

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

import altair as alt
import numpy as np
import plotly.express as px
import polars as pl

from ecoli.analysis.multivariant import _variant_label
from ecoli.library.parquet_emitter import (
    field_metadata,
    ndlist_to_ndarray,
    read_stacked_columns,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

alt.data_transformers.enable("vegafusion")

PASTEL = px.colors.qualitative.Pastel

# Tolerance added before log10 to handle zero fluxes
LOG_EPS = 1e-8
# Seconds per hour — converts mmol/(L·s) → mmol/(L·h)
S_PER_HR = 3600.0
FLUX_UNIT_STR = "mmol/(L·h)"


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
    Faceted scatter (log avg kinetic target vs log avg kinetic flux) and
    kinetic term over time, one column per variant.
    """
    # ── Resolve per-variant parameter dicts ───────────────────────────────────
    experiment_id = next(iter(variant_metadata.keys()), None)
    per_variant_params: dict[int, Any] = (
        variant_metadata[experiment_id] if experiment_id else {}
    )

    # ── Metadata: reaction name lists ─────────────────────────────────────────
    kinetic_rxn_names: list[str] = field_metadata(
        conn, config_sql, "listeners__fba_results__target_kinetic_fluxes"
    )
    all_rxn_names: list[str] = field_metadata(
        conn, config_sql, "listeners__fba_results__solution_fluxes"
    )
    kinetic_indices = np.array(
        [all_rxn_names.index(name) for name in kinetic_rxn_names], dtype=int
    )

    # ── Load raw listener data ─────────────────────────────────────────────────
    raw = pl.DataFrame(
        read_stacked_columns(
            history_sql,
            [
                "listeners__fba_results__target_kinetic_fluxes AS target_kinetic_fluxes",
                "listeners__fba_results__estimated_fluxes AS estimated_fluxes",
                "listeners__fba_results__kinetics_term AS kinetics_term",
                "listeners__enzyme_kinetics__counts_to_molar AS counts_to_molar",
            ],
            order_results=True,
            conn=conn,
            remove_first=True,
        )
    )

    if raw.is_empty():
        print("kinetic_flux_analysis: no rows returned; skipping.")
        return

    # Continuous relative time per lineage_seed
    min_t = raw.group_by(["lineage_seed"]).agg(pl.col("time").min().alias("t_min"))
    raw = raw.join(min_t, on=["lineage_seed"])
    raw = raw.with_columns(
        ((pl.col("time") - pl.col("t_min")) / 60).alias("Time (min)")
    )

    # ── Variant label mapping ──────────────────────────────────────────────────
    unique_variants: list[int] = sorted(raw["variant"].unique().to_list())

    def _make_label(v: int) -> str:
        raw_label = _variant_label(v, per_variant_params)
        return " ".join(raw_label) if isinstance(raw_label, list) else raw_label

    variant_label_map = {v: _make_label(v) for v in unique_variants}
    variant_labels = [variant_label_map[v] for v in unique_variants]
    color_range = PASTEL[: len(unique_variants)]
    color_scale = alt.Scale(domain=variant_labels, range=color_range)

    # ── Numpy arrays ──────────────────────────────────────────────────────────
    target_arr = ndlist_to_ndarray(raw["target_kinetic_fluxes"])  # (T, n_kinetic)
    estimated_arr = ndlist_to_ndarray(raw["estimated_fluxes"])  # (T, n_all_rxns)
    kinetic_flux_arr = estimated_arr[:, kinetic_indices]  # (T, n_kinetic)
    # counts_to_molar [mmol/L per count]; multiply by S_PER_HR to get mmol/(L·h)
    counts_to_molar = raw["counts_to_molar"].to_numpy()[:, np.newaxis] * S_PER_HR
    variants_col = np.array(raw["variant"].to_list())

    # ── Build scatter DataFrame ────────────────────────────────────────────────
    # Each data row is one (reaction, variant) pair averaged over all timesteps.
    # Two extra rows per variant encode the y=x reference line endpoints on the
    # log-transformed axes.
    scatter_rows: list[dict] = []
    for v in unique_variants:
        label = variant_label_map[v]
        mask = variants_col == v
        ctm = counts_to_molar[mask]  # (T_v, 1)
        mean_target = (target_arr[mask] * ctm).mean(axis=0)  # mmol/(L·h)
        mean_flux = (kinetic_flux_arr[mask] * ctm).mean(axis=0)

        log_target = np.log10(mean_target + LOG_EPS)
        log_flux = np.log10(mean_flux + LOG_EPS)

        # R² metrics computed on log-space values
        ss_res = np.sum((log_flux - log_target) ** 2)
        ss_tot = np.sum((log_target - log_target.mean()) ** 2)
        r2_to_yx = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        pearson_r2 = float(np.corrcoef(log_flux, log_target)[0, 1]) ** 2

        for i, rxn in enumerate(kinetic_rxn_names):
            scatter_rows.append(
                {
                    "Reaction": rxn,
                    "log_target": float(log_target[i]),
                    "log_flux": float(log_flux[i]),
                    "Variant": label,
                    "r2_to_yx": r2_to_yx,
                    "pearson_r2": pearson_r2,
                    "is_ref": False,
                }
            )

        # y=x reference line endpoints in log space for this variant's scale
        ref_lo = float(min(log_target.min(), log_flux.min()))
        ref_hi = float(max(log_target.max(), log_flux.max()))
        for ref_val in (ref_lo, ref_hi):
            scatter_rows.append(
                {
                    "Reaction": f"_ref_{ref_val}",
                    "log_target": ref_val,
                    "log_flux": ref_val,
                    "Variant": label,
                    "r2_to_yx": r2_to_yx,
                    "pearson_r2": pearson_r2,
                    "is_ref": True,
                }
            )

    scatter_df = pl.DataFrame(scatter_rows).to_pandas()

    log_axis_title_x = f"log₁₀(Mean Kinetic Target + ε)  [{FLUX_UNIT_STR}]"
    log_axis_title_y = f"log₁₀(Mean Kinetic Flux + ε)  [{FLUX_UNIT_STR}]"

    # ── Scatter layer definitions ──────────────────────────────────────────────
    ref_line = (
        alt.Chart()
        .mark_line(color="lightgray", strokeDash=[5, 4], strokeWidth=1.2)
        .transform_filter("datum.is_ref")
        .encode(
            x=alt.X("log_target:Q"),
            y=alt.Y("log_flux:Q"),
        )
    )

    scatter_pts = (
        alt.Chart()
        .mark_circle(size=55, opacity=0.75)
        .transform_filter("!datum.is_ref")
        .encode(
            x=alt.X("log_target:Q", title=log_axis_title_x),
            y=alt.Y("log_flux:Q", title=log_axis_title_y),
            color=alt.Color("Variant:N", scale=color_scale, legend=None),
            tooltip=[
                alt.Tooltip("Reaction:N"),
                alt.Tooltip("Variant:N"),
                alt.Tooltip("log_target:Q", title="log₁₀(target)", format=".3f"),
                alt.Tooltip("log_flux:Q", title="log₁₀(flux)", format=".3f"),
            ],
        )
    )

    annotation = (
        alt.Chart()
        .mark_text(
            lineBreak="\n",
            align="left",
            baseline="top",
            dx=5,
            dy=5,
            fontSize=10,
        )
        .transform_filter("!datum.is_ref")
        .transform_aggregate(
            pearson_r2="mean(pearson_r2)",
            r2_to_yx="mean(r2_to_yx)",
            groupby=["Variant"],
        )
        .transform_calculate(
            annotation_label=(
                "'Pearson R\u00b2 = ' + format(datum.pearson_r2, '.2f') + '\\n'"
                "+ 'R\u00b2 to y=x = ' + format(datum.r2_to_yx, '.2f')"
            )
        )
        .encode(
            x=alt.value(5),
            y=alt.value(5),
            text="annotation_label:N",
        )
    )

    num_cols = min(len(unique_variants), 4)

    scatter_faceted = (
        alt.layer(ref_line, scatter_pts, annotation, data=scatter_df)
        .properties(width=280, height=280)
        .facet(
            facet=alt.Facet("Variant:N", title="Variant"),
            columns=num_cols,
        )
        .resolve_scale(x="independent", y="independent")
        .properties(
            title=f"log₁₀(Avg Kinetic Target + ε) vs log₁₀(Avg Estimated Flux + ε)  [ε={LOG_EPS}]"
        )
    )

    # ── Line-plot DataFrame ────────────────────────────────────────────────────
    variant_label_col = [variant_label_map[v] for v in raw["variant"].to_list()]
    line_df = (
        raw.select(["Time (min)", "generation", "lineage_seed", "kinetics_term"])
        .with_columns(pl.Series("Variant", variant_label_col))
        .to_pandas()
    )

    line_faceted = (
        alt.Chart(line_df)
        .mark_line(strokeWidth=1.3, opacity=0.85)
        .encode(
            x=alt.X("Time (min):Q", title="Time (min)"),
            y=alt.Y("mean(kinetics_term):Q", title="Unweighted Kinetic Term"),
            color=alt.Color("Variant:N", scale=color_scale, legend=None),
            detail=alt.Detail("generation:N"),
        )
        .properties(width=280, height=200)
        .facet(
            facet=alt.Facet("Variant:N", title="Variant"),
            columns=num_cols,
        )
        .resolve_scale(x="independent", y="independent")
        .properties(title="Kinetic Objective Term Over Time")
    )

    # ── Combine into faceted 2-row layout ─────────────────────────────────────
    final = (
        alt.vconcat(scatter_faceted, line_faceted)
        .resolve_scale(color="shared")
        .properties(title="Kinetic Flux Analysis by Variant")
    )

    out_path = os.path.join(outdir, "kinetic_flux_analysis.html")
    final.save(out_path)
    print(f"Saved kinetic flux analysis to: {out_path}")
