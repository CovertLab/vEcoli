"""
Distribution of time-mean estimated fluxes across reactions for multivariant simulation.

One Altair bar-chart (histogram) subplot per variant, faceted in a grid.
Bins are shared across all variants so panels are directly comparable.
Each bin is displayed as a nominal category of equal visual width, which
handles unevenly-spaced (e.g. logarithmic) bin edges gracefully.
"""

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING, cast

import altair as alt
import numpy as np
import polars as pl

from ecoli.analysis.multivariant import _variant_label
from ecoli.library.parquet_emitter import (
    ndlist_to_ndarray,
    read_stacked_columns,
    field_metadata,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

alt.data_transformers.enable("vegafusion")

DEFAULT_FACET_COLUMNS = 2
# Pastel salmon — matches the rest of the multivariant palette
PASTEL_COLOR = "#FBB4AE"


def _fmt(x: float) -> str:
    """Compact number formatter for bin-edge labels."""
    if x == 0:
        return "0"
    if abs(x) >= 1e3 or abs(x) < 1e-2:
        return f"{x:.0E}"
    return f"{x:.2g}"


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
    """Histogram of per-reaction mean estimated flux; one facet per variant."""
    experiment_id = next(iter(variant_metadata.keys()), None)
    per_variant_params: dict[int, Any] = (
        variant_metadata[experiment_id] if experiment_id else {}
    )

    in_molar = params.get("in_molar", True)
    is_reduxclassic = params.get("is_reduxclassic", True)
    facet_columns = int(params.get("facet_columns", DEFAULT_FACET_COLUMNS))
    bin_edges_param = params.get("bin_edges")
    subplot_w = int(params.get("subplot_width", 400))
    subplot_h = int(params.get("subplot_height", 300))

    hist_subquery = cast(
        str,
        read_stacked_columns(
            history_sql,
            [
                "listeners__fba_results__estimated_fluxes AS estimated_fluxes",
                "listeners__enzyme_kinetics__counts_to_molar AS counts_to_molar",
            ],
            order_results=True,
            success_sql=success_sql,
            remove_first=is_reduxclassic,
        ),
    )
    flux_expr = (
        "unnest(estimated_fluxes) * counts_to_molar"
        if in_molar
        else "unnest(estimated_fluxes)"
    )

    raw = conn.sql(
        f"""
        WITH unnested AS (
            SELECT {flux_expr} AS flux,
                generate_subscripts(estimated_fluxes, 1) AS idx,
                experiment_id, variant, lineage_seed, generation, agent_id
            FROM ({hist_subquery})
        ),
        avg_flux AS (
            SELECT avg(flux) AS avg_flux,
                experiment_id, variant, lineage_seed, generation, agent_id, idx
            FROM unnested
            GROUP BY experiment_id, variant, lineage_seed, generation, agent_id, idx
        )
        SELECT list(avg_flux ORDER BY idx) AS estimated_fluxes,
            experiment_id, variant, lineage_seed, generation, agent_id
        FROM avg_flux
        GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
        """
    ).pl()

    if raw.is_empty():
        print("mean_flux_distribution: no rows returned; skipping.")
        return

    flux_matrix = ndlist_to_ndarray(raw["estimated_fluxes"])
    unit_label = "mmol/L/s" if in_molar else "counts/s"

    unique_variants = sorted(raw["variant"].unique().to_list())

    diversity_weight: float = field_metadata(
        conn, config_sql, "listeners__fba_results__diversity_term"
    )[0]
    # ── Bin edges (shared across all variants) ────────────────────────────────
    if bin_edges_param is not None:
        bin_edges = np.asarray(bin_edges_param, dtype=float)
    else:
        sim_flux_mean_global = flux_matrix.mean(axis=0)
        lo = float(np.min(sim_flux_mean_global))
        hi = float(np.max(sim_flux_mean_global))
        n_bins = int(params.get("n_bins", 30))
        bin_edges = np.linspace(lo, hi, n_bins + 1)

    # Nominal string labels — one per bin, in order.
    # Using nominal categories gives each bin equal visual width regardless of
    # how unevenly spaced the edges are (e.g. log-spaced custom edges).
    bin_labels = [
        f"[{_fmt(bin_edges[i])}, {_fmt(bin_edges[i + 1])})"
        for i in range(len(bin_edges) - 1)
    ]

    # ── Build tidy DataFrame: one row per (variant, bin) ──────────────────────
    variant_col = raw["variant"].to_numpy()
    rows: list[dict] = []
    for variant_val in unique_variants:
        mask = variant_col == variant_val
        sim_flux_mean = flux_matrix[mask, :].mean(axis=0)
        counts, _ = np.histogram(sim_flux_mean, bins=bin_edges)
        label_l = _variant_label(variant_val, per_variant_params)
        label = " ".join(label_l)
        for i, count in enumerate(counts):
            rows.append(
                {
                    "Variant": label,
                    "Bin": bin_labels[i],
                    "bin_order": i,
                    "Count": float(count),
                }
            )

    df_plot = pl.DataFrame(rows).to_pandas()

    x_enc = alt.X(
        "Bin:N",
        sort=bin_labels,
        title=f"Mean Estimated Flux ({unit_label})",
        axis=alt.Axis(labelAngle=-40, labelFontSize=10),
    )
    y_enc = alt.Y("Count:Q", title="Reaction count")
    tooltip_enc = [
        alt.Tooltip("Variant:N"),
        alt.Tooltip("Bin:N", title="Flux bin"),
        alt.Tooltip("Count:Q", title="Count"),
    ]

    bars = (
        alt.Chart(df_plot)
        .mark_bar(color=PASTEL_COLOR, stroke="white", strokeWidth=0.8)
        .encode(x=x_enc, y=y_enc, tooltip=tooltip_enc)
    )

    labels = (
        alt.Chart(df_plot)
        .mark_text(dy=-4, fontSize=10, fontWeight="bold")
        .encode(
            x=x_enc,
            y=y_enc,
            text=alt.Text("Count:Q", format=".0f"),
        )
        .transform_filter("datum.Count > 0")
    )

    chart = (
        alt.layer(bars, labels)
        .properties(width=subplot_w, height=subplot_h)
        .facet(
            facet=alt.Facet(
                "Variant:N",
                title=f"Variant \n Diversity weight = {diversity_weight:.2e}",
            ),
            columns=facet_columns,
        )
        .properties(title="Distribution of Mean Fluxes Across Reactions by Variant")
    )

    out_path = os.path.join(outdir, "mean_flux_distribution.html")
    chart.save(out_path)
    print(f"Saved multivariant mean flux distribution to {out_path}")
