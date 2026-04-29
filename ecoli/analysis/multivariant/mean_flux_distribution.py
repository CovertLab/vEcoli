"""
Distribution of time-mean estimated fluxes across reactions for multivariant simulation.

One Plotly histogram subplot per variant, labeled by variant name.
"""

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING, cast

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ecoli.library.parquet_emitter import (
    ndlist_to_ndarray,
    read_stacked_columns,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

DEFAULT_FACET_COLUMNS = 2
PASTEL = px.colors.qualitative.Pastel[0]


def _format_number(x: float) -> str:
    if x >= 1e3 or x < 1e-2:
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
    """Histogram of per-reaction mean estimated flux; one subplot per variant."""
    in_molar = params.get("in_molar", True)
    is_reduxclassic = params.get("is_reduxclassic", True)
    facet_columns = int(params.get("facet_columns", DEFAULT_FACET_COLUMNS))
    bin_edges_param = params.get("bin_edges")

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

    if bin_edges_param is not None:
        bin_edges = np.asarray(bin_edges_param, dtype=float)
    else:
        sim_flux_mean_global = flux_matrix.mean(axis=0)
        lo, hi = (
            float(np.min(sim_flux_mean_global)),
            float(np.max(sim_flux_mean_global)),
        )
        n_bins = int(params.get("n_bins", 30))
        bin_edges = np.ceil(np.linspace(lo, hi, n_bins + 1))

    bin_labels = [
        f"[{_format_number(bin_edges[i])}, {_format_number(bin_edges[i + 1])})"
        for i in range(len(bin_edges) - 1)
    ]

    traces: list[tuple[str, list[str], np.ndarray]] = []
    variant_col = raw["variant"].to_numpy()
    for variant_val in unique_variants:
        mask = variant_col == variant_val
        sim_flux_mean = flux_matrix[mask, :].mean(axis=0)
        binned_data, _ = np.histogram(sim_flux_mean, bins=bin_edges)
        label = variant_names.get(variant_val, f"Variant {variant_val}")
        traces.append((label, bin_labels, binned_data.astype(float)))

    title = "Distribution of Mean Fluxes Across Reactions by Variant"
    n = len(traces)
    cols = max(1, facet_columns) if n > 1 else 1
    rows = (n + cols - 1) // cols

    v_spacing = 0.08 if rows <= 1 else min(0.4, 0.08 * rows / (rows - 1))
    h_spacing = min(0.3, 0.10)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[t[0] for t in traces],
        vertical_spacing=v_spacing,
        horizontal_spacing=h_spacing,
    )

    for i, (_label, blabels, binned_data) in enumerate(traces):
        r = i // cols + 1
        c = i % cols + 1
        fig.add_trace(
            go.Bar(
                x=blabels,
                y=binned_data,
                marker=dict(color=PASTEL, line=dict(color="white", width=1)),
                text=binned_data,
                textposition="auto",
                showlegend=False,
            ),
            row=r,
            col=c,
        )
        fig.update_xaxes(
            title_text=f"Mean Estimated Flux ({unit_label})",
            tickangle=-40,
            tickfont=dict(size=10),
            row=r,
            col=c,
        )
        fig.update_yaxes(title_text="Count", row=r, col=c)

    subplot_w = int(params.get("subplot_width", 480))
    subplot_h = int(params.get("subplot_height", 380))
    fig_w = int(params.get("figure_width", subplot_w * cols))
    fig_h = int(params.get("figure_height", subplot_h * rows + 80))

    annotation_font_size = max(10, 14 - cols)
    for ann in fig.layout.annotations:
        ann.font.size = annotation_font_size

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        title=dict(text=title, font=dict(size=20)),
        width=fig_w,
        height=fig_h,
        margin=dict(t=80 + annotation_font_size * 2, b=60, l=60, r=40),
    )

    out_path = os.path.join(outdir, "mean_flux_distribution.html")
    fig.write_html(out_path, include_plotlyjs="cdn", config={"displayModeBar": True})
    print(f"Saved multivariant mean flux distribution to {out_path}")
