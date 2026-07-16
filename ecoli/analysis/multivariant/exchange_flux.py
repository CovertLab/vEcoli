"""
Exchange flux (import/secretion) for all molecules exchanged with the
environment, for multivariant simulations run with the stock ``Metabolism``
process (``ecoli/processes/metabolism.py``).

Reads ``listeners.fba_results.external_exchange_fluxes`` (mmol/gDCW/hr;
negative = import, positive = secretion), indexed by
``self.model.fba.getExternalMoleculeIDs()`` -- the full set of molecules
wired into the FBA problem's exchange reactions, i.e.
``sim_data.external_state.all_external_exchange_molecules``.

Three figures are saved, each faceted/stacked per variant (top-N molecules
are ranked independently within each variant):

- A 2x1 timeseries per variant: top panel is bulk molecule counts, bottom
  panel is exchange flux, for either a user-specified list of molecule IDs
  or (default) that variant's own top-N molecules by mean |exchange flux|.
  A molecule missing from ``bulk`` is skipped in the top panel only; a
  molecule missing from the exchange molecule list is skipped in the bottom
  panel only -- the two panels' molecule sets can differ.
- A bar chart of each variant's top-N molecules by mean |exchange flux|.
- A molecule x time heatmap (all exchanged molecules, full timecourse), one
  row of subplots per variant, saved as a static PNG (matplotlib) rather
  than the Altair/HTML convention used elsewhere in this folder, since a
  full ~90-molecule x full-timecourse matrix renders far more reliably as a
  static raster image than as an interactive Vega chart. WATER[p]'s uptake
  flux is orders of magnitude larger than every other exchanged molecule,
  so it gets broken out into its own thin strip (own color scale) directly
  below each variant's main heatmap rather than washing out the shared
  color scale.

DISCLAIMER: This analysis currently only supports metabolism.py. See the
``is_redux`` param below.
"""

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING, cast

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from ecoli.analysis.multivariant.utils import create_variant_label
from ecoli.library.parquet_emitter import (
    field_metadata,
    named_idx,
    read_stacked_columns,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

alt.data_transformers.enable("vegafusion")

DEFAULT_TOP_N = 8
DEFAULT_SUBPLOT_WIDTH = 600
WATER_EXCHANGE_ID = "WATER[p]"
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
        params: Dictionary of parameters given under analysis name in
            configuration JSON. Config options look like this:

            .. code-block:: json

                {
                    // Not yet implemented -- see module docstring.
                    "is_redux": false,

                    // Exchange molecule IDs (with location tag, e.g.
                    // "GLC[p]") to plot in the timeseries figure. If not
                    // given, each variant's own top `top_n` molecules by
                    // mean |exchange flux| are used instead.
                    "metabolites_of_interest": null,

                    // Number of molecules to show in the top-N bar chart,
                    // and the timeseries fallback when
                    // `metabolites_of_interest` is not given.
                    "top_n": 8,

                    // Width of each per-variant subplot, in pixels.
                    "subplot_width": 600
                }
    """
    is_redux = params.get("is_redux", False)
    if is_redux:
        # metabolism_redux_classic's estimated_exchange_dmdt has a different
        # shape/unit convention (dict-per-timestep vs. a stable
        # name-indexed array) and its output format may still change; not
        # wired up yet.
        pass

    experiment_id = next(iter(variant_metadata.keys()), None)
    per_variant_params: dict[int, Any] = (
        variant_metadata[experiment_id] if experiment_id else {}
    )

    top_n = int(params.get("top_n", DEFAULT_TOP_N))
    metabolites_of_interest = params.get("metabolites_of_interest")
    subplot_width = int(params.get("subplot_width", DEFAULT_SUBPLOT_WIDTH))

    try:
        exchange_ids = field_metadata(
            conn, config_sql, "listeners__fba_results__external_exchange_fluxes"
        )
    except Exception:
        print(
            "exchange_flux: listeners__fba_results__external_exchange_fluxes "
            "not in config (e.g. non-metabolism sim); skipping."
        )
        return

    bulk_ids = field_metadata(conn, config_sql, "bulk")
    n_exch = len(exchange_ids)

    columns = [
        named_idx(
            "listeners__fba_results__external_exchange_fluxes",
            exchange_ids,
            [list(range(n_exch))],
        ),
    ]
    raw = pl.DataFrame(
        read_stacked_columns(
            history_sql,
            columns,
            conn=conn,
            order_results=True,
            success_sql=success_sql,
        )
    )

    if raw.is_empty():
        print("exchange_flux: no rows returned; skipping.")
        return

    id_vars = ["variant", "lineage_seed", "generation", "agent_id", "time"]
    flux_long = raw.select(id_vars + exchange_ids).melt(
        id_vars=id_vars,
        value_vars=exchange_ids,
        variable_name="molecule",
        value_name="flux",
    )

    # Relative time per (variant, generation, lineage_seed, agent_id),
    # matching the convention used elsewhere in this folder for
    # cross-cell comparability.
    t_min = raw.group_by(["variant", "generation", "lineage_seed", "agent_id"]).agg(
        pl.col("time").min().alias("t_min")
    )
    flux_long = flux_long.join(
        t_min, on=["variant", "generation", "lineage_seed", "agent_id"]
    )
    flux_long = flux_long.with_columns(
        ((pl.col("time") - pl.col("t_min")) / 60.0).alias("Time_min")
    )

    # Collapse to one value per (variant, Time_min, molecule), averaged
    # across cells -- reused by the timeseries, bar, and heatmap figures.
    agg = (
        flux_long.group_by(["variant", "Time_min", "molecule"])
        .agg(pl.col("flux").mean().alias("flux"))
        .sort("variant", "Time_min", "molecule")
    )

    variants = sorted(agg["variant"].unique().to_list())

    per_variant_top: dict[Any, pl.DataFrame] = {}
    per_variant_line: dict[Any, pl.DataFrame] = {}
    count_names_by_variant: dict[Any, list[str]] = {}
    bulk_names_needed: set[str] = set()

    for variant_val in variants:
        sub = agg.filter(pl.col("variant") == variant_val)
        if sub.is_empty():
            continue

        met_score = (
            sub.group_by("molecule")
            .agg(pl.col("flux").abs().mean().alias("mean_abs_flux"))
            .sort("mean_abs_flux", descending=True)
        )
        top_mets = met_score.head(top_n)["molecule"].to_list()
        per_variant_top[variant_val] = met_score.head(top_n)

        requested = (
            metabolites_of_interest if metabolites_of_interest is not None else top_mets
        )

        line_mets = [m for m in requested if m in exchange_ids]
        if not line_mets:
            line_mets = top_mets
        per_variant_line[variant_val] = sub.filter(pl.col("molecule").is_in(line_mets))

        count_mets = [m for m in requested if m in bulk_ids]
        count_names_by_variant[variant_val] = count_mets
        bulk_names_needed.update(count_mets)

    if not per_variant_top:
        print("exchange_flux: no per-variant data after aggregation; skipping.")
        return

    bulk_counts_long = _query_bulk_counts(
        bulk_names_needed, bulk_ids, history_sql, conn, success_sql
    )

    _plot_timeseries(
        per_variant_line,
        bulk_counts_long,
        variants,
        count_names_by_variant,
        per_variant_params,
        subplot_width,
        outdir,
    )
    _plot_top_n_bar(per_variant_top, per_variant_params, subplot_width, outdir)
    _plot_heatmap(agg, exchange_ids, variants, per_variant_params, outdir)


def _query_bulk_counts(
    bulk_names_needed: set[str],
    bulk_ids: list,
    history_sql: str,
    conn: "DuckDBPyConnection",
    success_sql: str,
) -> pl.DataFrame:
    """Bulk counts (mean over cells, relative time) for the union of
    molecule names needed across all variants' top panels."""
    if not bulk_names_needed:
        return pl.DataFrame()

    bulk_name_list = sorted(bulk_names_needed)
    bulk_idx = [bulk_ids.index(m) for m in bulk_name_list]
    bulk_columns = [named_idx("bulk", bulk_name_list, [bulk_idx])]

    bulk_raw = pl.DataFrame(
        read_stacked_columns(
            history_sql,
            bulk_columns,
            conn=conn,
            order_results=True,
            success_sql=success_sql,
        )
    )
    if bulk_raw.is_empty():
        return pl.DataFrame()

    bulk_id_vars = ["variant", "lineage_seed", "generation", "agent_id", "time"]
    bulk_long = bulk_raw.select(bulk_id_vars + bulk_name_list).melt(
        id_vars=bulk_id_vars,
        value_vars=bulk_name_list,
        variable_name="molecule",
        value_name="count",
    )
    t_min = bulk_raw.group_by(
        ["variant", "generation", "lineage_seed", "agent_id"]
    ).agg(pl.col("time").min().alias("t_min"))
    bulk_long = bulk_long.join(
        t_min, on=["variant", "generation", "lineage_seed", "agent_id"]
    )
    bulk_long = bulk_long.with_columns(
        ((pl.col("time") - pl.col("t_min")) / 60.0).alias("Time_min")
    )
    return (
        bulk_long.group_by(["variant", "Time_min", "molecule"])
        .agg(pl.col("count").mean().alias("count"))
        .sort("variant", "Time_min", "molecule")
    )


def _plot_timeseries(
    per_variant_line: dict[Any, pl.DataFrame],
    bulk_counts_long: pl.DataFrame,
    variants: list,
    count_names_by_variant: dict[Any, list[str]],
    per_variant_params: dict[int, Any],
    subplot_width: int,
    outdir: str,
) -> None:
    """One vconcat(counts, flux) block per variant, stacked vertically."""
    subplot_charts = []
    for variant_val in variants:
        line_df = per_variant_line.get(variant_val)
        if line_df is None or line_df.is_empty():
            continue
        label = create_variant_label(variant_val, per_variant_params)
        title = " ".join(label) if isinstance(label, list) else label

        count_names = count_names_by_variant.get(variant_val, [])
        if not bulk_counts_long.is_empty() and count_names:
            count_sub = bulk_counts_long.filter(
                (pl.col("variant") == variant_val)
                & (pl.col("molecule").is_in(count_names))
            )
        else:
            count_sub = pl.DataFrame(
                schema={
                    "Time_min": pl.Float64,
                    "molecule": pl.Utf8,
                    "count": pl.Float64,
                }
            )

        domain = sorted(set(count_names) | set(line_df["molecule"].unique().to_list()))
        color_range = [PASTEL[i % len(PASTEL)] for i in range(len(domain))]
        color_scale = alt.Scale(domain=domain, range=color_range)

        count_chart = (
            alt.Chart(count_sub.to_pandas())
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("Time_min:Q", title="Time (min)"),
                y=alt.Y("count:Q", title="Bulk molecule count"),
                color=alt.Color(
                    "molecule:N", scale=color_scale, legend=alt.Legend(title="Molecule")
                ),
                tooltip=["Time_min:Q", "molecule:N", "count:Q"],
            )
            .properties(height=220, width=subplot_width)
        )

        flux_chart = (
            alt.Chart(line_df.to_pandas())
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("Time_min:Q", title="Time (min)"),
                y=alt.Y("flux:Q", title="Exchange flux (mmol/gDCW/hr)"),
                color=alt.Color(
                    "molecule:N", scale=color_scale, legend=alt.Legend(title="Molecule")
                ),
                tooltip=["Time_min:Q", "molecule:N", "flux:Q"],
            )
            .properties(height=220, width=subplot_width)
        )

        subplot_charts.append(
            cast(
                alt.VConcatChart,
                alt.vconcat(count_chart, flux_chart, spacing=30).properties(
                    title=title
                ),
            )
        )

    if not subplot_charts:
        print("exchange_flux: no timeseries data to plot; skipping timeseries figure.")
        return

    combined = alt.vconcat(*subplot_charts).properties(
        title="Bulk counts vs. exchange flux by variant"
    )
    out_path = os.path.join(outdir, "exchange_flux_timeseries.html")
    combined.save(out_path)
    print(f"Saved exchange flux timeseries to {out_path}")


def _plot_top_n_bar(
    per_variant_top: dict[Any, pl.DataFrame],
    per_variant_params: dict[int, Any],
    subplot_width: int,
    outdir: str,
) -> None:
    """One bar-chart subplot per variant, stacked vertically."""
    subplot_charts = []
    for variant_val, top_df in per_variant_top.items():
        label = create_variant_label(variant_val, per_variant_params)
        title = " ".join(label) if isinstance(label, list) else label
        df_bar = top_df.to_pandas()

        bar_base = alt.Chart(df_bar).encode(
            x=alt.X("molecule:N", title="Molecule", sort="-y"),
            color=alt.Color("molecule:N", legend=None),
            tooltip=["molecule:N", "mean_abs_flux:Q"],
        )
        bars = bar_base.mark_bar(cornerRadiusEnd=8, size=28).encode(
            y=alt.Y(
                "mean_abs_flux:Q",
                title="Mean |exchange flux| (mmol/gDCW/hr)",
                scale=alt.Scale(type="symlog"),
            ),
        )
        bar_labels = bar_base.mark_text(
            align="center", baseline="bottom", dy=-4, fontSize=12, fontWeight="bold"
        ).encode(
            y=alt.Y("mean_abs_flux:Q", scale=alt.Scale(type="symlog")),
            text=alt.Text("mean_abs_flux:Q", format=".2e"),
        )
        chart = (bars + bar_labels).properties(
            height=260, width=subplot_width, title=title
        )
        subplot_charts.append(chart)

    if not subplot_charts:
        print("exchange_flux: no top-N data to plot; skipping bar chart.")
        return

    combined = alt.vconcat(*subplot_charts).properties(
        title="Top exchange fluxes by variant"
    )
    out_path = os.path.join(outdir, "exchange_flux_top_n_bar.html")
    combined.save(out_path)
    print(f"Saved exchange flux top-N bar chart to {out_path}")


def _imshow_panel(
    ax: Any,
    sub: Any,
    row_ids: list,
    cmap: str,
    vmax: float,
    show_xticklabels: bool,
) -> Any:
    """Pivot ``sub`` (variant-filtered long data) to a molecule x time
    matrix over ``row_ids`` and draw it on ``ax``. Returns the image."""
    mat = sub.pivot(index="molecule", columns="Time_min", values="flux")
    mat = mat.reindex(row_ids)

    im = ax.imshow(
        mat.to_numpy(),
        aspect="auto",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_yticks(range(len(row_ids)))
    ax.set_yticklabels(row_ids, fontsize=4 if len(row_ids) > 1 else 6)

    if mat.shape[1] > 0:
        n_ticks = min(6, mat.shape[1])
        tick_pos = np.linspace(0, mat.shape[1] - 1, n_ticks)
        if show_xticklabels:
            tick_labels = [mat.columns[int(p)] for p in tick_pos]
            ax.set_xticks(tick_pos)
            ax.set_xticklabels([f"{t:.0f}" for t in tick_labels])
        else:
            ax.set_xticks(tick_pos)
            ax.set_xticklabels([])
    return im


def _plot_heatmap(
    agg: pl.DataFrame,
    exchange_ids: list,
    variants: list,
    per_variant_params: dict[int, Any],
    outdir: str,
) -> None:
    """Molecule x time heatmap, one row of subplots per variant, saved as a
    static PNG. WATER[p]'s uptake flux dwarfs every other exchanged
    molecule, so if present it's broken out into its own thin strip (own
    color scale) directly below each variant's main heatmap rather than
    washing out the shared color scale."""
    df = agg.to_pandas()
    if df.empty:
        print("exchange_flux: no data for heatmap; skipping.")
        return

    water_id = WATER_EXCHANGE_ID if WATER_EXCHANGE_ID in exchange_ids else None
    main_ids = [m for m in exchange_ids if m != water_id]

    # Percentile rather than raw max: a single outlier cell among the
    # non-water molecules would otherwise dominate this scale the same way
    # WATER[p] used to, washing out everything else again one level down.
    main_df = df[df["molecule"] != water_id] if water_id else df
    main_flux_abs = np.abs(main_df["flux"].to_numpy())
    main_vmax = (
        float(np.nanpercentile(main_flux_abs, 99)) if main_flux_abs.size else 0.0
    )
    main_vmax = main_vmax if main_vmax > 0 else 1.0

    water_vmax = 1.0
    if water_id:
        water_df = df[df["molecule"] == water_id]
        if not water_df.empty:
            water_vmax = float(np.nanmax(np.abs(water_df["flux"].to_numpy())))
            water_vmax = water_vmax if water_vmax > 0 else 1.0

    print(
        f"exchange_flux: heatmap color scale -- main (excl. {water_id}): "
        f"+/-{main_vmax:.4g}; {water_id or 'water strip'}: +/-{water_vmax:.4g}"
    )

    n_variants = len(variants)
    main_height = max(4.0, len(main_ids) * 0.15)
    water_height = 0.7

    if water_id:
        height_ratios = [h for _ in variants for h in (main_height, water_height)]
        n_rows = n_variants * 2
    else:
        height_ratios = [main_height for _ in variants]
        n_rows = n_variants

    fig, axes_grid = plt.subplots(
        n_rows,
        1,
        figsize=(14, sum(height_ratios)),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.35},
        squeeze=False,
    )
    axes = axes_grid[:, 0]

    main_im = None
    water_im = None
    main_axes = []
    water_axes = []
    for i, variant_val in enumerate(variants):
        main_ax = axes[i * 2] if water_id else axes[i]
        sub = df[df["variant"] == variant_val]

        main_sub = sub[sub["molecule"] != water_id] if water_id else sub
        main_im = _imshow_panel(
            main_ax,
            main_sub,
            main_ids,
            "RdBu_r",
            main_vmax,
            show_xticklabels=not water_id,
        )
        main_axes.append(main_ax)

        label = create_variant_label(variant_val, per_variant_params)
        title = " ".join(label) if isinstance(label, list) else label
        main_ax.set_title(title, fontsize=10)
        if not water_id:
            main_ax.set_xlabel("Time (min)")

        if water_id:
            water_ax = axes[i * 2 + 1]
            water_sub = sub[sub["molecule"] == water_id]
            water_im = _imshow_panel(
                water_ax,
                water_sub,
                [water_id],
                "PuOr_r",
                water_vmax,
                show_xticklabels=True,
            )
            water_ax.set_xlabel("Time (min)")
            water_axes.append(water_ax)

    if main_im is not None:
        fig.colorbar(main_im, ax=main_axes, label="Exchange flux (mmol/gDCW/hr)")
    if water_im is not None:
        fig.colorbar(water_im, ax=water_axes, label="Exchange flux (mmol/gDCW/hr)")
    fig.suptitle("Exchange flux heatmap (all exchanged molecules)")

    out_path = os.path.join(outdir, "exchange_flux_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved exchange flux heatmap to {out_path}")
