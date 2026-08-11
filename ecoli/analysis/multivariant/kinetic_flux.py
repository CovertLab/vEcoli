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

Optionally, a ``plot_catalysts`` parameter can be used to highlight the
scatter points for the kinetic reaction(s) associated with one or more
catalysts (enzymes), labeling each point with its reaction ID and coloring
it by catalyst.

To reduce emit and process memory, data is binned by time (default is 1 min)
and averaged in SQL before being returned to Python for plotting
"""

from __future__ import annotations

import os
import pickle
from typing import Any, TYPE_CHECKING

import altair as alt
import numpy as np
import plotly.express as px
import polars as pl

from ecoli.analysis.multivariant.utils import compute_variant_grid, create_variant_label
from ecoli.library.parquet_emitter import (
    field_metadata,
    ndlist_to_ndarray,
    open_arbitrary_sim_data,
    read_stacked_columns,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

alt.data_transformers.enable("vegafusion")

PASTEL = px.colors.qualitative.Pastel
CATALYST_PALETTE = px.colors.qualitative.D3

# Tolerance added before log10 to handle zero fluxes
LOG_EPS = 1e-8
# Seconds per hour — converts mmol/(L·s) → mmol/(L·h)
S_PER_HR = 3600.0
FLUX_UNIT_STR = "mmol/(L·h)"
DEFAULT_TIME_BIN_MIN = 1.0


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

    Args:
        params: Dictionary of parameters given under analysis
            name in configuration JSON. Supports:

            .. code-block:: json

                {
                    // plot_catalysts: catalyst (enzyme) IDs whose associated
                    // kinetic reaction(s) should be highlighted/labeled on
                    // the scatter plot. Can be a list of string IDs...
                    "plot_catalysts": ["UDP-NACMURALA-GLU-LIG-MONOMER[c]"],

                    // ...or a dict of string IDs paired with human-readable
                    // labels for the legend.
                    "plot_catalysts": {
                        "UDP-NACMURALA-GLU-LIG-MONOMER[c]": "MurD"
                    }
                }
    """
    # ── Resolve per-variant parameter dicts ───────────────────────────────────
    experiment_id = next(iter(variant_metadata.keys()), None)
    per_variant_params: dict[int, Any] = (
        variant_metadata[experiment_id] if experiment_id else {}
    )
    time_bin_min = float(params.get("time_bin_min", DEFAULT_TIME_BIN_MIN))

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

    # ── Resolve requested catalysts to their kinetic reaction(s) ──────────────
    # Maps each kinetic reaction name to the human-readable label of the
    # catalyst that should highlight it (only populated for catalysts that
    # were requested and have at least one associated kinetic reaction).
    rxn_to_catalyst_label: dict[str, str] = {}
    plot_catalysts_param = params.get("plot_catalysts", [])
    if plot_catalysts_param:
        if isinstance(plot_catalysts_param, dict):
            catalyst_labels = dict(plot_catalysts_param)
        else:
            catalyst_labels = dict(zip(plot_catalysts_param, plot_catalysts_param))

        with open_arbitrary_sim_data(sim_data_dict) as f:
            sim_data = pickle.load(f)
        reaction_catalysts = sim_data.process.metabolism.reaction_catalysts

        # Every kinetic reaction is associated with exactly one catalyst:
        # reactions with kinetic constraints that originally had more than
        # one possible catalyst are split into one reaction per catalyst
        # upstream in sim_data (see `_replace_enzyme_reactions`).
        kinetic_catalyst_to_rxns: dict[str, list[str]] = {}
        for rxn in kinetic_rxn_names:
            catalysts_for_rxn = reaction_catalysts.get(rxn, [])
            if catalysts_for_rxn:
                kinetic_catalyst_to_rxns.setdefault(catalysts_for_rxn[0], []).append(
                    rxn
                )

        for catalyst, label in catalyst_labels.items():
            matched_rxns = kinetic_catalyst_to_rxns.get(catalyst, [])
            if not matched_rxns:
                print(
                    f"kinetic_flux_analysis: catalyst '{catalyst}' has no "
                    "associated kinetic reaction(s); skipping."
                )
                continue
            for rxn in matched_rxns:
                rxn_to_catalyst_label[rxn] = label

    # ── Scatter: per-(variant, reaction) time-averaged target/estimated flux ──
    # Averaged directly in SQL via UNNEST + GROUP BY (same pattern as
    # average_monomer_counts.py)
    kinetic_indices_1based = (kinetic_indices + 1).tolist()
    scatter_subquery = read_stacked_columns(
        history_sql,
        [
            "listeners__fba_results__target_kinetic_fluxes AS target_kinetic_fluxes",
            f"list_select(listeners__fba_results__estimated_fluxes, {kinetic_indices_1based}) AS estimated_fluxes",
            "listeners__enzyme_kinetics__counts_to_molar AS counts_to_molar",
        ],
        order_results=False,
        remove_first=True,
    )
    # unnest() on two same-length lists in one SELECT aligns them
    # positionally (like zip), not as a cross product
    scatter_avg = conn.sql(f"""
        WITH unnested AS (
            SELECT
                unnest(target_kinetic_fluxes) AS target_val,
                unnest(estimated_fluxes) AS estimated_val,
                generate_subscripts(target_kinetic_fluxes, 1) AS rxn_idx,
                counts_to_molar,
                variant
            FROM ({scatter_subquery})
        ),
        averaged AS (
            SELECT variant, rxn_idx,
                avg(target_val * counts_to_molar) AS mean_target,
                avg(estimated_val * counts_to_molar) AS mean_estimated
            FROM unnested
            GROUP BY variant, rxn_idx
        )
        SELECT variant,
            list(mean_target ORDER BY rxn_idx) AS mean_target_list,
            list(mean_estimated ORDER BY rxn_idx) AS mean_estimated_list
        FROM averaged
        GROUP BY variant
        ORDER BY variant
        """).pl()

    # ── Line: time-binned average kinetics term, per (variant, generation) ────
    # Binned and averaged directly in SQL (same "time_bin_min" resolution,
    # defaults to 1 minute, as metabolite_unmet_need.py)
    line_subquery = read_stacked_columns(
        history_sql,
        ["listeners__fba_results__kinetics_term AS kinetics_term"],
        order_results=False,
        remove_first=True,
    )
    line_avg = conn.sql(f"""
        WITH t_min AS (
            SELECT lineage_seed, min(time) AS t_min
            FROM ({line_subquery})
            GROUP BY lineage_seed
        )
        SELECT s.variant, s.generation,
            floor((s.time - t.t_min) / 60.0 / {time_bin_min}) * {time_bin_min}
                AS "Time (min)",
            avg(s.kinetics_term) AS kinetics_term
        FROM ({line_subquery}) s
        JOIN t_min t USING (lineage_seed)
        GROUP BY s.variant, s.generation,
            floor((s.time - t.t_min) / 60.0 / {time_bin_min}) * {time_bin_min}
        ORDER BY s.variant, s.generation, "Time (min)"
        """).pl()

    if scatter_avg.is_empty() or line_avg.is_empty():
        print("kinetic_flux_analysis: no rows returned; skipping.")
        return

    # Sort by variant explicitly
    scatter_avg = scatter_avg.sort("variant")

    # ── Variant label mapping ──────────────────────────────────────────────────
    unique_variants: list[int] = scatter_avg["variant"].to_list()

    def _make_label(v: int) -> str:
        raw_label = create_variant_label(v, per_variant_params)
        return " ".join(raw_label) if isinstance(raw_label, list) else raw_label

    variant_label_map = {v: _make_label(v) for v in unique_variants}
    variant_labels = [variant_label_map[v] for v in unique_variants]
    color_range = PASTEL[: len(unique_variants)]
    color_scale = alt.Scale(domain=variant_labels, range=color_range)

    # Row-major grid order (grouped by first/second sweep param, baseline
    # first) so facets don't fall back to Vega-Lite's default alphabetical
    # string sort of "Variant 1, Variant 11, ..., Variant 2, ..."
    _, grid_columns, ordered_variant_ids = compute_variant_grid(per_variant_params)
    variant_sort_order = [
        variant_label_map[v] for v in ordered_variant_ids if v in variant_label_map
    ]
    num_cols = grid_columns

    # ── Numpy arrays ──────────────────────────────────────────────────────────
    # One row per variant. counts_to_molar [mmol/L per count] was
    # already applied in SQL; multiply by S_PER_HR to get mmol/(L·h).
    mean_target_arr = ndlist_to_ndarray(scatter_avg["mean_target_list"]) * S_PER_HR
    mean_flux_arr = ndlist_to_ndarray(scatter_avg["mean_estimated_list"]) * S_PER_HR

    # ── Build scatter DataFrame ────────────────────────────────────────────────
    # Each data row is one (reaction, variant) pair averaged over all timesteps.
    # Two extra rows per variant encode the y=x reference line endpoints on the
    # log-transformed axes.
    scatter_rows: list[dict] = []
    for i, v in enumerate(unique_variants):
        label = variant_label_map[v]
        mean_target = mean_target_arr[i]  # mmol/(L·h)
        mean_flux = mean_flux_arr[i]

        log_target = np.log10(mean_target + LOG_EPS)
        log_flux = np.log10(mean_flux + LOG_EPS)

        # R² metrics computed on log-space values
        ss_res = np.sum((log_flux - log_target) ** 2)
        ss_tot = np.sum((log_target - log_target.mean()) ** 2)
        r2_to_yx = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        pearson_r2 = float(np.corrcoef(log_flux, log_target)[0, 1]) ** 2

        for i, rxn in enumerate(kinetic_rxn_names):
            catalyst_label = rxn_to_catalyst_label.get(rxn)
            scatter_rows.append(
                {
                    "Reaction": rxn,
                    "log_target": float(log_target[i]),
                    "log_flux": float(log_flux[i]),
                    "Variant": label,
                    "r2_to_yx": r2_to_yx,
                    "pearson_r2": pearson_r2,
                    "is_ref": False,
                    "Catalyst": catalyst_label,
                    "is_highlight": catalyst_label is not None,
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
                    "Catalyst": None,
                    "is_highlight": False,
                }
            )

    scatter_df = pl.DataFrame(scatter_rows, infer_schema_length=None).to_pandas()

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

    scatter_layers = [ref_line, scatter_pts, annotation]
    if rxn_to_catalyst_label:
        # Catalyst color scale is independent of the Variant scale used by
        # ref_line/scatter_pts/annotation, so it gets its own legend.
        catalyst_domain = list(dict.fromkeys(rxn_to_catalyst_label.values()))
        catalyst_color_scale = alt.Scale(
            domain=catalyst_domain,
            range=[
                CATALYST_PALETTE[i % len(CATALYST_PALETTE)]
                for i in range(len(catalyst_domain))
            ],
        )

        highlight_pts = (
            alt.Chart()
            .mark_point(
                size=160, filled=True, shape="diamond", strokeWidth=1.5, stroke="black"
            )
            .transform_filter("datum.is_highlight")
            .encode(
                x=alt.X("log_target:Q"),
                y=alt.Y("log_flux:Q"),
                color=alt.Color(
                    "Catalyst:N",
                    scale=catalyst_color_scale,
                    legend=alt.Legend(title="Catalyst"),
                ),
                tooltip=[
                    alt.Tooltip("Reaction:N"),
                    alt.Tooltip("Catalyst:N"),
                    alt.Tooltip("Variant:N"),
                    alt.Tooltip("log_target:Q", title="log₁₀(target)", format=".3f"),
                    alt.Tooltip("log_flux:Q", title="log₁₀(flux)", format=".3f"),
                ],
            )
        )

        highlight_labels = (
            alt.Chart()
            .mark_text(align="left", baseline="middle", dx=9, dy=-9, fontSize=9)
            .transform_filter("datum.is_highlight")
            .encode(
                x=alt.X("log_target:Q"),
                y=alt.Y("log_flux:Q"),
                text="Reaction:N",
                color=alt.Color("Catalyst:N", scale=catalyst_color_scale, legend=None),
            )
        )

        scatter_layers.extend([highlight_pts, highlight_labels])

    scatter_layered = alt.layer(*scatter_layers, data=scatter_df)
    if rxn_to_catalyst_label:
        scatter_layered = scatter_layered.resolve_scale(color="independent")

    scatter_faceted = (
        scatter_layered.properties(width=280, height=280)
        .facet(
            facet=alt.Facet("Variant:N", title="Variant", sort=variant_sort_order),
            columns=num_cols,
        )
        .resolve_scale(x="independent", y="independent")
        .properties(
            title=f"log₁₀(Avg Kinetic Target + ε) vs log₁₀(Avg Estimated Flux + ε)  [ε={LOG_EPS}]"
        )
    )

    # ── Line-plot DataFrame ────────────────────────────────────────────────────
    variant_label_col = [variant_label_map[v] for v in line_avg["variant"].to_list()]
    line_df = (
        line_avg.select(["Time (min)", "generation", "kinetics_term"])
        .with_columns(pl.Series("Variant", variant_label_col))
        .to_pandas()
    )

    line_faceted = (
        alt.Chart(line_df)
        .mark_line(strokeWidth=1.3, opacity=0.85)
        .encode(
            x=alt.X("Time (min):Q", title="Time (min)"),
            y=alt.Y("kinetics_term:Q", title="Unweighted Kinetic Term"),
            color=alt.Color("Variant:N", scale=color_scale, legend=None),
            detail=alt.Detail("generation:N"),
        )
        .properties(width=280, height=200)
        .facet(
            facet=alt.Facet("Variant:N", title="Variant", sort=variant_sort_order),
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
