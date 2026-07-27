"""
Log-uniform Pareto Front Exploration
=====================================
Samples weight combinations within feasible ranges (from pairwise analysis)
on a log scale, solves the NetworkFlowModel for each combination, and
visualizes the resulting Pareto front across all objective terms.

Produces three plots:
    pareto_results/pairwise_homeostatic.html     — Altair pairwise scatter + table + weight distribution
    pareto_results/parallel_coordinates.html     — Altair parallel coordinates
    pareto_results/pareto_3d.html                — Plotly 3D interactive

Usage:
    python pareto_exploration.py
    python pareto_exploration.py --n_samples 500 --n_jobs 4
"""

from ecoli.processes.metabolism_redux_classic import (
    FlowResult,
    FREE_RXNS,
    MetabolismReduxClassic,
    NetworkFlowModel,
)
from ecoli.library.parquet_emitter import (
    dataset_sql,
    field_metadata,
    ndlist_to_ndarray,
    read_stacked_columns,
)
from ecoli.library.sim_data import LoadSimData
from wholecell.utils import units, toya
import argparse
import glob
import os
import warnings
from typing import Optional
import pickle
import json
import duckdb
from fsspec import open as fsspec_open
import altair as alt
import cvxpy as cp
import numpy as np
import plotly.graph_objects as go
import polars as pl
from altair import datum
from joblib import Parallel, delayed
from tqdm import tqdm

os.chdir(os.path.expanduser("~/dev/vEcoli/"))

# --- Import and Define Units ---
COUNTS_UNITS = units.mmol
VOLUME_UNITS = units.L
MASS_UNITS = units.g
TIME_UNITS = units.s
CONC_UNITS = COUNTS_UNITS / VOLUME_UNITS
FLUX_UNITS = COUNTS_UNITS / VOLUME_UNITS / TIME_UNITS

# ---------------------------------------------------------------------------
# Feasible weight ranges (from pairwise analysis). Log-spaced sampling.
# Homeostatic weight is always fixed at 1.
# ---------------------------------------------------------------------------
WEIGHT_RANGES = {
    "homeostatic": (1e-3, 1.0),
    "secretion": (1e-7, 1e-4),  # 2.12E-4
    "efficiency": (1e-7, 1e-4),  # 2.34E-5
    # Widened upper bound from 1e-3 to 1e-2: the previous range under-sampled
    # the large lambda_kin/lambda_hom ratios needed for a real (non-noise-floor)
    # trade-off between obj_kin and obj_homeo (see 20260723 re-analysis).
    "kinetics": (1e-5, 1e-2),
    "diversity": (1e-5, 1e-2),  # 8.53E-3
}

OUT_DIR = (
    "notebooks/Heena notebooks/Metabolism_New Genes/pareto_results_jul_10000samples"
)
os.makedirs(OUT_DIR, exist_ok=True)

with open(f"{OUT_DIR}/weight_info.json", "w") as fp:
    json.dump(WEIGHT_RANGES, fp)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def log_uniform_sample(n_samples: int, seed: int = 42) -> np.ndarray:
    """
    Draw n_samples weight combinations uniformly in log space within each
    term's feasible range.

    Returns array of shape (n_samples, 5) with columns ordered as
    WEIGHT_RANGES: [homeostatic, secretion, efficiency, kinetics, diversity].
    """
    rng = np.random.default_rng(seed)
    samples = []
    for lo, hi in WEIGHT_RANGES.values():
        log_samples = rng.uniform(np.log10(lo), np.log10(hi), size=n_samples)
        samples.append(10**log_samples)
    return np.column_stack(samples)  # (n_samples, 4)


# ---------------------------------------------------------------------------
# Get R-square of central metabolism reactions against Toya
# ---------------------------------------------------------------------------
def correlations_toya_fluxes(
    reaction_ids: str, sim_reaction_flux: np.ndarray
) -> tuple[float, float, np.ndarray]:
    # Load validation data (Toya 2010 fluxes and stdevs)
    validation_data_path = "out/kb/validationData.cPickle"
    with fsspec_open(validation_data_path, "rb") as f:
        validation_data = pickle.load(f)

    cell_mass = units.fg * 1745.814482240506  # mean cell mass from a sim
    dry_mass = units.fg * 524.0582963771143  # mean dry mass from a sim
    cell_density = units.g / units.L * 1100  # constant

    toya_reactions = validation_data.reactionFlux.toya2010fluxes["reactionID"]
    toya_fluxes = toya.adjust_toya_data(
        validation_data.reactionFlux.toya2010fluxes["reactionFlux"],
        cell_mass,
        dry_mass,
        cell_density,
    )  # outputs in mmol/L/s

    # Align simulated and Toya fluxes to matching reaction IDs
    # Comment: we are using a single time point FBA, so we can't get stdevs out
    #          to still use the process_simulated_fluxes function, we will pass
    #          a dummy time course array of (2, n_reactions) where the two time
    #          points are identical (the single FBA solution)
    sim_reaction_fluxes = FLUX_UNITS * np.vstack(
        [sim_reaction_flux, sim_reaction_flux]
    )  # (2, n_reactions)
    sim_flux_means, sim_flux_stdevs = toya.process_simulated_fluxes(
        toya_reactions, reaction_ids, sim_reaction_fluxes
    )

    toya_flux_means = toya.process_toya_data(
        toya_reactions, toya_reactions, toya_fluxes
    )

    sim_means_num = sim_flux_means.asNumber(FLUX_UNITS)
    toya_means_num = toya_flux_means.asNumber(FLUX_UNITS)

    pearson_r_squared = float(np.corrcoef(sim_means_num, toya_means_num)[0, 1]) ** 2
    ss_res = np.sum((sim_means_num - toya_means_num) ** 2)
    ss_tot = np.sum((toya_means_num - np.mean(toya_means_num)) ** 2)
    r_squared = float(1 - ss_res / ss_tot)

    return pearson_r_squared, r_squared, toya_fluxes


# ---------------------------------------------------------------------------
# Single solve — wraps existing NetworkFlowModel
# ---------------------------------------------------------------------------
def solve_one(
    lam_hom: float,
    lam_sec: float,
    lam_eff: float,
    lam_kin: float,
    lam_div: float,
    # ----- fixed problem data -----
    stoichiometry: np.ndarray,
    metabolites: list,
    reaction_names: list,
    metabolism,  # MetabolismReduxClassic used for this exploration
    homeostatic_metabolite_counts,
    homeostatic_dm_targets,
    kinetic,
    maintenance,
    counts_to_molar,
    solver_choice=cp.GLOP,
    binary_kinetics_idx=None,
    fraction_kinetic_target: float = 1.0,
) -> Optional[dict]:
    """
    Build and solve the NetworkFlowModel for one weight combination.
    Returns a flat dict of weights + objective term values, or None on failure.

    `fraction_kinetic_target` scales all kinetic targets uniformly (the
    global proxy for a kinetic-enzyme knockdown used throughout this
    project — see NetworkFlowModel.solve()); defaults to 1.0 (no knockdown).
    """
    weights = {
        "homeostatic": lam_hom,
        "secretion": lam_sec,
        "efficiency": lam_eff,
        "kinetics": lam_kin,
        "diversity": lam_div,
    }

    try:
        model = NetworkFlowModel(
            stoich_arr=stoichiometry,
            metabolites=metabolites,
            reactions=reaction_names,
            homeostatic_metabolites=metabolism.homeostatic_metabolites,
            kinetic_reactions=metabolism.kinetic_constraint_reactions,
            free_reactions=FREE_RXNS,
        )
        model.set_up_exchanges(
            exchanges=metabolism.exchange_molecules,
            uptakes=metabolism.allowed_exchange_uptake,
        )
        solution: FlowResult = model.solve(
            homeostatic_concs=homeostatic_metabolite_counts,
            homeostatic_dm_targets=np.array(
                list(dict(homeostatic_dm_targets).values())
            ),
            maintenance_target=maintenance,
            kinetic_targets=np.array(list(dict(kinetic).values())),
            binary_kinetic_idx=binary_kinetics_idx,
            objective_weights=weights,
            upper_flux_bound=100,
            target_minimal_flux=counts_to_molar[-1],
            fraction_kinetic_target=fraction_kinetic_target,
            include_new=True,
            new_reaction_idx=metabolism.new_reaction_idx,
            solver=solver_choice,
        )

        return {
            "lambda_hom": lam_hom,
            "lambda_sec": lam_sec,
            "lambda_eff": lam_eff,
            "lambda_kin": lam_kin,
            "lambda_div": lam_div,
            "fraction_kinetic_target": fraction_kinetic_target,
            "obj_total": solution.objective,
            "obj_homeo": solution.homeostatic_term,
            "obj_kin": solution.kinetics_term,
            "obj_eff": solution.efficiency_term,
            "obj_sec": solution.secretion_term,
            "obj_div": solution.diversity_term,
            # Weighted contributions (w_i * T_i) — what the optimizer actually sees
            "obj_hom_w": lam_hom * solution.homeostatic_term,
            "obj_kin_w": lam_kin * solution.kinetics_term,
            "obj_eff_w": lam_eff * solution.efficiency_term,
            "obj_sec_w": lam_sec * solution.secretion_term,
            "obj_div_w": lam_div * solution.diversity_term,
            "solution_flux": solution.velocities,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_pairwise_altair(df: pl.DataFrame) -> None:
    """
    2x2 grid: homeostatic objective vs each of the four secondary terms.
    Points are coloured by the corresponding lambda for that term.
    """
    terms = [
        ("obj_sec", "lambda_sec", "Secretion"),
        ("obj_eff", "lambda_eff", "Efficiency"),
        ("obj_kin", "lambda_kin", "Kinetic"),
        ("obj_div", "lambda_div", "Diversity"),
    ]

    # --- Make Scatter Plots ---
    charts = []
    interval = alt.selection_interval()
    for obj_col, lam_col, title in terms:
        chart = (
            alt.Chart(df)
            .mark_circle(size=40, opacity=0.6)
            .transform_filter(datum.toya_r_squared > 0)  # filter out negative R²
            .encode(
                y=alt.Y("obj_homeo:Q", title="Homeostatic Objective"),
                x=alt.X(f"{obj_col}:Q", title=f"{title} Objective"),
                color=alt.condition(
                    interval,
                    alt.Color(
                        "toya_r_squared:Q",
                        scale=alt.Scale(scheme="viridis", domain=[0, 1.0]),
                        legend=alt.Legend(
                            title="R² (Coefficient of Determination)",
                            values=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        ),
                    ),
                    alt.value("lightgray"),
                ),
                tooltip=[
                    alt.Tooltip("obj_homeo:Q", format=".4e"),
                    alt.Tooltip(f"{obj_col}:Q", format=".4e"),
                    alt.Tooltip(
                        f"{lam_col}:Q", format=".2e", title=f"λ_{title[:3].lower()}"
                    ),
                    alt.Tooltip("toya_r_squared:Q", format=".3f", title="R² (COD)"),
                ],
            )
            .properties(title=f"Homeostatic vs {title}", width=280, height=250)
        ).add_selection(interval)
        charts.append(chart)

    combined_scatter = (
        ((charts[0] | charts[1]) & (charts[2] | charts[3]))
        .properties(title="Pairwise Pareto: Homeostatic vs Secondary Objectives")
        .resolve_scale(color="shared")
    )  # shared color scale across all charts

    # --- Make Table Alongside Selection ---
    # --- Rank by fit to Toya fluxes and largest lambda_norm ---
    # Base chart for data tables
    ranked_text = (
        alt.Chart(df)
        .mark_text(align="right")
        .encode(y=alt.Y("rank:O", axis=None))
        .transform_filter(interval)
        .transform_calculate(
            lambda_norm="sqrt(datum.lambda_sec * datum.lambda_sec + "
            "datum.lambda_eff * datum.lambda_eff + "
            "datum.lambda_kin * datum.lambda_kin + "
            "datum.lambda_div * datum.lambda_div)"
        )
        .transform_window(
            rank="rank()",
            sort=[
                alt.SortField("toya_r_squared", order="descending"),
                alt.SortField("toya_pearson_r_squared", order="descending"),
                alt.SortField("lambda_norm", order="descending"),
            ],
        )
        .transform_filter(alt.datum.rank <= 10)
        .properties(height=240)
    )

    # Select Columns to Display for Data Tables
    lambda_sec = ranked_text.encode(
        text=alt.Text("lambda_sec:Q", format=".2e")
    ).properties(title=alt.Title(text="λ_sec", align="right"))
    lambda_eff = ranked_text.encode(
        text=alt.Text("lambda_eff:Q", format=".2e")
    ).properties(title=alt.Title(text="λ_eff", align="right"))
    lambda_kin = ranked_text.encode(
        text=alt.Text("lambda_kin:Q", format=".2e")
    ).properties(title=alt.Title(text="λ_kin", align="right"))
    lambda_div = ranked_text.encode(
        text=alt.Text("lambda_div:Q", format=".2e")
    ).properties(title=alt.Title(text="λ_div", align="right"))
    # obj_homeo = ranked_text.encode(
    #     text=alt.Text("obj_homeo:Q", format=".3e")
    # ).properties(title=alt.Title(text="Homeostatic Obj.", align="right"))
    obj_kinetic = ranked_text.encode(
        text=alt.Text("obj_kin:Q", format=".3f")
    ).properties(title=alt.Title(text="Unweighted Kinetic Obj.", align="right"))
    # obj_secretion = ranked_text.encode(
    #     text=alt.Text("obj_sec:Q", format=".3f")
    # ).properties(title=alt.Title(text="Unweighted Secretion Obj.", align="right"))
    toya_r2 = ranked_text.encode(
        text=alt.Text("toya_r_squared:Q", format=".3f")
    ).properties(title=alt.Title(text="R² (COD)", align="right"))
    toya_pearson_r2 = ranked_text.encode(
        text=alt.Text("toya_pearson_r_squared:Q", format=".3f")
    ).properties(title=alt.Title(text="Pearson R²", align="right"))

    # Combine Columns to Display
    text = alt.hconcat(
        lambda_sec,
        lambda_eff,
        lambda_kin,
        lambda_div,
        # obj_homeo,
        obj_kinetic,
        # obj_secretion,
        toya_r2,
        toya_pearson_r2,
    )

    # --- Plot Lambda Distributions for Selected Points ---
    density = (
        alt.Chart(df)
        .transform_filter(interval)
        .transform_fold(
            ["lambda_sec", "lambda_eff", "lambda_kin", "lambda_div"],
            as_=["lambda_type", "value"],
        )
        .transform_calculate(log_value="log(datum.value) / log(10)")
        .mark_bar(opacity=0.4, binSpacing=0)
        .encode(
            x=alt.X("log_value:Q", title="log₁₀(Lambda Value)").bin(
                maxbins=40, base=10
            ),
            y=alt.Y("count()", title="Count", stack=False),
            color=alt.Color(
                "lambda_type:N",
                title="Lambda",
                legend=alt.Legend(orient="none", legendX=550, legendY=300),
            ),
        )
        .properties(title="Distribution of Objected Weights", width=500, height=300)
    )

    # Build chart
    c2 = text & density
    combined = (
        (combined_scatter | c2)
        .configure_title(fontSize=14, anchor="middle")
        .configure_view(stroke=None)
    )

    out = os.path.join(OUT_DIR, "pairwise_analysis.html")
    combined.save(out)
    print(f"  Saved: {out}")


def plot_pairwise_altair_weighted(df: pl.DataFrame) -> None:
    """
    Same layout as plot_pairwise_altair but uses weighted objective contributions
    (w_i * T_i) so axes reflect what the optimizer actually balances.
    Saved to pairwise_analysis_weighted.html.
    """
    terms = [
        ("obj_sec_w", "lambda_sec", "Secretion"),
        ("obj_eff_w", "lambda_eff", "Efficiency"),
        ("obj_kin_w", "lambda_kin", "Kinetic"),
        ("obj_div_w", "lambda_div", "Diversity"),
    ]

    charts = []
    interval = alt.selection_interval()
    for obj_col, lam_col, title in terms:
        chart = (
            alt.Chart(df)
            .mark_circle(size=40, opacity=0.6)
            .transform_filter(datum.toya_r_squared > 0)  # filter out negative R²
            .encode(
                y=alt.Y("obj_hom_w:Q", title="Weighted Homeostatic (w·T)"),
                x=alt.X(f"{obj_col}:Q", title=f"Weighted {title} (w·T)"),
                color=alt.condition(
                    interval,
                    alt.Color(
                        "toya_r_squared:Q",
                        scale=alt.Scale(scheme="viridis", domain=[0, 1.0]),
                        legend=alt.Legend(
                            title="R² (Coefficient of Determination)",
                            values=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        ),
                    ),
                    alt.value("lightgray"),
                ),
                tooltip=[
                    alt.Tooltip(
                        "obj_hom_w:Q", format=".4e", title="Weighted Homeostatic"
                    ),
                    alt.Tooltip(
                        f"{obj_col}:Q", format=".4e", title=f"Weighted {title}"
                    ),
                    alt.Tooltip(
                        f"{lam_col}:Q", format=".2e", title=f"λ_{title[:3].lower()}"
                    ),
                    alt.Tooltip("toya_r_squared:Q", format=".3f", title="R² (COD)"),
                ],
            )
            .properties(
                title=f"Weighted Homeostatic vs Weighted {title}", width=280, height=250
            )
        ).add_selection(interval)
        charts.append(chart)

    combined_scatter = (
        ((charts[0] | charts[1]) & (charts[2] | charts[3]))
        .properties(
            title="Pairwise Pareto: Weighted Homeostatic vs Weighted Secondary Objectives"
        )
        .resolve_scale(color="shared")
    )

    ranked_text = (
        alt.Chart(df)
        .mark_text(align="right")
        .encode(y=alt.Y("rank:O", axis=None))
        .transform_filter(interval)
        .transform_calculate(
            lambda_norm="sqrt(datum.lambda_sec * datum.lambda_sec + "
            "datum.lambda_eff * datum.lambda_eff + "
            "datum.lambda_kin * datum.lambda_kin + "
            "datum.lambda_div * datum.lambda_div)"
        )
        .transform_window(
            rank="rank()",
            sort=[
                alt.SortField("toya_r_squared", order="descending"),
                alt.SortField("toya_pearson_r_squared", order="descending"),
                alt.SortField("lambda_norm", order="descending"),
            ],
        )
        .transform_filter(alt.datum.rank <= 10)
        .properties(height=240)
    )

    lambda_sec = ranked_text.encode(
        text=alt.Text("lambda_sec:Q", format=".2e")
    ).properties(title=alt.Title(text="λ_sec", align="right"))
    lambda_eff = ranked_text.encode(
        text=alt.Text("lambda_eff:Q", format=".2e")
    ).properties(title=alt.Title(text="λ_eff", align="right"))
    lambda_kin = ranked_text.encode(
        text=alt.Text("lambda_kin:Q", format=".2e")
    ).properties(title=alt.Title(text="λ_kin", align="right"))
    lambda_div = ranked_text.encode(
        text=alt.Text("lambda_div:Q", format=".2e")
    ).properties(title=alt.Title(text="λ_div", align="right"))
    obj_kinetic_w = ranked_text.encode(
        text=alt.Text("obj_kin_w:Q", format=".3e")
    ).properties(title=alt.Title(text="Weighted Kinetic (w·T)", align="right"))
    toya_r2 = ranked_text.encode(
        text=alt.Text("toya_r_squared:Q", format=".3f")
    ).properties(title=alt.Title(text="R² (COD)", align="right"))
    toya_pearson_r2 = ranked_text.encode(
        text=alt.Text("toya_pearson_r_squared:Q", format=".3f")
    ).properties(title=alt.Title(text="Pearson R²", align="right"))

    text = alt.hconcat(
        lambda_sec,
        lambda_eff,
        lambda_kin,
        lambda_div,
        obj_kinetic_w,
        toya_r2,
        toya_pearson_r2,
    )

    density = (
        alt.Chart(df)
        .transform_filter(interval)
        .transform_fold(
            ["lambda_sec", "lambda_eff", "lambda_kin", "lambda_div"],
            as_=["lambda_type", "value"],
        )
        .transform_calculate(log_value="log(datum.value) / log(10)")
        .mark_bar(opacity=0.4, binSpacing=0)
        .encode(
            x=alt.X("log_value:Q", title="log₁₀(Lambda Value)").bin(
                maxbins=40, base=10
            ),
            y=alt.Y("count()", title="Count", stack=False),
            color=alt.Color(
                "lambda_type:N",
                title="Lambda",
                legend=alt.Legend(orient="none", legendX=550, legendY=300),
            ),
        )
        .properties(title="Distribution of Objective Weights", width=500, height=300)
    )

    c2 = text & density
    combined = (
        (combined_scatter | c2)
        .configure_title(fontSize=14, anchor="middle")
        .configure_view(stroke=None)
    )

    out = os.path.join(OUT_DIR, "pairwise_analysis_weighted.html")
    combined.save(out)
    print(f"  Saved: {out}")


def plot_parallel_coordinates_altair(df: pl.DataFrame) -> None:
    """
    Parallel coordinates across all 5 objective terms, normalised to [0, 1].
    Lines are coloured by homeostatic objective value so you can spot which
    weight combinations keep homeostasis low while varying the rest.
    """
    obj_cols = ["obj_homeo", "obj_sec", "obj_eff", "obj_kin", "obj_div"]
    axis_labels = {
        "obj_homeo": "Homeostatic",
        "obj_sec": "Secretion",
        "obj_eff": "Efficiency",
        "obj_kin": "Kinetic",
        "obj_div": "Diversity",
    }

    norm_data = {}
    for col in obj_cols:
        vals = df[col].to_numpy()
        lo, hi = vals.min(), vals.max()
        norm_data[col] = (vals - lo) / (hi - lo + 1e-30)

    norm_df = pl.DataFrame(
        {**norm_data, "obj_homeo_raw": df["obj_homeo"]}
    ).with_row_index("sample_id")
    melted = norm_df.melt(
        id_vars=["sample_id", "obj_homeo_raw"],
        value_vars=obj_cols,
        variable_name="objective",
        value_name="normalized_value",
    ).with_columns(pl.col("objective").replace(axis_labels).alias("objective_label"))

    chart = (
        alt.Chart(melted)
        .mark_line(opacity=0.3)
        .encode(
            x=alt.X(
                "objective_label:N",
                sort=list(axis_labels.values()),
                title="Objective",
                axis=alt.Axis(labelAngle=-20),
            ),
            y=alt.Y("normalized_value:Q", title="Normalised Value [0–1]"),
            color=alt.Color(
                "obj_homeo_raw:Q",
                scale=alt.Scale(scheme="plasma"),
                title="Homeostatic Value",
            ),
            detail="sample_id:N",
            tooltip=[
                alt.Tooltip("objective_label:N", title="Objective"),
                alt.Tooltip("normalized_value:Q", format=".3f", title="Normalised"),
                alt.Tooltip("obj_homeo_raw:Q", title="Homeostatic (raw)", format=".4e"),
            ],
        )
        .properties(
            title="Parallel Coordinates: All Objectives (normalised)",
            width=600,
            height=350,
        )
        .configure_title(fontSize=14)
    )
    out = os.path.join(OUT_DIR, "parallel_coordinates.html")
    chart.save(out)
    print(f"  Saved: {out}")


def plot_3d_plotly(df: pl.DataFrame) -> None:
    """
    3D interactive scatter: Kinetic (x) vs Diversity (y) vs Homeostatic (z).
    Colour encodes total objective. Hover shows all 5 objectives and all 4 λs.
    """
    fig = go.Figure(
        data=go.Scatter3d(
            x=df["obj_kin"].to_numpy(),
            y=df["obj_div"].to_numpy(),
            z=df["obj_homeo"].to_numpy(),
            mode="markers",
            marker=dict(
                size=4,
                color=df["obj_total"].to_numpy(),
                colorscale="Viridis",
                colorbar=dict(title="Total Objective"),
                opacity=0.7,
            ),
            customdata=np.column_stack(
                [
                    df["obj_sec"].to_numpy(),
                    df["obj_eff"].to_numpy(),
                    df["lambda_sec"].to_numpy(),
                    df["lambda_eff"].to_numpy(),
                    df["lambda_kin"].to_numpy(),
                    df["lambda_div"].to_numpy(),
                ]
            ),
            hovertemplate=(
                "<b>Kinetic:</b>     %{x:.4e}<br>"
                "<b>Diversity:</b>   %{y:.4e}<br>"
                "<b>Homeostatic:</b> %{z:.4e}<br>"
                "<b>Secretion:</b>   %{customdata[0]:.4e}<br>"
                "<b>Efficiency:</b>  %{customdata[1]:.4e}<br>"
                "<hr>"
                "λ_sec=%{customdata[2]:.2e}  λ_eff=%{customdata[3]:.2e}<br>"
                "λ_kin=%{customdata[4]:.2e}  λ_div=%{customdata[5]:.2e}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="3D Pareto Front: Kinetic / Diversity / Homeostatic",
        scene=dict(
            xaxis_title="Kinetic Objective",
            yaxis_title="Diversity Objective",
            zaxis_title="Homeostatic Objective",
        ),
        width=800,
        height=700,
        template="plotly_white",
    )
    out = os.path.join(OUT_DIR, "pareto_3d.html")
    fig.write_html(out)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run(
    stoichiometry,
    metabolites,
    reaction_names,
    metabolism,
    homeostatic_metabolite_counts,
    homeostatic_dm_targets,
    kinetic,
    maintenance,
    counts_to_molar,
    n_samples: int = 200,
    n_jobs: int = 1,
    seed: int = 42,
    solver_choice=cp.GLOP,
    binary_kinetics_idx=None,
    sample_fn=log_uniform_sample,
) -> pl.DataFrame:
    """
    Run Pareto exploration and generate all three plots.

    `sample_fn(n_samples, seed=seed)` must return an array of shape
    (n_samples, 5) ordered [homeostatic, secretion, efficiency, kinetics,
    diversity]; defaults to the independent log-uniform sampler
    (`log_uniform_sample`). Pass a different callable to explore
    weight combinations constrained by known relationships between terms
    (see pareto_exploration_relationship.py).

    Returns a Polars DataFrame with one row per successful solve containing
    all five lambda values and all five objective term values.
    """
    print(f"Sampling {n_samples} weight combinations (via {sample_fn.__name__})...")
    weight_samples = sample_fn(n_samples, seed=seed)

    fixed = dict(
        stoichiometry=stoichiometry,
        metabolites=metabolites,
        reaction_names=reaction_names,
        metabolism=metabolism,
        homeostatic_metabolite_counts=homeostatic_metabolite_counts,
        homeostatic_dm_targets=homeostatic_dm_targets,
        kinetic=kinetic,
        maintenance=maintenance,
        counts_to_molar=counts_to_molar,
        solver_choice=solver_choice,
        binary_kinetics_idx=binary_kinetics_idx,
    )

    def _solve(i):
        lam_hom, lam_sec, lam_eff, lam_kin, lam_div = weight_samples[i]
        results = solve_one(lam_hom, lam_sec, lam_eff, lam_kin, lam_div, **fixed)

        # convert solution_flux to base_reaction flux
        if results is not None:
            solution_flux = results["solution_flux"]  # in  mmol/L/s
            base_reaction_flux = metabolism.reaction_mapping_matrix.dot(
                solution_flux
            )  # units of mmol/L/s
            pearson_r_squared, r_squared, toya_fluxes = correlations_toya_fluxes(
                metabolism.base_reaction_ids, base_reaction_flux
            )

            results.pop(
                "solution_flux", None
            )  # drop large flux array from results dict
            results["toya_pearson_r_squared"] = pearson_r_squared
            results["toya_r_squared"] = r_squared

        return results

    print(f"Solving {n_samples} problems ({n_jobs} parallel job(s))...")
    if n_jobs == 1:
        results = [_solve(i) for i in tqdm(range(n_samples))]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_solve)(i) for i in tqdm(range(n_samples))
        )

    valid = [r for r in results if r is not None]
    n_failed = n_samples - len(valid)
    if n_failed:
        warnings.warn(f"{n_failed}/{n_samples} solves failed or were infeasible.")
    print(f"  {len(valid)} successful solves.")

    df = pl.DataFrame(valid)
    csv_path = os.path.join(OUT_DIR, "pareto_results.csv")
    df.write_csv(csv_path)
    print(f"  Saved raw results: {csv_path}")

    print("Generating plots...")
    plot_pairwise_altair(df)
    plot_pairwise_altair_weighted(df)
    plot_parallel_coordinates_altair(df)
    plot_3d_plotly(df)
    print("Done.")

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def load_problem_data(sim_out_dir: str):
    """
    Load the fixed problem data run() needs from a fresh homeostatic-only
    whole-cell sim (parquet output from
    `uvenv runscripts/workflow.py --config configs/metabolism_redux_classic.json`),
    mirroring the extraction that 20260127_pairwise_to_homeo_weights.ipynb did
    against the older raw-numpy (0_output.npy + agent_steps.pkl) snapshot
    format, adapted for parquet + a bare (non-live) process reconstruction.
    """
    sim_dirs = sorted(glob.glob(os.path.join(sim_out_dir, "homeostatic_only_*")))
    if not sim_dirs:
        raise FileNotFoundError(
            f"No homeostatic_only_* experiment directories found under "
            f"{sim_out_dir}. Run `uvenv runscripts/workflow.py --config "
            f"configs/metabolism_redux_classic.json` first."
        )
    experiment_dir = sim_dirs[-1]
    experiment_id = os.path.basename(experiment_dir)
    print(f"Loading fixed problem data from: {experiment_dir}")

    # --- Reconstruct the static MetabolismReduxClassic attributes directly
    # from sim_data, without running a live sim (see metabolism_redux_classic.py
    # __init__, lines ~150-311, for what's fully determined by parameters alone).
    sim_data_path = os.path.join(experiment_dir, "parca", "kb", "simData.cPickle")
    load_sim_data = LoadSimData(
        sim_data_path=sim_data_path,
        seed=0,
        fixed_media="minimal",
        condition="basal",
    )
    metabolism_config = load_sim_data.get_metabolism_redux_config()
    metabolism = MetabolismReduxClassic(metabolism_config)

    # `allowed_exchange_uptake` is normally only populated live from
    # environment state on the first next_update() call (metabolism_redux_classic.py:459-478).
    # Media is fixed for the whole sim, so replicate that one-time computation
    # directly from sim_data instead of running a live step.
    exchange_data = load_sim_data.sim_data.external_state.exchange_data_from_media(
        metabolism.media_id
    )
    unconstrained_uptake = exchange_data["importUnconstrainedExchangeMolecules"]
    constrained_uptake = exchange_data["importConstrainedExchangeMolecules"]
    metabolism.allowed_exchange_uptake = set(unconstrained_uptake).union(
        constrained_uptake.keys()
    )
    metabolism.exchange_molecules = set(metabolism.exchange_molecules).union(
        metabolism.allowed_exchange_uptake
    )

    # `new_reaction_idx` is likewise only assigned inside next_update()
    # (metabolism_redux_classic.py:536-538), but is fully static — computed
    # from `reaction_names` and the `fba_new_reaction_ids` parameter alone.
    fba_new_reaction_ids = metabolism.parameters["fba_new_reaction_ids"]
    metabolism.new_reaction_idx = np.where(
        np.isin(metabolism.reaction_names, fba_new_reaction_ids)
    )

    # --- Pull time-resolved FBA listener targets from the parquet history ---
    history_sql, config_sql, success_sql = dataset_sql(sim_out_dir, [experiment_id])
    conn = duckdb.connect()

    homeostatic_metabolites_meta = field_metadata(
        conn, config_sql, "listeners__fba_results__target_homeostatic_dmdt"
    )
    kinetic_constraint_reactions_meta = field_metadata(
        conn, config_sql, "listeners__fba_results__target_kinetic_fluxes"
    )
    if list(homeostatic_metabolites_meta) != list(metabolism.homeostatic_metabolites):
        raise ValueError(
            "Homeostatic metabolite order mismatch between parquet metadata "
            "and the reconstructed process — check fixed_media/condition "
            "passed to LoadSimData."
        )
    if list(kinetic_constraint_reactions_meta) != list(
        metabolism.kinetic_constraint_reactions
    ):
        raise ValueError(
            "Kinetic reaction order mismatch between parquet metadata and "
            "the reconstructed process — check fixed_media/condition passed "
            "to LoadSimData."
        )

    raw = read_stacked_columns(
        history_sql,
        [
            "listeners__fba_results__target_homeostatic_dmdt AS target_homeostatic_dmdt",
            "listeners__fba_results__homeostatic_metabolite_counts AS homeostatic_metabolite_counts",
            "listeners__fba_results__maintenance_target AS maintenance_target",
            "listeners__fba_results__target_kinetic_fluxes AS target_kinetic_fluxes",
            "listeners__enzyme_kinetics__counts_to_molar AS counts_to_molar",
        ],
        remove_first=True,  # first emitted timestep has empty placeholder targets
        conn=conn,
    )

    target_homeostatic_dmdt = ndlist_to_ndarray(raw["target_homeostatic_dmdt"])
    homeostatic_metabolite_counts_ts = ndlist_to_ndarray(
        raw["homeostatic_metabolite_counts"]
    )
    target_kinetic_fluxes = ndlist_to_ndarray(raw["target_kinetic_fluxes"])
    maintenance_target_ts = raw["maintenance_target"].to_numpy()
    counts_to_molar = raw["counts_to_molar"].to_numpy()

    # Counts -> conc, aggregated over time the same way the pairwise-weights
    # notebook did against the old snapshot (homeostatic target: max over
    # time; everything else: mean over time).
    homeostatic_dm_targets_vals = np.max(
        target_homeostatic_dmdt * counts_to_molar[:, None], axis=0
    )
    homeostatic_metabolite_counts_vals = np.mean(
        homeostatic_metabolite_counts_ts * counts_to_molar[:, None], axis=0
    )
    kinetic_vals = np.mean(target_kinetic_fluxes * counts_to_molar[:, None], axis=0)
    maintenance = float(np.mean(maintenance_target_ts * counts_to_molar))

    homeostatic_dm_targets = dict(
        zip(metabolism.homeostatic_metabolites, homeostatic_dm_targets_vals)
    )
    kinetic = dict(zip(metabolism.kinetic_constraint_reactions, kinetic_vals))

    return dict(
        stoichiometry=metabolism.stoichiometry,
        metabolites=metabolism.metabolite_names,
        reaction_names=metabolism.reaction_names,
        metabolism=metabolism,
        homeostatic_metabolite_counts=homeostatic_metabolite_counts_vals,
        homeostatic_dm_targets=homeostatic_dm_targets,
        kinetic=kinetic,
        maintenance=maintenance,
        counts_to_molar=counts_to_molar,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log-uniform Pareto front exploration")
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=6,
        help="Parallel solves via joblib. Note: CVXPY itself is "
        "multi-threaded, so n_jobs * CVXPY threads must fit "
        "within your CPU budget.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sim_out_dir",
        type=str,
        default="out/objective_weights_jul",
        help="Directory containing the fresh homeostatic_only_* sim run "
        "produced by `uvenv runscripts/workflow.py --config "
        "configs/metabolism_redux_classic.json`.",
    )
    args = parser.parse_args()

    problem_data = load_problem_data(args.sim_out_dir)
    run(
        n_samples=args.n_samples,
        n_jobs=args.n_jobs,
        seed=args.seed,
        **problem_data,
    )
