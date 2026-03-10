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
    NetworkFlowModel,
)
import argparse
import os
import warnings
from typing import Optional

import altair as alt
import cvxpy as cp
import numpy as np
import plotly.graph_objects as go
import polars as pl
from joblib import Parallel, delayed
from tqdm import tqdm

os.chdir(os.path.expanduser("~/dev/vEcoli/"))


# ---------------------------------------------------------------------------
# Feasible weight ranges (from pairwise analysis). Log-spaced sampling.
# Homeostatic weight is always fixed at 1.
# ---------------------------------------------------------------------------
WEIGHT_RANGES = {
    "secretion": (1e-5, 1e-3),
    "efficiency": (1e-6, 1e-4),
    "kinetics": (1e-4, 1e-2),
    "diversity": (1e-4, 1e-2),
}

OUT_DIR = (
    "notebooks/Heena notebooks/Metabolism_New Genes/pareto_results_shrunk_1000samples_2"
)
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def log_uniform_sample(n_samples: int, seed: int = 42) -> np.ndarray:
    """
    Draw n_samples weight combinations uniformly in log space within each
    term's feasible range.

    Returns array of shape (n_samples, 4) with columns ordered as
    WEIGHT_RANGES: [secretion, efficiency, kinetics, diversity].
    """
    rng = np.random.default_rng(seed)
    samples = []
    for lo, hi in WEIGHT_RANGES.values():
        log_samples = rng.uniform(np.log10(lo), np.log10(hi), size=n_samples)
        samples.append(10**log_samples)
    return np.column_stack(samples)  # (n_samples, 4)


# ---------------------------------------------------------------------------
# Single solve — wraps existing NetworkFlowModel
# ---------------------------------------------------------------------------
def solve_one(
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
) -> Optional[dict]:
    """
    Build and solve the NetworkFlowModel for one weight combination.
    Returns a flat dict of weights + objective term values, or None on failure.
    """
    weights = {
        "homeostatic": 1.0,
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
            solver=solver_choice,
        )
        return {
            "lambda_sec": lam_sec,
            "lambda_eff": lam_eff,
            "lambda_kin": lam_kin,
            "lambda_div": lam_div,
            "obj_total": solution.objective,
            "obj_homeo": solution.homeostatic_term,
            "obj_kin": solution.kinetics_term,
            "obj_eff": solution.efficiency_term,
            "obj_sec": solution.secretion_term,
            "obj_div": solution.diversity_term,
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
            .encode(
                y=alt.Y("obj_homeo:Q", title="Homeostatic Objective"),
                x=alt.X(f"{obj_col}:Q", title=f"{title} Objective"),
                color=alt.condition(
                    interval,
                    alt.Color(
                        f"{lam_col}:Q",
                        scale=alt.Scale(scheme="viridis"),
                        title=f"λ_{title[:3].lower()}",
                    ),
                    alt.value("lightgray"),
                ),
                tooltip=[
                    alt.Tooltip("obj_homeo:Q", format=".4e"),
                    alt.Tooltip(f"{obj_col}:Q", format=".4e"),
                    alt.Tooltip(
                        f"{lam_col}:Q", format=".2e", title=f"λ_{title[:3].lower()}"
                    ),
                ],
            )
            .properties(title=f"Homeostatic vs {title}", width=280, height=250)
        ).add_selection(interval)
        charts.append(chart)

    combined_scatter = ((charts[0] | charts[1]) & (charts[2] | charts[3])).properties(
        title="Pairwise Pareto: Homeostatic vs Secondary Objectives"
    )

    # --- Make Table Alongside Selection (Rank By Homeostatic Objective)---
    # Base chart for data tables
    ranked_text = (
        alt.Chart(df)
        .mark_text(align="right")
        .encode(y=alt.Y("rank:O", axis=None))
        .transform_filter(interval)
        .transform_window(
            rank="rank()",
            sort=[
                alt.SortField("obj_homeo", order="ascending"),
                alt.SortField("obj_kin", order="ascending"),
            ],
        )
        .transform_filter(alt.datum.rank <= 10)
        .properties(height=240)
    )

    # Data Tables
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
    obj_homeo = ranked_text.encode(
        text=alt.Text("obj_homeo:Q", format=".3e")
    ).properties(title=alt.Title(text="Homeostatic Objective", align="right"))
    obj_kinetic = ranked_text.encode(
        text=alt.Text("obj_kin:Q", format=".3f")
    ).properties(title=alt.Title(text="Unweighted Kinetic Objective", align="right"))
    text = alt.hconcat(
        lambda_sec, lambda_eff, lambda_kin, lambda_div, obj_homeo, obj_kinetic
    )

    density = (
        alt.Chart(df)
        .transform_filter(interval)
        .transform_fold(
            ["lambda_sec", "lambda_eff", "lambda_kin", "lambda_div"],
            as_=["lambda_type", "value"],
        )
        .transform_density(
            density="value",
            groupby=["lambda_type"],
            as_=["value", "density"],
        )
        .mark_area(opacity=0.4)
        .encode(
            x=alt.X("value:Q", title="Lambda Value").scale(type="log"),
            y=alt.Y("density:Q", title="Density", stack=False),
            color=alt.Color(
                "lambda_type:N",
                title="Lambda",
                legend=alt.Legend(orient="none", legendX=550, legendY=300),
            ),
        )
        .properties(title="Distribution of lambda weights", width=500, height=300)
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
) -> pl.DataFrame:
    """
    Run log-uniform Pareto exploration and generate all three plots.

    Returns a Polars DataFrame with one row per successful solve containing
    all four lambda values and all five objective term values.
    """
    print(
        f"Sampling {n_samples} weight combinations (log-uniform in feasible ranges)..."
    )
    weight_samples = log_uniform_sample(n_samples, seed=seed)

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
        lam_sec, lam_eff, lam_kin, lam_div = weight_samples[i]
        return solve_one(lam_sec, lam_eff, lam_kin, lam_div, **fixed)

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
    plot_parallel_coordinates_altair(df)
    plot_3d_plotly(df)
    print("Done.")

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log-uniform Pareto front exploration")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Parallel solves via joblib. Note: CVXPY itself is "
        "multi-threaded, so n_jobs * CVXPY threads must fit "
        "within your CPU budget.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load problem data here, then call run().
    # This mirrors the setup in other notebook before test_NetworkFlowModel.
    # ------------------------------------------------------------------
    # Example:
    #   run(
    #       stoichiometry=stoichiometry,
    #       metabolites=metabolites,
    #       reaction_names=reaction_names,
    #       metabolism=metabolism,
    #       homeostatic_metabolite_counts=homeostatic_metabolite_counts,
    #       homeostatic_dm_targets=homeostatic_dm_targets,
    #       kinetic=kinetic,
    #       maintenance=maintenance,
    #       counts_to_molar=counts_to_molar,
    #       n_samples=args.n_samples,
    #       n_jobs=args.n_jobs,
    #       seed=args.seed,
    #   )
    raise NotImplementedError("Fill in problem data above, then call run(). ")
