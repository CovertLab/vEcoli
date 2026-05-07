"""
Sobol Global Sensitivity Analysis — NetworkFlowModel Lambda Weights
=====================================================================
Generates N*(k+2) = 7*N Saltelli samples (k=5 lambda inputs), runs the
NetworkFlowModel for each, then computes first-order (S1) and total-order (ST)
Sobol sensitivity indices for all model outputs:

    toya_r_squared          — coefficient of determination vs Toya 2010 fluxes
    obj_homeo, obj_kin,     — unweighted objective terms
    obj_eff, obj_sec,
    obj_div
    solve_failed            — 1 if the solver returned no solution, else 0

Why two analyses (run this file twice with different subsets of the CSV):
    Run 1 — ALL samples, output = toya_r_squared
             → Which lambdas drive feasibility?
    Run 2 — FEASIBLE rows only (toya_r_squared > 0.5), surrogated, outputs =
             obj_homeo + obj_kin
             → Within the feasible region, which lambdas control the
               homeostatic-kinetic Pareto position?
    (Run 2 is handled by a separate surrogate script; this file covers Run 1.)

Failed solves are filled with 0.0 for ALL outputs so that:
  (a) The Sobol estimator receives a complete array (no NaNs)
  (b) Sensitivity of `solve_failed` tells you which inputs cause solver failures

Outputs saved to  sobol_results_N{N}/  (or --out_dir):
    sobol_problem.json       — sampling config
    saltelli_results.csv     — raw model output for all 12*N samples
    sobol_indices.csv        — S1, S1_conf, ST, ST_conf per (lambda, output)
    sobol_bar_all.svg        — panel of bar charts for all outputs
    sobol_bar_{output}.svg   — individual bar chart per output

Prerequisites:
    pip install SALib

Usage (CLI):
    python sobol_sensitivity.py \\
        --time_num 600 \\
        --date 2026-04-06 \\
        --experiment_name homeostatic_only \\
        --condition basal \\
        --experiment_type objective_weight \\
        --N 1024 \\
        --n_jobs 20

Usage (from a notebook — mirrors how 20260303_pareto_exploration.ipynb
calls pareto_exploration.py):

    import os, dill, numpy as np, pandas as pd
    os.chdir(os.path.expanduser('~/dev/vEcoli'))
    from notebooks.Heena notebooks.Metabolism_New_Genes.sobol_sensitivity import run_sobol

    # ... load fba, metabolism, output exactly as in 20260303_pareto_exploration.ipynb ...

    run_sobol(
        stoichiometry=stoichiometry,
        metabolites=metabolites,
        reaction_names=reaction_names,
        metabolism=metabolism,
        homeostatic_metabolite_counts=homeostatic_metabolite_counts,
        homeostatic_dm_targets=homeostatic_dm_targets,
        kinetic=kinetic,
        maintenance=maintenance,
        counts_to_molar=counts_to_molar,
        N=1000,
        n_jobs=20,
    )
"""

import argparse
import json
import os
import pickle
import warnings
from math import log10
from typing import Optional

import cvxpy as cp
import matplotlib
import numpy as np
import pandas as pd
from fsspec import open as fsspec_open
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
from wholecell.utils import units, toya
from ecoli.processes.metabolism_redux_classic import (
    FlowResult,
    FREE_RXNS,
    NetworkFlowModel,
)

matplotlib.use("Agg")
plt.style.use("default")
try:
    # SALib >= 1.5 renamed saltelli -> sobol; fall back for older versions.
    try:
        from SALib.sample import sobol as saltelli_sampler
    except ImportError:
        from SALib.sample import saltelli as saltelli_sampler  # type: ignore[no-redef]
    from SALib.analyze import sobol as sobol_analyzer
except ImportError as e:
    raise ImportError(
        "SALib is required for Sobol analysis. Install it with:\n    pip install SALib"
    ) from e

os.chdir(os.path.expanduser("~/dev/vEcoli/"))

# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------
COUNTS_UNITS = units.mmol
VOLUME_UNITS = units.L
TIME_UNITS = units.s
FLUX_UNITS = COUNTS_UNITS / VOLUME_UNITS / TIME_UNITS

# ---------------------------------------------------------------------------
# Weight ranges — keep in sync with pareto_exploration.py so that Sobol
# indices are directly comparable to the random-sample pareto results.
# ---------------------------------------------------------------------------
WEIGHT_RANGES = {
    "homeostatic": (1e-3, 1.0),
    "secretion": (1e-7, 1e-4),
    "efficiency": (1e-7, 1e-4),
    "kinetics": (1e-5, 1e-3),
    "diversity": (1e-5, 1e-2),
}

LAMBDA_NAMES = ["lambda_hom", "lambda_sec", "lambda_eff", "lambda_kin", "lambda_div"]

# Outputs for which Sobol indices are computed
SOBOL_TARGETS = [
    "toya_r_squared",
    "obj_homeo",
    "obj_kin",
    "obj_eff",
    "obj_sec",
    "obj_div",
    "solve_failed",
]

# All numeric output columns produced by each solve (excluding lambdas and solve_failed)
_OBJECTIVE_COLS = [
    "obj_homeo",
    "obj_kin",
    "obj_eff",
    "obj_sec",
    "obj_div",
    "obj_hom_w",
    "obj_kin_w",
    "obj_eff_w",
    "obj_sec_w",
    "obj_div_w",
    "toya_pearson_r_squared",
    "toya_r_squared",
]

# Human-readable labels for plots
_LAMBDA_LABELS = [n.replace("lambda_", "λ_") for n in LAMBDA_NAMES]


# ---------------------------------------------------------------------------
# SALib problem definition (sampling in log10 space)
# ---------------------------------------------------------------------------
def _make_salib_problem() -> dict:
    """Build SALib problem dict with bounds in log10(lambda) space."""
    return {
        "num_vars": len(WEIGHT_RANGES),
        "names": [f"log10_{n}" for n in LAMBDA_NAMES],
        "bounds": [[log10(lo), log10(hi)] for lo, hi in WEIGHT_RANGES.values()],
    }


# ---------------------------------------------------------------------------
# Saltelli sampling
# ---------------------------------------------------------------------------
def saltelli_sample(N: int, seed: int = 42) -> np.ndarray:
    """
    Generate N*(k+2) = 7*N samples in original lambda space using Saltelli's
    extension of Sobol sequences, sampling uniformly in log10 space.
    calc_second_order=False gives first-order + total-order indices only.

    IMPORTANT: N must be a power of 2 (512, 1024, 2048 ...) for the Sobol
    sequence to have its convergence guarantees. Non-power-of-2 values work
    but emit a warning and give slightly worse space-filling.

    Returns
    -------
    np.ndarray of shape (N*(k+2), 5) in original lambda space (10**log10).
    """
    problem = _make_salib_problem()
    # Saltelli/Sobol sequences are quasi-random (low-discrepancy), so they
    # are fully deterministic — no seed parameter is needed or available.
    log10_samples = saltelli_sampler.sample(problem, N, calc_second_order=False)
    return 10**log10_samples


# ---------------------------------------------------------------------------
# Load simulation data (mirrors load_sim in 20260303_pareto_exploration.ipynb)
# ---------------------------------------------------------------------------
def load_sim(
    time_num: int,
    date: str,
    experiment_name: str,
    condition: str,
    experiment_type: str,
):
    """Load agent state and FBA listener data from a completed simulation."""
    import dill

    entry = f"{experiment_name}_{time_num}_{date}"
    folder = f"out/{experiment_type}/{condition}/{entry}/"
    output = np.load(folder + "0_output.npy", allow_pickle=True).item()
    output = output["agents"]["0"]
    fba = output["listeners"]["fba_results"]
    bulk = pd.DataFrame(output["bulk"])
    with open(folder + "agent_steps.pkl", "rb") as f:
        agent = dill.load(f)
    metabolism = agent["ecoli-metabolism-redux-classic"]
    return fba, bulk, metabolism, output


# ---------------------------------------------------------------------------
# Toya flux correlation (mirrors correlations_toya_fluxes in pareto_exploration.py)
# ---------------------------------------------------------------------------
def _correlations_toya_fluxes(
    reaction_ids, sim_reaction_flux: np.ndarray
) -> tuple[float, float]:
    validation_data_path = "out/kb/validationData.cPickle"
    with fsspec_open(validation_data_path, "rb") as f:
        validation_data = pickle.load(f)

    cell_mass = units.fg * 1745.814482240506
    dry_mass = units.fg * 524.0582963771143
    cell_density = units.g / units.L * 1100

    toya_reactions = validation_data.reactionFlux.toya2010fluxes["reactionID"]
    toya_fluxes = toya.adjust_toya_data(
        validation_data.reactionFlux.toya2010fluxes["reactionFlux"],
        cell_mass,
        dry_mass,
        cell_density,
    )

    sim_fluxes_2d = FLUX_UNITS * np.vstack([sim_reaction_flux, sim_reaction_flux])
    sim_means, _ = toya.process_simulated_fluxes(
        toya_reactions, reaction_ids, sim_fluxes_2d
    )
    toya_means = toya.process_toya_data(toya_reactions, toya_reactions, toya_fluxes)

    sim_num = sim_means.asNumber(FLUX_UNITS)
    toya_num = toya_means.asNumber(FLUX_UNITS)

    pearson_r2 = float(np.corrcoef(sim_num, toya_num)[0, 1]) ** 2
    ss_res = np.sum((sim_num - toya_num) ** 2)
    ss_tot = np.sum((toya_num - np.mean(toya_num)) ** 2)
    r2 = float(1 - ss_res / ss_tot)
    return pearson_r2, r2


# ---------------------------------------------------------------------------
# Single solve (mirrors solve_one in pareto_exploration.py)
# ---------------------------------------------------------------------------
def _solve_one(
    lam_hom: float,
    lam_sec: float,
    lam_eff: float,
    lam_kin: float,
    lam_div: float,
    stoichiometry,
    metabolites,
    reaction_names,
    metabolism,
    homeostatic_metabolite_counts,
    homeostatic_dm_targets,
    kinetic,
    maintenance,
    counts_to_molar,
    solver_choice=cp.GLOP,
    binary_kinetics_idx=None,
) -> Optional[dict]:
    """Solve the NetworkFlowModel for one weight combination. Returns None on failure."""
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
            solver=solver_choice,
        )
        return {
            "obj_homeo": solution.homeostatic_term,
            "obj_kin": solution.kinetics_term,
            "obj_eff": solution.efficiency_term,
            "obj_sec": solution.secretion_term,
            "obj_div": solution.diversity_term,
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
# Main run function (callable from a notebook)
# ---------------------------------------------------------------------------
def run_sobol(
    stoichiometry,
    metabolites,
    reaction_names,
    metabolism,
    homeostatic_metabolite_counts,
    homeostatic_dm_targets,
    kinetic,
    maintenance,
    counts_to_molar,
    N: int = 1000,
    n_jobs: int = 1,
    seed: int = 42,
    out_dir: Optional[str] = None,
    solver_choice=cp.GLOP,
    binary_kinetics_idx=None,
) -> pd.DataFrame:
    """
    Run Sobol GSA on the NetworkFlowModel.

    Parameters
    ----------
    N : int
        Saltelli N. Total solves = N * (k + 2) = 7*N for k=5 inputs.
    n_jobs : int
        Parallel workers passed to joblib.
    out_dir : str, optional
        Output directory. Defaults to
        'notebooks/Heena notebooks/Metabolism_New Genes/sobol_results_N{N}'.

    Returns
    -------
    pd.DataFrame
        Raw Saltelli results (one row per solve).
    """
    if out_dir is None:
        out_dir = f"notebooks/Heena notebooks/Metabolism_New Genes/sobol_results_N{N}"
    os.makedirs(out_dir, exist_ok=True)

    n_total = N * (len(WEIGHT_RANGES) + 2)  # = 7*N for k=5
    print(f"Generating {n_total} Saltelli samples (N={N}, k={len(WEIGHT_RANGES)})...")
    samples = saltelli_sample(N, seed=seed)  # (n_total, 5)

    problem_meta = {
        "N": N,
        "k": len(WEIGHT_RANGES),
        "total_samples": n_total,
        "seed": seed,
        "weight_ranges": WEIGHT_RANGES,
    }
    with open(os.path.join(out_dir, "sobol_problem.json"), "w") as fp:
        json.dump(problem_meta, fp, indent=2)

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

    def _solve(i: int) -> dict:
        lam_hom, lam_sec, lam_eff, lam_kin, lam_div = samples[i]
        row = {
            "lambda_hom": lam_hom,
            "lambda_sec": lam_sec,
            "lambda_eff": lam_eff,
            "lambda_kin": lam_kin,
            "lambda_div": lam_div,
            "solve_failed": 0,
        }
        result = _solve_one(lam_hom, lam_sec, lam_eff, lam_kin, lam_div, **fixed)
        if result is None:
            # Fill with 0 so Sobol estimator gets a complete array.
            # solve_failed=1 allows a separate Sobol analysis of "what causes failures".
            row["solve_failed"] = 1
            for col in _OBJECTIVE_COLS:
                row[col] = 0.0
        else:
            solution_flux = result.pop("solution_flux")
            base_flux = metabolism.reaction_mapping_matrix.dot(solution_flux)
            pearson_r2, r2 = _correlations_toya_fluxes(
                metabolism.base_reaction_ids, base_flux
            )
            result["toya_pearson_r_squared"] = pearson_r2
            result["toya_r_squared"] = r2
            row.update(result)
        return row

    print(f"Solving {n_total} problems ({n_jobs} parallel job(s))...")
    if n_jobs == 1:
        rows = [_solve(i) for i in tqdm(range(n_total))]
    else:
        rows = Parallel(n_jobs=n_jobs)(delayed(_solve)(i) for i in tqdm(range(n_total)))

    df = pd.DataFrame(rows)
    n_failed = int(df["solve_failed"].sum())
    if n_failed:
        warnings.warn(
            f"{n_failed}/{n_total} solves failed and were filled with zeros. "
            "Check sobol_indices.csv column 'solve_failed' for which lambdas are responsible."
        )
    print(f"  {n_total - n_failed}/{n_total} successful solves.")

    csv_path = os.path.join(out_dir, "saltelli_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # ------------------------------------------------------------------
    # Sobol analysis
    # ------------------------------------------------------------------
    print("Computing Sobol indices...")
    problem = _make_salib_problem()
    all_indices = []

    for output_col in SOBOL_TARGETS:
        Y = df[output_col].to_numpy(dtype=float)
        Si = sobol_analyzer.analyze(
            problem, Y, calc_second_order=False, print_to_console=False
        )
        for j, lam_name in enumerate(LAMBDA_NAMES):
            all_indices.append(
                {
                    "output": output_col,
                    "lambda": lam_name,
                    "S1": float(Si["S1"][j]),
                    "S1_conf": float(Si["S1_conf"][j]),
                    "ST": float(Si["ST"][j]),
                    "ST_conf": float(Si["ST_conf"][j]),
                }
            )

    indices_df = pd.DataFrame(all_indices)
    indices_path = os.path.join(out_dir, "sobol_indices.csv")
    indices_df.to_csv(indices_path, index=False)
    print(f"  Saved: {indices_path}")

    # ------------------------------------------------------------------
    # Visualize
    # ------------------------------------------------------------------
    print("Generating bar charts...")
    _plot_sobol_bars(indices_df, out_dir)

    print("Done.")
    return df


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def _plot_sobol_bars(indices_df: pd.DataFrame, out_dir: str) -> None:
    """Save one bar chart per output variable, plus a combined panel."""
    outputs = list(indices_df["output"].unique())
    n_out = len(outputs)
    x = np.arange(len(LAMBDA_NAMES))
    width = 0.35

    # Combined panel
    fig, axes = plt.subplots(1, n_out, figsize=(4 * n_out, 4.5), sharey=False)
    if n_out == 1:
        axes = [axes]

    for ax, output_col in zip(axes, outputs):
        sub = indices_df[indices_df["output"] == output_col]
        ax.bar(
            x - width / 2,
            sub["S1"].values,
            width,
            yerr=sub["S1_conf"].values,
            label="S1 (first-order)",
            capsize=3,
            color="#4C72B0",
            alpha=0.85,
        )
        ax.bar(
            x + width / 2,
            sub["ST"].values,
            width,
            yerr=sub["ST_conf"].values,
            label="ST (total-order)",
            capsize=3,
            color="#DD8452",
            alpha=0.85,
        )
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(_LAMBDA_LABELS, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Sobol Index")
        ax.set_title(output_col, fontsize=10)
        ax.legend(fontsize=7)

    fig.suptitle("Sobol Sensitivity Indices — All Samples", fontsize=12, y=1.01)
    plt.tight_layout()
    panel_path = os.path.join(out_dir, "sobol_bar_all.svg")
    fig.savefig(panel_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {panel_path}")

    # Individual charts
    for output_col in outputs:
        sub = indices_df[indices_df["output"] == output_col]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(
            x - width / 2,
            sub["S1"].values,
            width,
            yerr=sub["S1_conf"].values,
            label="S1 (first-order)",
            capsize=4,
            color="#4C72B0",
            alpha=0.85,
        )
        ax.bar(
            x + width / 2,
            sub["ST"].values,
            width,
            yerr=sub["ST_conf"].values,
            label="ST (total-order)",
            capsize=4,
            color="#DD8452",
            alpha=0.85,
        )
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(_LAMBDA_LABELS, rotation=35, ha="right")
        ax.set_ylabel("Sobol Index")
        ax.set_title(f"Sobol Sensitivity — {output_col}", fontsize=12)
        ax.legend()
        plt.tight_layout()
        safe = output_col.replace("/", "_")
        fig.savefig(
            os.path.join(out_dir, f"sobol_bar_{safe}.svg"), dpi=150, bbox_inches="tight"
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sobol GSA for NetworkFlowModel lambda weights",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--time_num", type=int, default=600, help="Simulation time step to load"
    )
    parser.add_argument(
        "--date",
        type=str,
        default="2026-04-06",
        help="Simulation date string (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="homeostatic_only",
        help="Experiment name folder",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="basal",
        help="Condition subfolder (e.g. basal)",
    )
    parser.add_argument(
        "--experiment_type",
        type=str,
        default="objective_weight",
        help="Experiment type top-level folder",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=1024,
        help="Saltelli N — MUST be a power of 2 (512, 1024, 2048). "
        "Total solves = N*(k+2) = 7*N for k=5.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Parallel workers (joblib). CVXPY is already "
        "multi-threaded, so n_jobs * CVXPY threads must "
        "fit within your CPU budget.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: sobol_results_N{N}/)",
    )
    args = parser.parse_args()

    print(
        f"Loading sim: {args.experiment_name}_{args.time_num}_{args.date} "
        f"({args.condition}/{args.experiment_type})"
    )
    fba, bulk, metabolism, output = load_sim(
        args.time_num,
        args.date,
        args.experiment_name,
        args.condition,
        args.experiment_type,
    )

    stoichiometry = metabolism.stoichiometry.copy()
    reaction_names = metabolism.reaction_names
    metabolites = metabolism.metabolite_names.copy()
    counts_to_molar = output["listeners"]["enzyme_kinetics"]["counts_to_molar"]

    homeostatic_dm_targets = (
        pd.DataFrame(
            fba["target_homeostatic_dmdt"],
            columns=metabolism.homeostatic_metabolites,
        )
        .mul(counts_to_molar, axis=0)
        .iloc[1]
    )

    homeostatic_metabolite_counts = (
        pd.DataFrame(
            fba["homeostatic_metabolite_counts"],
            columns=metabolism.homeostatic_metabolites,
        )
        .mul(counts_to_molar, axis=0)
        .iloc[1]
    )

    maintenance = (
        pd.DataFrame(
            fba["maintenance_target"][1:],
            columns=["maintenance_reaction"],
        )
        .mul(counts_to_molar[1:], axis=0)
        .iloc[1]
    )

    kinetic = (
        pd.DataFrame(
            fba["target_kinetic_fluxes"],
            columns=metabolism.kinetic_constraint_reactions,
        )
        .mul(counts_to_molar, axis=0)
        .iloc[1]
    )

    run_sobol(
        stoichiometry=stoichiometry,
        metabolites=metabolites,
        reaction_names=reaction_names,
        metabolism=metabolism,
        homeostatic_metabolite_counts=homeostatic_metabolite_counts,
        homeostatic_dm_targets=homeostatic_dm_targets,
        kinetic=kinetic,
        maintenance=maintenance,
        counts_to_molar=counts_to_molar,
        N=args.N,
        n_jobs=args.n_jobs,
        seed=args.seed,
        out_dir=args.out_dir,
    )
