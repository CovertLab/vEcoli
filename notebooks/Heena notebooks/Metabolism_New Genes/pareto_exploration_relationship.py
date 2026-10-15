"""
Relationship-constrained twin of pareto_exploration.py.

Instead of sampling all five lambda weights independently in log space, draws
lambda_hom and lambda_eff independently and derives the other three weights
from log-ratios:

    log10(lambda_hom) ~ U(HOM_RANGE)
    log10(lambda_eff) ~ U(EFF_RANGE)
    log10(lambda_kin) = log10(lambda_hom) - U(HOM_KIN_RATIO_RANGE)
    log10(lambda_sec) = log10(lambda_kin) + U(SEC_KIN_RATIO_RANGE)
    log10(lambda_div) = log10(lambda_hom) - U(HOM_DIV_RATIO_RANGE)

Draws are rejected unless ratio_hom_sec = ratio_hom_kin - ratio_sec_kin is in
HOM_SEC_TARGET_RANGE. Rejection happens before the CVXPY solve, so it costs
almost nothing; only accepted draws reach run()'s solve loop.

This reuses pareto_exploration.py's run()/solve_one()/load_problem_data()/
plotting functions as-is (they don't care how weight samples were
generated) rather than duplicating them -- only the sampling function
differs. See run()'s `sample_fn` parameter in pareto_exploration.py.

Usage:
    uvenv pareto_exploration_relationship.py
    uvenv "notebooks/Heena notebooks/Metabolism_New Genes/pareto_exploration_relationship.py" --n_samples 500 --n_jobs 4
"""

import argparse
import json
import os

import numpy as np

import pareto_exploration as pe

# ---------------------------------------------------------------------------
# Draw ranges. All are log10-space quantities, drawn linear-uniform.
# ---------------------------------------------------------------------------
HOM_RANGE = (-3, 1)  # log10(lambda_hom)
EFF_RANGE = (-6, -1)  # log10(lambda_eff)
HOM_KIN_RATIO_RANGE = (1, 5)  # log10(lambda_hom) - log10(lambda_kin)
SEC_KIN_RATIO_RANGE = (2, 4)  # log10(lambda_sec) - log10(lambda_kin)
HOM_DIV_RATIO_RANGE = (4, 8)  # log10(lambda_hom) - log10(lambda_div)

# Rejection window for ratio_hom_sec = ratio_hom_kin - ratio_sec_kin (derived
# from the draws above, not drawn directly). Set to None to disable.
HOM_SEC_TARGET_RANGE = (0, 4)

OUT_DIR = "notebooks/Heena notebooks/Metabolism_New Genes/pareto_results_enzymatic_v4_10000samples_try_kin"
pe.OUT_DIR = OUT_DIR  # redirect run()'s output (CSV + 4 plots) to this directory


def implied_ranges() -> dict:
    """Ranges of the derived log-weights / ratios implied by the draw ranges."""

    def diff(a, b):
        return (a[0] - b[1], a[1] - b[0])

    def add(a, b):
        return (a[0] + b[0], a[1] + b[1])

    log_kin = diff(HOM_RANGE, HOM_KIN_RATIO_RANGE)
    return {
        "log_lambda_kin": log_kin,
        "log_lambda_sec": add(log_kin, SEC_KIN_RATIO_RANGE),
        "log_lambda_div": diff(HOM_RANGE, HOM_DIV_RATIO_RANGE),
        "ratio_hom_sec": diff(HOM_KIN_RATIO_RANGE, SEC_KIN_RATIO_RANGE),
        "ratio_hom_eff": diff(HOM_RANGE, EFF_RANGE),
        "ratio_eff_kin": diff(EFF_RANGE, log_kin),
    }


def _in_window(x: np.ndarray, window) -> np.ndarray:
    if window is None:
        return np.ones(x.shape, dtype=bool)
    return (x >= window[0]) & (x <= window[1])


def check_target_ranges() -> None:
    """Fail early if the target window can't be reached from the draw ranges."""
    if HOM_SEC_TARGET_RANGE is None:
        return
    lo, hi = implied_ranges()["ratio_hom_sec"]
    if HOM_SEC_TARGET_RANGE[1] < lo or HOM_SEC_TARGET_RANGE[0] > hi:
        raise ValueError(
            f"ratio_hom_sec target {HOM_SEC_TARGET_RANGE} does not overlap its "
            f"reachable range ({lo}, {hi}); every draw would be rejected."
        )


def _draw(rng: np.random.Generator, size: int) -> dict:
    return {
        "log_hom": rng.uniform(*HOM_RANGE, size=size),
        "log_eff": rng.uniform(*EFF_RANGE, size=size),
        "ratio_hom_kin": rng.uniform(*HOM_KIN_RATIO_RANGE, size=size),
        "ratio_sec_kin": rng.uniform(*SEC_KIN_RATIO_RANGE, size=size),
        "ratio_hom_div": rng.uniform(*HOM_DIV_RATIO_RANGE, size=size),
    }


def _accept(draws: dict) -> np.ndarray:
    ratio_hom_sec = draws["ratio_hom_kin"] - draws["ratio_sec_kin"]
    return _in_window(ratio_hom_sec, HOM_SEC_TARGET_RANGE)


def acceptance_rate(n_trial: int = 200_000, seed: int = 0) -> float:
    """Fraction of draws that pass the target window (Monte Carlo estimate)."""
    return float(_accept(_draw(np.random.default_rng(seed), n_trial)).mean())


def relationship_sample(n_samples: int, seed: int = 42) -> np.ndarray:
    """
    Draw log10(lambda_hom), log10(lambda_eff), ratio_hom_kin, ratio_sec_kin
    and ratio_hom_div independently (uniform in their ranges), reject draws
    whose ratio_hom_sec falls outside HOM_SEC_TARGET_RANGE, and derive the
    five weights from the accepted draws.

    Returns array of shape (n_samples, 5) with columns ordered
    [homeostatic, secretion, efficiency, kinetics, diversity], matching
    pareto_exploration.log_uniform_sample()'s convention.
    """
    check_target_ranges()
    rng = np.random.default_rng(seed)

    accepted: dict = {}
    n_accepted = 0
    n_drawn = 0
    n_passed = 0
    while n_accepted < n_samples:
        n_remaining = n_samples - n_accepted
        draws = _draw(rng, max(n_remaining * 5, 1000))
        passed = np.flatnonzero(_accept(draws))
        n_drawn += len(draws["log_hom"])
        n_passed += len(passed)
        idx = passed[:n_remaining]
        for key, values in draws.items():
            accepted.setdefault(key, []).append(values[idx])
        n_accepted += len(idx)
        if n_drawn > 1000 * max(n_samples, 1000):
            raise RuntimeError(
                f"Only {n_accepted}/{n_samples} draws accepted after {n_drawn} "
                "tries; the target window is too narrow for the draw ranges."
            )
    print(
        f"  rejection sampling: {n_passed}/{n_drawn} draws "
        f"({n_passed / n_drawn:.0%}) passed the target window; kept {n_samples}"
    )

    log_hom = np.concatenate(accepted["log_hom"])
    log_eff = np.concatenate(accepted["log_eff"])
    ratio_hom_kin = np.concatenate(accepted["ratio_hom_kin"])
    ratio_sec_kin = np.concatenate(accepted["ratio_sec_kin"])
    ratio_hom_div = np.concatenate(accepted["ratio_hom_div"])

    log_kin = log_hom - ratio_hom_kin
    lambda_hom = 10**log_hom
    lambda_eff = 10**log_eff
    lambda_kin = 10**log_kin
    lambda_sec = 10 ** (log_kin + ratio_sec_kin)
    lambda_div = 10 ** (log_hom - ratio_hom_div)

    return np.column_stack([lambda_hom, lambda_sec, lambda_eff, lambda_kin, lambda_div])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Relationship-constrained Pareto front exploration"
    )
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
        help="Directory containing the homeostatic_only_* sim run produced "
        "by `uvenv runscripts/workflow.py --config "
        "configs/metabolism_redux_classic.json`.",
    )
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/weight_info.json", "w") as fp:
        json.dump(
            {
                "hom_range": HOM_RANGE,
                "eff_range": EFF_RANGE,
                "hom_kin_ratio_range": HOM_KIN_RATIO_RANGE,
                "sec_kin_ratio_range": SEC_KIN_RATIO_RANGE,
                "hom_div_ratio_range": HOM_DIV_RATIO_RANGE,
                "hom_sec_target_range": HOM_SEC_TARGET_RANGE,
                "acceptance_rate": acceptance_rate(),
                "implied": implied_ranges(),
            },
            fp,
            indent=2,
        )

    problem_data = pe.load_problem_data(args.sim_out_dir)
    pe.run(
        n_samples=args.n_samples,
        n_jobs=args.n_jobs,
        seed=args.seed,
        sample_fn=relationship_sample,
        **problem_data,
    )
