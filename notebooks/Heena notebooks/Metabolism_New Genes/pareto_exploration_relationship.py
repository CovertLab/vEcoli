"""
Relationship-constrained twin of pareto_exploration.py.

Instead of sampling all five lambda weights independently in log space,
draws `lambda_hom`, `lambda_div`, and `lambda_sec` independently (no known
explicit relationship for lambda_sec), then *derives* `lambda_kin` and
`lambda_eff` from relationships found in prior analysis:

- lambda_kin = lambda_hom * ratio, ratio drawn log-uniform in
  KIN_HOM_RATIO_RANGE.
- lambda_eff is drawn uniformly (in log space) from the region at or above
  the fitted lower-bound line log10(lambda_eff) = 1.1*log10(lambda_kin) - 0.5
  (2026-07-24 refit) — i.e. the "lower-right triangular region" visible in
  the lambda_eff-vs-lambda_kin feasibility scatter
  (notebooks/Heena notebooks/Metabolism_New Genes/out/objective_weights/
  pareto_conditional_analysis/lamda_pairwise_scatter_feasibility.svg),
  sampled directly (no rejection needed) rather than placed exactly on the
  line.

This reuses pareto_exploration.py's run()/solve_one()/load_problem_data()/
plotting functions as-is (they don't care how weight samples were
generated) rather than duplicating them — only the sampling function
differs. See run()'s `sample_fn` parameter in pareto_exploration.py.

Usage:
    python pareto_exploration_relationship.py
    python pareto_exploration_relationship.py --n_samples 500 --n_jobs 4
"""

import argparse
import json
import os

import numpy as np

import pareto_exploration as pe

# ---------------------------------------------------------------------------
# Independent-draw ranges (log-uniform), same defaults as
# pareto_exploration.WEIGHT_RANGES — override here if needed.
# ---------------------------------------------------------------------------
HOM_RANGE = (1e-3, 1.0)
DIV_RANGE = (1e-5, 1e-2)
SEC_RANGE = (1e-7, 1e-4)  # no known explicit relationship for lambda_sec

# lambda_kin is derived from lambda_hom via this ratio band (log-uniform).
KIN_HOM_RATIO_RANGE = (0.0005, 0.005)

# lambda_eff is derived from lambda_kin: drawn uniformly (log space) from
# [fitted lower bound, EFF_LOG_RANGE[1]], clipped into EFF_LOG_RANGE so the
# sampled range never inverts (can happen when lambda_kin is large enough
# that the fitted lower bound would exceed EFF_LOG_RANGE[1]).
EFF_LOG_RANGE = (-7.0, -4.0)  # bounds for log10(lambda_eff), same as the
# efficiency range in pareto_exploration.WEIGHT_RANGES
EFF_KIN_LOG_SLOPE = 2
EFF_KIN_LOG_INTERCEPT = 1.8

OUT_DIR = "notebooks/Heena notebooks/Metabolism_New Genes/pareto_results_relationship_v1_10000samples"
pe.OUT_DIR = OUT_DIR  # redirect run()'s output (CSV + 4 plots) to this directory


def derive_lambda_kin(lambda_hom, ratio):
    return lambda_hom * ratio


def derive_lambda_eff(lambda_kin, rng):
    # return 10 ** np.clip(
    #     EFF_KIN_LOG_SLOPE * np.log10(lambda_kin) + EFF_KIN_LOG_INTERCEPT,
    #     EFF_LOG_RANGE[0],
    #     EFF_LOG_RANGE[1],
    # )
    return 10 ** rng.uniform(
        np.clip(
            EFF_KIN_LOG_SLOPE * np.log10(lambda_kin) + EFF_KIN_LOG_INTERCEPT,
            EFF_LOG_RANGE[0],
            EFF_LOG_RANGE[1],
        ),
        EFF_LOG_RANGE[1],
    )


def relationship_sample(n_samples: int, seed: int = 42) -> np.ndarray:
    """
    Draw lambda_hom and lambda_div independently (log-uniform); derive
    lambda_kin, lambda_eff, lambda_sec from the relationships above.

    Returns array of shape (n_samples, 5) with columns ordered
    [homeostatic, secretion, efficiency, kinetics, diversity], matching
    pareto_exploration.log_uniform_sample()'s convention.
    """
    rng = np.random.default_rng(seed)
    lambda_hom = 10 ** rng.uniform(
        np.log10(HOM_RANGE[0]), np.log10(HOM_RANGE[1]), size=n_samples
    )
    lambda_div = 10 ** rng.uniform(
        np.log10(DIV_RANGE[0]), np.log10(DIV_RANGE[1]), size=n_samples
    )
    lambda_sec = 10 ** rng.uniform(
        np.log10(SEC_RANGE[0]), np.log10(SEC_RANGE[1]), size=n_samples
    )
    ratio = 10 ** rng.uniform(
        np.log10(KIN_HOM_RATIO_RANGE[0]),
        np.log10(KIN_HOM_RATIO_RANGE[1]),
        size=n_samples,
    )
    lambda_kin = derive_lambda_kin(lambda_hom, ratio)
    lambda_eff = derive_lambda_eff(lambda_kin, rng)

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
                "homeostatic": HOM_RANGE,
                "diversity": DIV_RANGE,
                "kin_hom_ratio": KIN_HOM_RATIO_RANGE,
                "eff_log_range": EFF_LOG_RANGE,
                "eff_kin_log_slope": EFF_KIN_LOG_SLOPE,
                "eff_kin_log_intercept": EFF_KIN_LOG_INTERCEPT,
                "secretion_range_placeholder": SEC_RANGE,
            },
            fp,
        )

    problem_data = pe.load_problem_data(args.sim_out_dir)
    pe.run(
        n_samples=args.n_samples,
        n_jobs=args.n_jobs,
        seed=args.seed,
        sample_fn=relationship_sample,
        **problem_data,
    )
