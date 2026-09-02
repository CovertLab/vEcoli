"""
Relationship-constrained twin of pareto_exploration.py.

Instead of sampling all five lambda weights independently in log space,
anchors on `lambda_kin` and draws three ratios-to-kin independently, then
derives `lambda_hom`, `lambda_eff`, and `lambda_sec` from them:

- log10(lambda_kin) drawn uniform in KIN_RANGE.
- ratio_hom_kin = log10(lambda_hom) - log10(lambda_kin), drawn uniform in
  HOM_KIN_RATIO_RANGE; derives lambda_hom.
- ratio_eff_kin = log10(lambda_eff) - log10(lambda_kin), drawn uniform in
  EFF_KIN_RATIO_RANGE; derives lambda_eff.
- ratio_sec_kin = log10(lambda_sec) - log10(lambda_kin), drawn uniform in
  SEC_KIN_RATIO_RANGE; derives lambda_sec.

ratio_hom_eff and ratio_hom_sec are NOT independent of the above — they are
exactly ratio_hom_kin - ratio_eff_kin and ratio_hom_kin - ratio_sec_kin
respectively (differences of the same 4 logs). To still land in a target
window for those two derived ratios (found via the joint obj_homeo/toya_r²
constraint analysis in 20260804_pareto_conditional_analysis.ipynb), each
candidate draw is rejection-sampled: reject and redraw unless the derived
ratio_hom_eff falls in HOM_EFF_TARGET_RANGE and ratio_hom_sec falls in
HOM_SEC_TARGET_RANGE. This is cheap since rejection happens before the
expensive CVXPY solve in solve_one() -- only accepted candidates ever reach
run()'s solve loop.

lambda_div is drawn independently (no known relationship), unchanged from
prior versions of this script.

This reuses pareto_exploration.py's run()/solve_one()/load_problem_data()/
plotting functions as-is (they don't care how weight samples were
generated) rather than duplicating them -- only the sampling function
differs. See run()'s `sample_fn` parameter in pareto_exploration.py.

Usage:
    uvenv pareto_exploration_relationship.py
    uvenv pareto_exploration_relationship.py --n_samples 500 --n_jobs 4
"""

import argparse
import json
import os

import numpy as np

import pareto_exploration as pe

# ---------------------------------------------------------------------------
# Independent-draw ranges. KIN_RANGE and the three *_KIN_RATIO_RANGE bounds
# are already log10-space quantities, so they're drawn linear-uniform (no
# extra log-transform). DIV_RANGE is in linear lambda-space, same as
# pareto_exploration.WEIGHT_RANGES, and drawn log-uniform as before.
# ---------------------------------------------------------------------------
KIN_RANGE = (-4.0, -2.0)  # log10(lambda_kin)
HOM_KIN_RATIO_RANGE = (2.77, 845)  # log10(lambda_hom) - log10(lambda_kin)
EFF_KIN_RATIO_RANGE = (-0.8, 0.4)  # log10(lambda_eff) - log10(lambda_kin)
SEC_KIN_RATIO_RANGE = (-0.2, 1)  # log10(lambda_sec) - log10(lambda_kin)
DIV_RANGE = (1e-5, 1e-2)

# Rejection windows for the two ratios that are *derived* from the three
# ratios-to-kin above (ratio_hom_eff = ratio_hom_kin - ratio_eff_kin;
# ratio_hom_sec = ratio_hom_kin - ratio_sec_kin) -- not independently
# drawable, so enforced by rejecting candidates that land outside these.
# These ranges (and the three above) were derived from the knockdown-
# responsiveness analysis in pareto_all/ (see 20260807 conversation):
# candidates with ratio_hom_kin in [2.81,2.91] and ratio_hom_sec in
# [2.7,2.76] showed a genuine, reproducible orders-of-magnitude obj_homeo
# response to kinetic knockdown, cross-validated against the raw
# ratio_pairwise_analysis_obj_home.html plots for the jul and v6 datasets.
HOM_EFF_TARGET_RANGE = (4.6, 6.0)
HOM_SEC_TARGET_RANGE = (2.85, 5.5)

OUT_DIR = "notebooks/Heena notebooks/Metabolism_New Genes/pareto_results_relationship_sep_v1_10000samples"
pe.OUT_DIR = OUT_DIR  # redirect run()'s output (CSV + 4 plots) to this directory


def relationship_sample(n_samples: int, seed: int = 42) -> np.ndarray:
    """
    Draw log10(lambda_kin) and three ratios-to-kin independently (log_lambda_kin,
    ratio_hom_kin, ratio_eff_kin, ratio_sec_kin), derive lambda_hom/lambda_eff/
    lambda_sec/lambda_kin from them, and draw lambda_div independently.

    Rejection-samples so that the two *derived* ratios ratio_hom_eff and
    ratio_hom_sec also land within HOM_EFF_TARGET_RANGE / HOM_SEC_TARGET_RANGE
    (see module docstring -- these aren't independently controllable).

    Returns array of shape (n_samples, 5) with columns ordered
    [homeostatic, secretion, efficiency, kinetics, diversity], matching
    pareto_exploration.log_uniform_sample()'s convention.
    """
    rng = np.random.default_rng(seed)

    accepted_log_kin = []
    accepted_ratio_hom_kin = []
    accepted_ratio_eff_kin = []
    accepted_ratio_sec_kin = []

    n_remaining = n_samples
    while n_remaining > 0:
        batch_size = max(
            n_remaining * 5, 100
        )  # ~50% acceptance rate observed, 5x is generous headroom
        log_kin = rng.uniform(KIN_RANGE[0], KIN_RANGE[1], size=batch_size)
        ratio_hom_kin = rng.uniform(
            HOM_KIN_RATIO_RANGE[0], HOM_KIN_RATIO_RANGE[1], size=batch_size
        )
        ratio_eff_kin = rng.uniform(
            EFF_KIN_RATIO_RANGE[0], EFF_KIN_RATIO_RANGE[1], size=batch_size
        )
        ratio_sec_kin = rng.uniform(
            SEC_KIN_RATIO_RANGE[0], SEC_KIN_RATIO_RANGE[1], size=batch_size
        )

        ratio_hom_eff = ratio_hom_kin - ratio_eff_kin
        ratio_hom_sec = ratio_hom_kin - ratio_sec_kin
        accept = (
            (ratio_hom_eff >= HOM_EFF_TARGET_RANGE[0])
            & (ratio_hom_eff <= HOM_EFF_TARGET_RANGE[1])
            & (ratio_hom_sec >= HOM_SEC_TARGET_RANGE[0])
            & (ratio_hom_sec <= HOM_SEC_TARGET_RANGE[1])
        )

        n_take = min(int(accept.sum()), n_remaining)
        idx = np.flatnonzero(accept)[:n_take]
        accepted_log_kin.append(log_kin[idx])
        accepted_ratio_hom_kin.append(ratio_hom_kin[idx])
        accepted_ratio_eff_kin.append(ratio_eff_kin[idx])
        accepted_ratio_sec_kin.append(ratio_sec_kin[idx])
        n_remaining -= n_take

    log_kin = np.concatenate(accepted_log_kin)
    ratio_hom_kin = np.concatenate(accepted_ratio_hom_kin)
    ratio_eff_kin = np.concatenate(accepted_ratio_eff_kin)
    ratio_sec_kin = np.concatenate(accepted_ratio_sec_kin)

    lambda_kin = 10**log_kin
    lambda_hom = 10 ** (log_kin + ratio_hom_kin)
    lambda_eff = 10 ** (log_kin + ratio_eff_kin)
    lambda_sec = 10 ** (log_kin + ratio_sec_kin)
    lambda_div = 10 ** rng.uniform(
        np.log10(DIV_RANGE[0]), np.log10(DIV_RANGE[1]), size=n_samples
    )

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
                "kin_range": KIN_RANGE,
                "hom_kin_ratio_range": HOM_KIN_RATIO_RANGE,
                "eff_kin_ratio_range": EFF_KIN_RATIO_RANGE,
                "sec_kin_ratio_range": SEC_KIN_RATIO_RANGE,
                "hom_eff_target_range": HOM_EFF_TARGET_RANGE,
                "hom_sec_target_range": HOM_SEC_TARGET_RANGE,
                "diversity": DIV_RANGE,
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
