"""
Phase B: cheap single-FBA-solve knockdown re-test on the Phase A shortlist.

Re-runs each Phase A candidate (best_of_best_v2.csv) at
fraction_kinetic_target in {1.0, 0.5, 0.1} using the same fixed problem data
as the Phase 0 sweep, and reports delta_homeo / delta_kin / delta_toya_r2 so
a genuine (non-noise-floor) compensatory response can be identified before
any whole-cell sim is run.

Note: this fixes a bug in the earlier 20260506_fraction_kinetic_target.ipynb
notebook, which labeled the already-unweighted `solution.homeostatic_term`
as "Weighted Homeostatic Term" and then divided by the weight again. Here,
`obj_homeo`/`obj_kin` are the raw unweighted terms straight from
solve_one() — no re-normalization is applied.

Usage:
    uvenv notebooks/Heena\ notebooks/Metabolism_New\ Genes/20260723_knockdown_retest.py
"""

import itertools
import os

from concurrent.futures import ProcessPoolExecutor, as_completed

import plotly.express.colors as pc
import plotly.graph_objects as go
import polars as pl

from pareto_exploration import (
    correlations_toya_fluxes,
    load_problem_data,
    solve_one,
)

OUT_DIR = "notebooks/Heena notebooks/Metabolism_New Genes/pareto_results_relationship_sep_v3_10000samples"
SHORTLIST_PATH = f"{OUT_DIR}/best_of_best.csv"
KNOCKDOWN_DIR = f"{OUT_DIR}/knockdown_retested"
FRACTIONS = [1.0, 0.8, 0.5, 0.3, 0.1]

# Conservative default: CVXPY's own solve is multi-threaded (see
# pareto_exploration.py's --n_jobs docstring), and each worker rebuilds the
# full load_problem_data() state independently, so N_JOBS * (solver threads +
# per-worker memory) must fit the machine's budget -- os.cpu_count() workers
# can oversubscribe and be *slower*, or OOM. Override via the env var if you've
# checked the machine can take it.
N_JOBS = int(os.environ.get("N_JOBS", 20))

# Loaded once per worker process by _init_worker, not pickled per task.
_problem_data = None


def _init_worker() -> None:
    # Avoid each worker's BLAS/OpenMP libraries spawning their own thread
    # pools on top of N_JOBS-way process parallelism -- only takes effect if
    # the caller hasn't already set these.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    global _problem_data
    os.chdir(os.path.expanduser("~/dev/vEcoli/"))
    _problem_data = load_problem_data("out/objective_weights_jul")


def solve_and_score(
    candidate: dict, fraction: float, problem_data: dict
) -> dict | None:
    result = solve_one(
        candidate["lambda_hom"],
        candidate["lambda_sec"],
        candidate["lambda_eff"],
        candidate["lambda_kin"],
        candidate["lambda_div"],
        fraction_kinetic_target=fraction,
        **problem_data,
    )
    if result is None:
        return None

    metabolism = problem_data["metabolism"]
    base_reaction_flux = metabolism.reaction_mapping_matrix.dot(
        result.pop("solution_flux")
    )
    pearson_r2, r2, _ = correlations_toya_fluxes(
        metabolism.base_reaction_ids, base_reaction_flux
    )
    result["Index"] = candidate["Index"]
    result["toya_pearson_r_squared"] = pearson_r2
    result["toya_r_squared"] = r2
    return result


def _worker_solve(candidate: dict, fraction: float) -> dict | None:
    # Uses the module-level _problem_data set up by _init_worker in this process.
    return solve_and_score(candidate, fraction, _problem_data)


def retest() -> pl.DataFrame:
    os.chdir(os.path.expanduser("~/dev/vEcoli/"))
    os.makedirs(KNOCKDOWN_DIR, exist_ok=True)

    shortlist = pl.read_csv(SHORTLIST_PATH)
    tasks = list(itertools.product(shortlist.iter_rows(named=True), FRACTIONS))
    results_path = f"{KNOCKDOWN_DIR}/knockdown_retest_results.csv"

    rows = []
    n_done = 0
    with ProcessPoolExecutor(max_workers=N_JOBS, initializer=_init_worker) as executor:
        futures = {
            executor.submit(_worker_solve, candidate, fraction): (candidate, fraction)
            for candidate, fraction in tasks
        }
        for fut in as_completed(futures):
            candidate, fraction = futures[fut]
            try:
                row = fut.result()
            except Exception as e:
                print(
                    f"  solve raised: Index={candidate['Index']} fraction={fraction}: {e}"
                )
                continue
            if row is not None:
                rows.append(row)
            else:
                print(f"  solve failed: Index={candidate['Index']} fraction={fraction}")

            n_done += 1
            if n_done % 50 == 0 or n_done == len(tasks):
                print(f"  {n_done}/{len(tasks)} solves done")
                # Checkpoint so a later failure doesn't lose completed work.
                pl.DataFrame(rows).write_csv(results_path)

    results = pl.DataFrame(rows)
    results.write_csv(results_path)
    print(f"Saved: {results_path}")

    pivot = results.pivot(
        on="fraction_kinetic_target",
        index=[
            "Index",
            "lambda_hom",
            "lambda_kin",
            "lambda_eff",
            "lambda_sec",
            "lambda_div",
        ],
        values=["obj_homeo", "obj_kin", "toya_r_squared"],
    )
    pivot = pivot.with_columns(
        (pl.col("obj_homeo_0.1") - pl.col("obj_homeo_1.0")).alias("delta_homeo"),
        (pl.col("obj_kin_0.1") - pl.col("obj_kin_1.0")).alias("delta_kin"),
        (pl.col("toya_r_squared_0.1") - pl.col("toya_r_squared_1.0")).alias(
            "delta_toya_r2"
        ),
    ).sort("delta_homeo", descending=True)

    summary_path = f"{KNOCKDOWN_DIR}/knockdown_delta_summary.csv"
    pivot.write_csv(summary_path)
    print(f"Saved: {summary_path}")
    print(
        pivot.select(
            [
                "Index",
                "lambda_hom",
                "lambda_kin",
                "obj_homeo_1.0",
                "obj_homeo_0.1",
                "delta_homeo",
                "delta_kin",
                "toya_r_squared_1.0",
                "toya_r_squared_0.1",
                "delta_toya_r2",
            ]
        )
    )

    plot_knockdown_response(results)
    return pivot


def plot_knockdown_response(results: pl.DataFrame) -> None:
    """
    One line per candidate: obj_homeo (unweighted) vs fraction_kinetic_target.
    A genuine compensatory response shows obj_homeo rising as fraction drops
    from 1.0 -> 0.1; a flat line means that candidate is still noise-floor
    pinned. Pastel palette per project convention; saved as svg with
    title/axis labels/legend.
    """
    fig = go.Figure()
    palette = pc.qualitative.Pastel
    indices = sorted(results["Index"].unique().to_list())

    for i, idx in enumerate(indices):
        sub = results.filter(pl.col("Index") == idx).sort("fraction_kinetic_target")
        fig.add_trace(
            go.Scatter(
                x=sub["fraction_kinetic_target"],
                y=sub["obj_homeo"],
                mode="lines+markers",
                name=f"Index {idx}",
                line=dict(color=palette[i % len(palette)]),
            )
        )

    fig.update_xaxes(
        title="fraction_kinetic_target (1.0 = no knockdown, 0.1 = strong knockdown)",
        autorange="reversed",
    )
    fig.update_yaxes(type="log", title="Homeostatic Objective (obj_homeo, unweighted)")
    fig.update_layout(
        title="Phase B: homeostatic-objective response to kinetic-target knockdown",
        legend_title="Weight-combo candidate",
        template="plotly_white",
        width=950,
        height=650,
    )

    out_path = f"{KNOCKDOWN_DIR}/knockdown_response.svg"
    fig.write_image(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    retest()
