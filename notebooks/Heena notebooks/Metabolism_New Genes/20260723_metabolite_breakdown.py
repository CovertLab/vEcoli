"""
Phase C: per-metabolite interpretability check on the Phase B winners.

For a representative spread of Phase B candidates (chosen to span the
delta_homeo spectrum, from the strongest responders down to the mild
Toya-improving responders), solve at fraction_kinetic_target in {1.0, 0.1}
and break down *which* homeostatic metabolites' deviation-from-target grows
under knockdown. Confirms the aggregate obj_homeo response is a broad,
biologically sensible shift rather than one or two metabolites carrying the
whole delta, and cross-checks against which homeostatic metabolites are even
stoichiometrically reachable from a kinetically-constrained reaction (per
20260506_fraction_kinetic_target.ipynb cell 7: only 125/172 are).

Usage:
    python 20260723_metabolite_breakdown.py
"""

import os

import numpy as np
import plotly.express.colors as pc
import plotly.graph_objects as go
import polars as pl
from ecoli.processes.metabolism_redux_classic import FREE_RXNS, NetworkFlowModel

from pareto_exploration import load_problem_data

OUT_DIR = "notebooks/Heena notebooks/Metabolism_New Genes/pareto_results_jul_relationship_10000samples"
KNOCKDOWN_DIR = f"{OUT_DIR}/knockdown_v2"
# Representative spread across the Phase B delta_homeo spectrum:
# strongest responders (large delta_homeo, Toya fit collapses under knockdown),
# mild responders (small delta_homeo, Toya fit improves under knockdown), and
# one from the flat middle band.
CANDIDATE_INDICES = [25, 9481, 1025, 9098, 9645]
FRACTIONS = [1.0, 0.1]
TOP_N_METABOLITES = 15


def solve_with_dmdt(candidate: dict, fraction: float, problem_data: dict):
    """Like solve_one(), but keeps dm_dt instead of discarding it."""
    metabolism = problem_data["metabolism"]
    weights = {
        "homeostatic": candidate["lambda_hom"],
        "secretion": candidate["lambda_sec"],
        "efficiency": candidate["lambda_eff"],
        "kinetics": candidate["lambda_kin"],
        "diversity": candidate["lambda_div"],
    }
    model = NetworkFlowModel(
        stoich_arr=problem_data["stoichiometry"],
        metabolites=problem_data["metabolites"],
        reactions=problem_data["reaction_names"],
        homeostatic_metabolites=metabolism.homeostatic_metabolites,
        kinetic_reactions=metabolism.kinetic_constraint_reactions,
        free_reactions=FREE_RXNS,
    )
    model.set_up_exchanges(
        exchanges=metabolism.exchange_molecules,
        uptakes=metabolism.allowed_exchange_uptake,
    )
    counts_to_molar = problem_data["counts_to_molar"]
    return model.solve(
        homeostatic_concs=problem_data["homeostatic_metabolite_counts"],
        homeostatic_dm_targets=np.array(
            list(dict(problem_data["homeostatic_dm_targets"]).values())
        ),
        maintenance_target=problem_data["maintenance"],
        kinetic_targets=np.array(list(dict(problem_data["kinetic"]).values())),
        objective_weights=weights,
        upper_flux_bound=100,
        target_minimal_flux=counts_to_molar[-1],
        fraction_kinetic_target=fraction,
        include_new=True,
        new_reaction_idx=metabolism.new_reaction_idx,
    )


def connectivity(problem_data: dict) -> dict[str, bool]:
    """
    Whether each homeostatic metabolite has a nonzero stoichiometric
    coefficient in at least one kinetically-constrained reaction (same check
    as 20260506_fraction_kinetic_target.ipynb cell 7).
    """
    metabolism = problem_data["metabolism"]
    metabolite_pos = {m: i for i, m in enumerate(problem_data["metabolites"])}
    reaction_pos = {r: i for i, r in enumerate(problem_data["reaction_names"])}
    kinetic_cols = [reaction_pos[r] for r in metabolism.kinetic_constraint_reactions]
    stoich = problem_data["stoichiometry"]

    connected = {}
    for m in metabolism.homeostatic_metabolites:
        row = stoich[metabolite_pos[m], kinetic_cols]
        connected[m] = bool(np.any(row != 0))
    n_connected = sum(connected.values())
    print(
        f"{n_connected}/{len(connected)} homeostatic metabolites are "
        f"stoichiometrically reachable from a kinetic reaction."
    )
    return connected


def breakdown_candidate(
    candidate: dict, problem_data: dict, connected: dict[str, bool]
) -> pl.DataFrame:
    metabolism = problem_data["metabolism"]
    homeostatic_metabolites = list(metabolism.homeostatic_metabolites)
    metabolite_pos = {m: i for i, m in enumerate(problem_data["metabolites"])}
    homeostatic_dm_targets = dict(problem_data["homeostatic_dm_targets"])
    homeostatic_counts = dict(
        zip(homeostatic_metabolites, problem_data["homeostatic_metabolite_counts"])
    )

    deviations = {}
    for fraction in FRACTIONS:
        solution = solve_with_dmdt(candidate, fraction, problem_data)
        dm_dt = np.asarray(solution.dm_dt)
        dev = {}
        for m in homeostatic_metabolites:
            pos = metabolite_pos[m]
            dev[m] = abs(dm_dt[pos] - homeostatic_dm_targets[m]) / homeostatic_counts[m]
        deviations[fraction] = dev

    rows = []
    for m in homeostatic_metabolites:
        d1 = deviations[1.0][m]
        d01 = deviations[0.1][m]
        rows.append(
            {
                "metabolite": m,
                "connected_to_kinetic_rxn": connected[m],
                "deviation_1.0": d1,
                "deviation_0.1": d01,
                "delta_deviation": d01 - d1,
            }
        )
    df = pl.DataFrame(rows).sort("delta_deviation", descending=True)

    total_delta = df["delta_deviation"].sum()
    top_n_share = df.head(TOP_N_METABOLITES)["delta_deviation"].sum() / total_delta
    n_connected_in_movers = df.head(TOP_N_METABOLITES)["connected_to_kinetic_rxn"].sum()
    print(
        f"Index {candidate['Index']}: top {TOP_N_METABOLITES} metabolites "
        f"account for {top_n_share:.1%} of the total homeostatic-deviation "
        f"increase; {n_connected_in_movers}/{TOP_N_METABOLITES} of them are "
        f"stoichiometrically connected to a kinetic reaction."
    )
    return df.with_columns(pl.lit(candidate["Index"]).alias("Index"))


def run() -> None:
    os.chdir(os.path.expanduser("~/dev/vEcoli/"))
    os.makedirs(KNOCKDOWN_DIR, exist_ok=True)

    problem_data = load_problem_data("out/objective_weights_jul")
    connected = connectivity(problem_data)

    shortlist = pl.read_csv(f"{OUT_DIR}/best_of_best_v2.csv")
    candidates = {
        row["Index"]: row
        for row in shortlist.iter_rows(named=True)
        if row["Index"] in CANDIDATE_INDICES
    }

    all_breakdowns = []
    for idx in CANDIDATE_INDICES:
        df = breakdown_candidate(candidates[idx], problem_data, connected)
        all_breakdowns.append(df)

    combined = pl.concat(all_breakdowns)
    out_path = f"{KNOCKDOWN_DIR}/metabolite_breakdown.csv"
    combined.write_csv(out_path)
    print(f"Saved: {out_path}")

    plot_top_movers(combined)


def plot_top_movers(combined: pl.DataFrame) -> None:
    """
    Bar chart per candidate of the top-N metabolites by delta_deviation,
    colored by whether they're stoichiometrically connected to a kinetic
    reaction. Pastel palette per project convention; saved as svg with
    title/axis labels/legend.
    """
    fig = go.Figure()
    colors = {True: pc.qualitative.Pastel[2], False: pc.qualitative.Pastel[3]}
    indices = combined["Index"].unique().to_list()

    for i, idx in enumerate(indices):
        sub = (
            combined.filter(pl.col("Index") == idx)
            .sort("delta_deviation", descending=True)
            .head(TOP_N_METABOLITES)
        )
        fig.add_trace(
            go.Bar(
                x=[f"{idx}: {m}" for m in sub["metabolite"]],
                y=sub["delta_deviation"],
                name=f"Index {idx}",
                marker_color=[colors[c] for c in sub["connected_to_kinetic_rxn"]],
                showlegend=(i == 0),
                legendgroup="connected",
            )
        )

    fig.update_xaxes(title="Metabolite (grouped by candidate index)", tickangle=-60)
    fig.update_yaxes(title="Increase in target deviation (fraction=1.0 -> 0.1)")
    fig.update_layout(
        title="Phase C: which homeostatic metabolites drive the knockdown "
        "response (pastel green = stoichiometrically connected to a "
        "kinetic reaction, pastel red = not)",
        template="plotly_white",
        width=1200,
        height=650,
        barmode="group",
    )

    out_path = f"{KNOCKDOWN_DIR}/metabolite_breakdown.svg"
    fig.write_image(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    run()
