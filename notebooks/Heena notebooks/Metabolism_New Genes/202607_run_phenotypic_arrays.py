"""Run standalone FBA across the ~383 EcoCyc phenotype-microarray (PM) conditions.

`load_sim` and `test_NetworkFlowModel` are ported from
`Standalone_FBA.ipynb` (cells 1 and 3). `test_NetworkFlowModel` is lightly
extended to also return the `FlowResult` and `NetworkFlowModel` instance so
this script can pull `homeostatic_objective`/`homeostatic_dmdt` out of the
solve internals; the FBA math itself is untouched.

Each PM well swaps a carbon/nitrogen/phosphorus/sulfur source in for the
basal medium (see `build_conditions`) and is solved independently, so the
loop over wells is parallelized with `joblib.Parallel`.
"""

import argparse
import json
import os
import re
from pathlib import Path

import cvxpy as cp
import dill
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ecoli.processes.metabolism_redux_classic import FlowResult, NetworkFlowModel

REPO_ROOT = Path(__file__).resolve().parents[3]

# Mirrors the example weights in Standalone_FBA.ipynb cell 6 (no per-plate
# tuning has been done yet for the phenotype-array sweep).
DEFAULT_OBJECTIVE_WEIGHTS = {
    "secretion": 0.01,
    "efficiency": 1e-06,
    "kinetics": 1e-05,
    "diversity": 1e-07,
    "homeostatic": 1,
}

CARBON_REMOVE = {"GLC[p]", "CA+2[p]"}
NITROGEN_REMOVE = {"AMMONIUM[c]", "CA+2[p]"}
PHOSPHORUS_REMOVE = {"Pi[p]", "CA+2[p]"}
SULFUR_REMOVE = {"SULFATE[p]", "CA+2[p]"}

WELL_RE = re.compile(r"^([A-Ha-h])0*(\d{1,2})$")


def load_sim(
    folder_path: str,
):
    """This function is meant to load an output of a simulation in timeseries form.
    Note: This is not designed for parquet output format.
    """
    # --- Load Sim ---
    output = np.load(folder_path + "0_output.npy", allow_pickle="TRUE").item()
    output = output["agents"]["0"]
    fba = output["listeners"]["fba_results"]
    bulk = pd.DataFrame(output["bulk"])
    f = open(folder_path + "agent_steps.pkl", "rb")
    agent = dill.load(f)
    f.close()

    metabolism = agent["ecoli-metabolism-redux-classic"]

    return fba, bulk, metabolism, output


def test_NetworkFlowModel(
    objective_weights,
    metabolism,
    fba,
    uptake_addition=set([]),
    uptake_removal=set([]),
    new_exchange_molecules=set([]),
    add_metabolite=None,
    add_reaction=None,
    remove_reaction=None,
    add_kinetic=None,
    force_reaction=None,
    add_homeostatic_demand=None,
    solver_choice=cp.GLOP,
):
    # --- Update Exchanges ---
    # Note: Exchange molecules are the molecules that can be secreted
    #       Uptake molecules are the molecules that can be taken up.
    uptake = metabolism.allowed_exchange_uptake.copy()
    uptake = set(uptake)
    uptake = uptake | uptake_addition
    uptake = uptake - uptake_removal

    exchange_molecules = metabolism.exchange_molecules.copy()
    exchange_molecules = exchange_molecules | new_exchange_molecules

    # --- Get whole-cell model outputs for FBA ---
    reaction_names = metabolism.reaction_names.copy()
    metabolites = metabolism.metabolite_names.copy()

    kinetic_reaction_ids = metabolism.kinetic_constraint_reactions.copy()
    kinetic = (
        pd.DataFrame(
            fba["target_kinetic_fluxes"],
            columns=metabolism.kinetic_constraint_reactions,
        )
        .mean(axis=0)
        .copy()
    )  # units in counts/s
    homeostatic = (
        pd.DataFrame(
            fba["target_homeostatic_dmdt"], columns=metabolism.homeostatic_metabolites
        )
        .mean(axis=0)
        .copy()
    )  # units in counts/s
    homeostatic_counts = (
        pd.DataFrame(
            fba["homeostatic_metabolite_counts"],
            columns=metabolism.homeostatic_metabolites,
        )
        .mean(axis=0)
        .copy()
    )  # units in counts
    maintenance = (
        pd.DataFrame(fba["maintenance_target"][1:], columns=["maintenance_reaction"])
        .mean(axis=0)
        .copy()
    )

    S_new = metabolism.stoichiometry.copy()

    # --- Add/remove reactions, metabolites, kinetic constraints, homeostatic demands ---
    if add_metabolite is not None:
        for m in add_metabolite:
            if m not in metabolites:
                metabolites.append(m)
        # append rows of zeros to S_new of length add_metabolite
        S_new = np.concatenate(
            (S_new, np.zeros((len(add_metabolite), S_new.shape[1]))), axis=0
        )

    if add_reaction is not None:
        # Note: if one wants to add a reaction, it should be in a dictionary of
        #       form {"reaction_name": {"metabolite1": stoich1, "metabolite2": stoich2, ...}}
        assert isinstance(add_reaction, dict)

        for r, s in add_reaction.items():
            assert isinstance(s, dict)
            if r not in reaction_names:
                reaction_names.append(r)
            # append columns of reaction stoich to S_new of length add_reaction
            new_reaction = np.zeros((S_new.shape[0], 1))
            for m, v in s.items():
                new_reaction[metabolites.index(m), 0] = v
            S_new = np.concatenate((S_new, new_reaction), axis=1)

    if add_kinetic is not None:
        # Note: if one wants to add kinetics, it should be in a dictionary of
        #       form {"reaction_name": kinetic_target}
        assert isinstance(add_kinetic, dict)

        for r, v in add_kinetic.items():
            if r not in kinetic_reaction_ids:
                kinetic_reaction_ids.append(r)
                kinetic[r] = v
            if r in kinetic_reaction_ids:
                kinetic[r] = v

    if remove_reaction is not None:
        for r in remove_reaction:
            r_idx = reaction_names.index(r)
            S_new = np.delete(S_new, r_idx, axis=1)
            reaction_names.remove(r)
            if r in kinetic_reaction_ids:
                kinetic_reaction_ids.remove(r)
                del kinetic[r]

    if force_reaction is not None:
        force_reaction_idx = np.array([reaction_names.index(r) for r in force_reaction])
    else:
        force_reaction_idx = force_reaction

    if add_homeostatic_demand is not None:
        # assert add_homeostatic_demand is a list
        assert isinstance(add_homeostatic_demand, list)

        for met in add_homeostatic_demand:
            # here I just arbitrarily set the homeostatic demand to be 100 counts/s,
            # and homeostatic metabolite count to be 1, but this can be changed as needed.
            homeostatic[met] = 100
            homeostatic_counts[met] = 1

    # --- Convert every count-based FBA input to concentration ---
    # Everything above (homeostatic/homeostatic_counts/kinetic/maintenance) is in
    # raw molecule counts (or counts/s), straight from the fba listener. The live
    # whole-cell sim solves NetworkFlowModel in concentration units with
    # upper_flux_bound=100 (the class default); this used to instead leave
    # everything in raw counts and inflate upper_flux_bound to 10^9 to
    # compensate, which (per debugging -- see match_compounds.py's docstring
    # and conversation history) both inflated absolute magnitudes unpredictably
    # and let a few "hub" metabolites with huge raw dm/dt swamp the homeostatic
    # objective regardless of nutrient condition, making results insensitive to
    # which well was actually being tested. Multiplying every count-based
    # quantity by the same counts_to_molar factor keeps the LP fully
    # concentration-scale and consistent with the live sim's own units
    # (mirrors the same fix applied to Standalone_FBA.ipynb).
    counts_to_molar = metabolism.counts_to_molar.asNumber()
    homeostatic_concs_conc = homeostatic_counts.values * counts_to_molar
    homeostatic_dm_targets_conc = (
        np.array(list(dict(homeostatic).values())) * counts_to_molar
    )
    kinetic_targets_conc = np.array(list(dict(kinetic).values())) * counts_to_molar
    maintenance_conc = maintenance * counts_to_molar

    # --- Solve NetworkFlowModel ---
    model = NetworkFlowModel(
        stoich_arr=S_new,
        metabolites=metabolites,
        reactions=reaction_names,
        homeostatic_metabolites=metabolism.homeostatic_metabolites,
        kinetic_reactions=kinetic_reaction_ids,
    )
    model.set_up_exchanges(exchanges=exchange_molecules, uptakes=uptake)
    solution: FlowResult = model.solve(
        homeostatic_concs=homeostatic_concs_conc,
        homeostatic_dm_targets=homeostatic_dm_targets_conc,
        maintenance_target=maintenance_conc,
        kinetic_targets=kinetic_targets_conc,
        binary_kinetic_idx=None,
        force_flow_idx=force_reaction_idx,
        objective_weights=objective_weights,
        upper_flux_bound=100,  # matches the live WC sim's scale now everything is in concentration units.
        solver=solver_choice,
    )
    # `solution` and `model` are appended (beyond the notebook's original 6-tuple)
    # so callers can read FlowResult fields (e.g. homeostatic_term, dm_dt) and
    # NetworkFlowModel.homeostatic_idx without re-solving.
    return (
        solution.objective,
        solution.velocities,
        reaction_names,
        S_new,
        metabolites,
        kinetic,
        solution,
        model,
    )


def parse_well(well):
    """Split a well label like "A2" or "A02" into ("A", "2")."""
    match = WELL_RE.match(str(well).strip())
    if match is None:
        raise ValueError(f"Could not parse well label: {well!r}")
    row, col = match.groups()
    return row.upper(), str(int(col))


def normalize_plate(plate):
    return re.sub(r"[^A-Z0-9]", "", str(plate).upper())


def default_removals(plate_key, well_row):
    if plate_key in ("PM1", "PM2"):
        return set(CARBON_REMOVE)
    if plate_key == "PM3":
        return set(NITROGEN_REMOVE)
    if plate_key == "PM4":
        if well_row in "ABCDE":
            return set(PHOSPHORUS_REMOVE)
        if well_row in "FGH":
            return set(SULFUR_REMOVE)
    return set()


def build_conditions(mapping_csv_path):
    """Read compound_mapping.csv into a dict of well -> {Add, Remove, ...}.

    Rows are skipped only when the compound could not be resolved to a
    species at all (match_status != "matched" and matched_species_id is
    empty). Legitimate negative-control wells (match_status == "matched"
    with an empty matched_species_id) are kept with an empty Add set.
    """
    df = pd.read_csv(mapping_csv_path, dtype=str).fillna("")
    conditions = {}
    for _, row in df.iterrows():
        plate = row["plate"].strip()
        well = row["well"].strip()
        match_status = row["match_status"].strip().lower()
        species_id = row["matched_species_id"].strip()
        compartment = row["compartment"].strip()

        if match_status != "matched" and not species_id:
            continue

        well_row, well_col = parse_well(well)
        plate_key = normalize_plate(plate)

        add_set = set()
        if species_id:
            # Usually a single id (either already "ID[e]"-tagged, or a bare
            # id paired with the `compartment` column). One well (PM1 C3,
            # DL-Malic acid) legitimately has two species pipe-joined, both
            # already compartment-tagged, e.g. "CPD-660[e]|MAL[e]" -- split
            # on "|" and treat each piece independently rather than as one
            # bogus combined literal.
            for piece in species_id.split("|"):
                piece = piece.strip()
                if not piece:
                    continue
                if piece.endswith("]"):
                    add_set.add(piece)
                else:
                    comp = compartment or "c"
                    add_set.add(f"{piece}[{comp}]")

        conditions[f"{plate}_{well}"] = {
            "plate": plate,
            "plate_key": plate_key,
            "well": well,
            "well_row": well_row,
            "well_col": well_col,
            "compound_name": row.get("compound_name", ""),
            "mix0_id": row.get("mix0_id", ""),
            "species_id": species_id,
            "Add": add_set,
            "Remove": default_removals(plate_key, well_row),
        }
    return conditions


def load_growth_calls(wells_json_path):
    """Load aerobic/anaerobic growth calls keyed by (plate_key, well_row, well_col).

    Returns an empty lookup (rather than raising) if the file does not yet
    exist, since it is produced by a separate, later script.
    """
    path = Path(wells_json_path)
    if not path.exists():
        return {}

    with open(path) as f:
        data = json.load(f)

    records = data if isinstance(data, list) else data.get("wells", [])
    lookup = {}
    for rec in records:
        plate_key = normalize_plate(rec.get("plate", ""))
        well_row = str(rec.get("well_row", "")).strip().upper()
        well_col_raw = str(rec.get("well_col", "")).strip()
        try:
            well_col = str(int(well_col_raw))
        except ValueError:
            well_col = well_col_raw
        lookup[(plate_key, well_row, well_col)] = {
            "aerobic_growth_call": rec.get("aerobic_growth_call"),
            "anaerobic_growth_call": rec.get("anaerobic_growth_call"),
        }
    return lookup


def run_condition(cond, metabolism, fba, objective_weights, growth_calls):
    # Some wells' Add species (see match_compounds.py's "confirmed_absent" /
    # add_metabolite rows in compound_mapping.csv) don't exist as a row in
    # this model's stoichiometric matrix at all -- metabolism.metabolite_names
    # is the full universe (see match_compounds.py's module docstring), so
    # anything not in there has no S-matrix row and set_up_exchanges would
    # raise KeyError on it (met_map lookup) if just passed as
    # uptake_addition/new_exchange_molecules alone. test_NetworkFlowModel's
    # add_metabolite param appends a new all-zero row for exactly this case
    # before the model is built, which is what actually makes the well
    # runnable instead of erroring.
    novel_metabolites = [m for m in cond["Add"] if m not in metabolism.metabolite_names]
    add_metabolite = novel_metabolites or None

    try:
        result = test_NetworkFlowModel(
            objective_weights,
            metabolism=metabolism,
            fba=fba,
            uptake_addition=cond["Add"],
            uptake_removal=cond["Remove"],
            # A species newly Add-ed for a PM well usually isn't already in
            # this checkpoint's metabolism.exchange_molecules (that set only
            # reflects whatever media the checkpoint's own simulation history
            # happened to use, see match_compounds.py's module docstring) --
            # without also listing it here, set_up_exchanges never builds an
            # exchange reaction for it at all and uptake_addition is a no-op
            # (verified: omitting this made every well return ~the same oofv
            # regardless of condition).
            new_exchange_molecules=cond["Add"],
            add_metabolite=add_metabolite,
        )
        oofv, _, _, _, _, _, solution, model = result

        # homeostatic_objective: the unweighted FlowResult.homeostatic_term, i.e.
        # cp.norm1((dm[homeostatic_idx] - homeostatic_dm_targets) / homeostatic_concs)
        # from NetworkFlowModel.solve (metabolism_redux_classic.py, ~line 815-822),
        # before it is scaled by objective_weights["homeostatic"].
        homeostatic_objective = float(solution.homeostatic_term)

        # homeostatic_dmdt: sum of |dm/dt| restricted to the homeostatic metabolites
        # (FlowResult.dm_dt is the full per-metabolite mass-balance vector; indexing
        # by NetworkFlowModel.homeostatic_idx, ~line 736, gives the same subset used
        # for homeostatic_objective, but unnormalized and not L1-collapsed against
        # the target -- this is the raw counts/s flow into/out of the demand pool).
        homeostatic_dmdt = np.sum(np.asarray(solution.dm_dt)[model.homeostatic_idx])

        status = "feasible"
    except ValueError:
        oofv = None
        homeostatic_objective = None
        homeostatic_dmdt = None
        status = "infeasible"

    growth = growth_calls.get(
        (cond["plate_key"], cond["well_row"], cond["well_col"]), {}
    )

    return {
        "plate": cond["plate"],
        "well": cond["well"],
        "compound_name": cond["compound_name"],
        "mix0_id": cond["mix0_id"],
        "species_id": cond["species_id"],
        "oofv": oofv,
        "homeostatic_objective": homeostatic_objective,
        "homeostatic_dmdt": homeostatic_dmdt,
        "status": status,
        "aerobic_growth_call": growth.get("aerobic_growth_call"),
        "anaerobic_growth_call": growth.get("anaerobic_growth_call"),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run standalone FBA across EcoCyc phenotype-microarray conditions."
    )
    parser.add_argument(
        "--sim-folder",
        default="out/phenotypic/basal_min_weights_2500_2026-07-06/",
    )
    parser.add_argument(
        "--mapping-csv",
        default="notebooks/Heena notebooks/Metabolism_New Genes/compound_mapping.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="notebooks/Heena notebooks/Metabolism_New Genes/out/phenotypic_arrays/",
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--wells-json",
        default="notebooks/Heena notebooks/Metabolism_New Genes/phenotypic_array_wells.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(REPO_ROOT)

    sim_folder = (
        args.sim_folder if args.sim_folder.endswith("/") else args.sim_folder + "/"
    )
    fba, bulk, metabolism, output = load_sim(sim_folder)

    conditions = build_conditions(args.mapping_csv)
    growth_calls = load_growth_calls(args.wells_json)

    # Wells are independent FBA solves. Each call builds its own CVXPY Problem
    # (fresh NetworkFlowModel) inside test_NetworkFlowModel, so there is no
    # shared mutable state across wells. Threads (not processes) are used
    # because `metabolism`/`fba` are complex simulation objects (Vivarium
    # Step + raw listener output) that are not guaranteed picklable for
    # joblib's default loky/process backend; the CVXPY/GLOP solve itself
    # releases the GIL for its C++ core, so threading still parallelizes the
    # actual solver work.
    results = Parallel(n_jobs=args.n_jobs, prefer="threads")(
        delayed(run_condition)(
            cond, metabolism, fba, DEFAULT_OBJECTIVE_WEIGHTS, growth_calls
        )
        for cond in conditions.values()
    )

    # Minimal media (basal) reference: no Add/Remove at all, i.e. the
    # checkpoint's own unmodified media. Included as its own labeled row
    # (plate="BASAL") so 202607_plot_phenotypic_results.py can read it
    # directly as a real histogram entry / dotted reference line, instead of
    # that script having to reload this same ~300MB checkpoint and re-run
    # this exact solve itself just for one point.
    basal_cond = {
        "plate": "BASAL",
        "plate_key": "BASAL",
        "well": "minimal_media",
        "well_row": "",
        "well_col": "",
        "compound_name": "Minimal media (basal, no Add/Remove)",
        "mix0_id": "",
        "species_id": "",
        "Add": set(),
        "Remove": set(),
    }
    results.append(
        run_condition(
            basal_cond, metabolism, fba, DEFAULT_OBJECTIVE_WEIGHTS, growth_calls
        )
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Wrote {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
