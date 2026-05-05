"""Biomass-max LP driver. Loads pickled state, runs LPs with config knobs.

Pair with the pickle cell in standalone_fba_prodenvs.ipynb that writes
out/biomass_reaction/state.pkl. Re-pickle from the notebook whenever
the upstream extraction or model build changes.

Usage:
    uv run python notebooks/metabolic_engineering/_drivers/solve_biomass.py \\
        --source estimated --solver HIGHS

Returns status, v_biomass_user, objective, and the top-N
mass-balance duals (sorted by |dual|).
"""

import argparse
import json
import sys
from pathlib import Path

import cvxpy as cp
import dill
import numpy as np


def load_state(path):
    with open(path, "rb") as f:
        return dill.load(f)


def build_problem(
    state,
    source="estimated",
    drop_mets=None,
    kin_upper=True,
    drop_secretion=True,
    kinetic_weight=0.0,
    kinetics_in_range=1e-2,
    upper_flux_bound=1e7,
    glucose_multiplier=None,
    efficiency_weight=None,
    objective="biomass",
    fix_biomass_user=None,
):
    """Build the biomass-max CVXPY problem with config knobs.

    drop_mets: iterable of metabolite IDs to skip in dm[homeo]==0.
    kin_upper: include hard v[k] <= kinetic_targets[:, 2] (old Mode C).
    drop_secretion: omit secretion penalty term (canonical Mode B/C).
    kinetic_weight: soft kinetic loss weight w_kin. Mirrors
        metabolism_redux.py:1041-1071 — piecewise norm1 with a hard-bounded
        in-band variable (free zone between kinetic lower/upper) and an
        unbounded outside-band variable (heavy penalty). Set 0 for Mode B.
        Use with kin_upper=False for the new soft-kinetic Mode C.
    kinetics_in_range: in-band sub-weight (default 1e-2 per Standalone_FBA
        canonical). Effective in-band weight = kinetic_weight * kinetics_in_range.
    """
    if source == "target":
        model = state["model_t"]
        b_idx = state["biomass_idx_t"]
        b_scale = state["biomass_scale_t"]
    elif source == "estimated":
        model = state["model_e"]
        b_idx = state["biomass_idx_e"]
        b_scale = state["biomass_scale_e"]
    else:
        raise ValueError("source must be 'target' or 'estimated'")

    metabolites = state["metabolites"]
    homeo_idx = np.asarray(model.homeostatic_idx)
    drop_mets = set(drop_mets or [])
    keep_mask = np.array([metabolites[j] not in drop_mets for j in homeo_idx])
    kept_idx = homeo_idx[keep_mask]

    v = cp.Variable(model.n_orig_rxns)
    e = cp.Variable(model.n_exch_rxns)
    dm = model.S_orig @ v + model.S_exch @ e

    total_maintenance = state["maintenance"] + model.gam * e @ model.exchange_masses

    constr = [
        dm[model.intermediates_idx] == 0,
        dm[kept_idx] == 0,
        v >= 0,
        v <= upper_flux_bound,
        e >= 0,
        e <= upper_flux_bound,
    ]
    if model.maintenance_idx is not None:
        constr.append(v[model.maintenance_idx] == total_maintenance)
        constr.append(v[model.maintenance_idx] >= state["maintenance"])
    if kin_upper:
        constr.append(v[model.kinetic_rxn_idx] <= state["kinetic_targets_arr"][:, 2])

    # Soft kinetic loss decomposition (mirrors metabolism_redux.py:1041-1071).
    # Built here so the constraints land in `constr`; loss term added below.
    soft_kin = None
    if kinetic_weight > 0:
        kin_idx = model.kinetic_rxn_idx
        kin_targets = state["kinetic_targets_arr"]
        target_central = kin_targets[:, 1]
        nonzero_target = target_central.copy()
        nonzero_target[nonzero_target == 0] = 1.0
        lower_diff = kin_targets[:, 0] - target_central
        upper_diff = kin_targets[:, 2] - target_central
        n_kin = len(kin_idx)
        v_diff_in = cp.Variable(n_kin)
        v_diff_out = cp.Variable(n_kin)
        constr.append(v[kin_idx] == target_central + v_diff_in + v_diff_out)
        constr.append(v_diff_in >= lower_diff)
        constr.append(v_diff_in <= upper_diff)
        soft_kin = (v_diff_in, v_diff_out, nonzero_target)

    if glucose_multiplier is not None:
        glc_idx = state["glucose_exch_idx"]
        glc_bound = glucose_multiplier * state["glucose_bound_counts_s_default"]
        constr.append(e[glc_idx] <= glc_bound)

    if fix_biomass_user is not None:
        constr.append(v[b_idx] == fix_biomass_user * b_scale)

    if objective == "biomass":
        loss = -v[b_idx] / b_scale
        product_e_idx = None
    elif objective.startswith("secrete:"):
        exch_id = objective[len("secrete:") :]
        try:
            product_e_idx = state["exchange_metabolites"].index(exch_id)
        except ValueError:
            raise ValueError(
                f"Exchange '{exch_id}' not found. Available secretion entries: "
                + ", ".join(x for x in state["exchange_metabolites"] if "rev" in x)[
                    :500
                ]
            )
        loss = -e[product_e_idx]
    else:
        raise ValueError(
            f"objective must be 'biomass' or 'secrete:<id>', got {objective!r}"
        )

    if efficiency_weight is not None and efficiency_weight > 0:
        # pFBA-style ℓ1 penalty on internal fluxes; breaks degeneracy by
        # picking the min-flux optimum. Exclude biomass column itself
        # (v ≥ 0 → norm1(v) = sum(v), so subtract v[b_idx]).
        loss = loss + efficiency_weight * (cp.sum(v) - v[b_idx])
    if not drop_secretion:
        weights = state.get("objective_weights", {})
        if "secretion" in weights:
            loss = loss + weights["secretion"] * cp.sum(
                e[model.secretion_idx] @ -model.exchange_masses[model.secretion_idx]
            )
    if soft_kin is not None:
        v_diff_in, v_diff_out, nonzero_target = soft_kin
        loss = loss + kinetic_weight * (
            cp.norm1(v_diff_out / nonzero_target)
            + kinetics_in_range * cp.norm1(v_diff_in / nonzero_target)
        )

    p = cp.Problem(cp.Minimize(loss), constr)
    return p, {
        "v": v,
        "e": e,
        "b_idx": b_idx,
        "b_scale": b_scale,
        "kept_idx": kept_idx,
        "mass_bal_constr": constr[1],
        "product_e_idx": product_e_idx,
    }


def solve(
    state,
    source="estimated",
    drop_mets=None,
    kin_upper=True,
    drop_secretion=True,
    kinetic_weight=0.0,
    kinetics_in_range=1e-2,
    solver="HIGHS",
    upper_flux_bound=1e7,
    top_n_duals=20,
    glucose_multiplier=None,
    efficiency_weight=None,
    objective="biomass",
    fix_biomass_user=None,
):
    p, h = build_problem(
        state,
        source=source,
        drop_mets=drop_mets,
        kin_upper=kin_upper,
        drop_secretion=drop_secretion,
        kinetic_weight=kinetic_weight,
        kinetics_in_range=kinetics_in_range,
        upper_flux_bound=upper_flux_bound,
        glucose_multiplier=glucose_multiplier,
        efficiency_weight=efficiency_weight,
        objective=objective,
        fix_biomass_user=fix_biomass_user,
    )

    cp_solver = getattr(cp, solver)
    try:
        p.solve(solver=cp_solver)
    except cp.error.SolverError as exc:
        return {
            "solver": solver,
            "status": "SolverError",
            "error": str(exc),
        }

    metabolites = state["metabolites"]
    product_flux = None
    if h["product_e_idx"] is not None and h["e"].value is not None:
        product_flux = float(h["e"].value[h["product_e_idx"]])
    out = {
        "solver": solver,
        "status": p.status,
        "objective": float(p.value) if p.value is not None else None,
        "v_biomass_user": (
            float(h["v"].value[h["b_idx"]]) / h["b_scale"]
            if h["v"].value is not None
            else None
        ),
        "product_flux_counts_s": product_flux,
        "config": {
            "source": source,
            "drop_mets_count": len(drop_mets or []),
            "kin_upper": kin_upper,
            "drop_secretion": drop_secretion,
            "kinetic_weight": kinetic_weight,
            "kinetics_in_range": kinetics_in_range,
            "upper_flux_bound": upper_flux_bound,
            "glucose_multiplier": glucose_multiplier,
        },
    }

    duals = h["mass_bal_constr"].dual_value
    if duals is not None and top_n_duals > 0:
        kept_idx = h["kept_idx"]
        order = np.argsort(-np.abs(duals))[:top_n_duals]
        out["top_duals"] = [
            {"met": metabolites[kept_idx[i]], "dual": float(duals[i])} for i in order
        ]

    # Surface fluxes near upper_flux_bound (likely binders)
    if h["v"].value is not None:
        v_arr = np.asarray(h["v"].value)
        e_arr = np.asarray(h["e"].value)
        threshold = 0.99 * upper_flux_bound
        v_near_cap = np.where(np.abs(v_arr) >= threshold)[0]
        e_near_cap = np.where(np.abs(e_arr) >= threshold)[0]

        model = state["model_t" if source == "target" else "model_e"]
        reactions = list(
            getattr(model, "rxns", None) or getattr(model, "reactions", [])
        ) or [f"rxn_{i}" for i in range(len(v_arr))]
        exch_names = state.get(
            "exchange_metabolites", [f"exch_{i}" for i in range(len(e_arr))]
        )

        out["v_near_cap"] = [
            {
                "rxn": reactions[i] if i < len(reactions) else f"rxn_{i}",
                "flux": float(v_arr[i]),
            }
            for i in v_near_cap[:30]
        ]
        out["e_near_cap"] = [
            {
                "exch": exch_names[i] if i < len(exch_names) else f"exch_{i}",
                "flux": float(e_arr[i]),
            }
            for i in e_near_cap[:30]
        ]
        out["n_v_near_cap"] = int(len(v_near_cap))
        out["n_e_near_cap"] = int(len(e_near_cap))

        # Top-N exchanges by |flux| (informative regardless of cap binding)
        e_order = np.argsort(-np.abs(e_arr))[:20]
        out["e_top_abs"] = [
            {
                "exch": exch_names[i] if i < len(exch_names) else f"exch_{i}",
                "flux": float(e_arr[i]),
            }
            for i in e_order
            if abs(e_arr[i]) > 1e-9
        ]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", default="out/biomass_reaction/state.pkl")
    ap.add_argument("--source", choices=["target", "estimated"], default="estimated")
    ap.add_argument(
        "--drop-mets",
        default="",
        help="comma-separated metabolite IDs to skip in mass balance",
    )
    ap.add_argument(
        "--no-kin-upper",
        action="store_true",
        help="drop hard kinetic upper bound (Mode B-ish)",
    )
    ap.add_argument(
        "--keep-secretion",
        action="store_true",
        help="include secretion penalty (Mode A-ish)",
    )
    ap.add_argument(
        "--kinetic-weight",
        type=float,
        default=0.0,
        help="soft kinetic loss weight w_kin (new soft Mode C). "
        "Use with --no-kin-upper. 0 disables (default = Mode B). "
        "Try 1e-9 .. 1e-1 for the sweep.",
    )
    ap.add_argument(
        "--kinetics-in-range",
        type=float,
        default=1e-2,
        help="in-band sub-weight (default 1e-2, Standalone_FBA pattern).",
    )
    ap.add_argument("--solver", default="HIGHS")
    ap.add_argument("--upper-flux-bound", type=float, default=1e7)
    ap.add_argument("--top-n-duals", type=int, default=20)
    ap.add_argument(
        "--glucose-multiplier",
        type=float,
        default=None,
        help="apply glucose uptake bound = multiplier × pickled default "
        "(default = 10 mmol/gDCW/h). Omit for unbounded glucose.",
    )
    ap.add_argument(
        "--efficiency-weight",
        type=float,
        default=None,
        help="pFBA-style L1 penalty weight on |v| to break degeneracy. "
        "Try 1e-12 → 1e-7. Omit to skip.",
    )
    ap.add_argument(
        "--objective",
        default="biomass",
        help="'biomass' (default) or 'secrete:<exchange_id>' "
        "(e.g. 'secrete:ETOH[p] exchange rev')",
    )
    ap.add_argument(
        "--fix-biomass",
        type=float,
        default=None,
        help="constrain v_biomass_user to this value (production envelope x-axis)",
    )
    ap.add_argument("--json", action="store_true", help="emit JSON to stdout")
    args = ap.parse_args()

    state_path = Path(args.state)
    if not state_path.exists():
        print(f"state not found: {state_path}", file=sys.stderr)
        print("  Run the pickle cell in the notebook first.", file=sys.stderr)
        sys.exit(2)

    state = load_state(state_path)
    result = solve(
        state,
        source=args.source,
        drop_mets=[m for m in args.drop_mets.split(",") if m] or None,
        kin_upper=not args.no_kin_upper,
        drop_secretion=not args.keep_secretion,
        kinetic_weight=args.kinetic_weight,
        kinetics_in_range=args.kinetics_in_range,
        solver=args.solver,
        upper_flux_bound=args.upper_flux_bound,
        top_n_duals=args.top_n_duals,
        glucose_multiplier=args.glucose_multiplier,
        efficiency_weight=args.efficiency_weight,
        objective=args.objective,
        fix_biomass_user=args.fix_biomass,
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return

    cfg = result["config"]
    print(f"solver:           {result['solver']}")
    print(f"status:           {result['status']}")
    if "error" in result:
        print(f"error:            {result['error']}")
    print(f"v_biomass_user:   {result.get('v_biomass_user')}")
    print(f"objective:        {result.get('objective')}")
    print(
        f"config:           source={cfg['source']}  "
        f"drop_mets={cfg['drop_mets_count']}  "
        f"kin_upper={cfg['kin_upper']}  "
        f"drop_secretion={cfg['drop_secretion']}"
    )
    if "top_duals" in result:
        print(f"\ntop {len(result['top_duals'])} mass-balance duals (|dual| desc):")
        for d in result["top_duals"]:
            print(f"  {d['met']:35s}  {d['dual']:>12.4f}")

    if "n_v_near_cap" in result:
        print(
            f"\n{result['n_v_near_cap']} v near cap, {result['n_e_near_cap']} e near cap"
        )
        for d in result.get("v_near_cap", []):
            print(f"  v: {d['rxn']:50s}  {d['flux']:>12.3e}")
        for d in result.get("e_near_cap", []):
            print(f"  e: {d['exch']:50s}  {d['flux']:>12.3e}")


if __name__ == "__main__":
    main()
