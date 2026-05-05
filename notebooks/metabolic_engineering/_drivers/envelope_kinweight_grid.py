"""Production-envelope grid sweep over (product × kinetic_weight).

For each product and kinetic weight, finds v_biomass_max, then sweeps
v_biomass from 0 to v_max maxing product secretion at each fixed point.
Writes one long-format CSV with columns:
    product, kinetic_weight, v_biomass_user, product_counts_s,
    product_mmol_gdcw_h, status, v_biomass_max

Also captures exchange flux profile at envelope extremes (v_biomass = 0
and v_biomass = v_max) per (product, weight) — written to a sidecar JSON.

Usage:
    uv run python notebooks/metabolic_engineering/_drivers/envelope_kinweight_grid.py \\
        --products 'ETOH[p] exchange rev,TRP[c] exchange rev' \\
        --kinetic-weights 0,1e-9,1e-7,1e-5,1e-3 --n-steps 15
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from solve_biomass import load_state, solve


# Bridge constants (counts/s -> mmol/gDCW/h). Matches envelope_sweep.py.
C2M = 1.343e-6  # mM/count, empirical from steady-state window
RHO_DRY = 330.0  # gDCW/L


def to_mmol(counts_s):
    if counts_s is None:
        return None
    return counts_s * C2M * 3600 / RHO_DRY


def find_vmax(
    state, source, glucose_multiplier, kinetic_weight, efficiency_weight, solver, ufb
):
    r = solve(
        state,
        source=source,
        kin_upper=False,
        drop_secretion=True,
        kinetic_weight=kinetic_weight,
        efficiency_weight=efficiency_weight,
        solver=solver,
        upper_flux_bound=ufb,
        top_n_duals=0,
        glucose_multiplier=glucose_multiplier,
        objective="biomass",
    )
    if r["status"] != "optimal":
        return None, r
    return r["v_biomass_user"], r


def sweep_envelope(
    state,
    source,
    product,
    glucose_multiplier,
    kinetic_weight,
    efficiency_weight,
    n_steps,
    v_max,
    solver,
    ufb,
):
    biomass_pts = np.linspace(0, v_max, n_steps)
    rows = []
    for vb in biomass_pts:
        r = solve(
            state,
            source=source,
            kin_upper=False,
            drop_secretion=True,
            kinetic_weight=kinetic_weight,
            efficiency_weight=efficiency_weight,
            solver=solver,
            upper_flux_bound=ufb,
            top_n_duals=0,
            glucose_multiplier=glucose_multiplier,
            objective=f"secrete:{product}",
            fix_biomass_user=float(vb),
        )
        prod = r.get("product_flux_counts_s") if r["status"] == "optimal" else None
        rows.append(
            {
                "v_biomass_user": float(vb),
                "product_counts_s": prod,
                "product_mmol_gdcw_h": to_mmol(prod),
                "status": r["status"],
                "e_near_cap": r.get("e_near_cap", []),
                "e_top_abs": r.get("e_top_abs", []),
                "v_near_cap_count": r.get("n_v_near_cap", 0),
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", default="out/biomass_reaction/state.pkl")
    ap.add_argument("--source", choices=["target", "estimated"], default="estimated")
    ap.add_argument(
        "--products",
        required=True,
        help="comma-separated exchange IDs to maximize, "
        "e.g. 'ETOH[p] exchange rev,TRP[c] exchange rev'",
    )
    ap.add_argument(
        "--kinetic-weights",
        default="0,1e-4,3e-4,5e-4,7e-4,1e-3,3e-3",
        help="comma-separated soft kinetic weights to sweep. "
        "Default spans the phase transition observed at GUR=10.",
    )
    ap.add_argument(
        "--efficiency-weight",
        type=float,
        default=1e-9,
        help="pFBA-style L1 weight on |v|, breaks futile-cycle "
        "degeneracy and stabilizes solver across kinetic "
        "weights. Default 1e-9 is small enough not to "
        "shift envelopes but large enough to fix tolerance "
        "flapping. Set 0 to disable.",
    )
    ap.add_argument(
        "--glucose-multiplier",
        type=float,
        default=1.0,
        help="glucose uptake multiplier (1.0 = iML1515 default 10 mmol/gDCW/h)",
    )
    ap.add_argument("--n-steps", type=int, default=15)
    ap.add_argument("--solver", default="HIGHS")
    ap.add_argument("--upper-flux-bound", type=float, default=1e7)
    ap.add_argument(
        "--out-csv", default="out/biomass_reaction/envelope_kinweight_grid.csv"
    )
    ap.add_argument(
        "--out-extremes", default="out/biomass_reaction/envelope_extremes.json"
    )
    args = ap.parse_args()

    products = [p.strip() for p in args.products.split(",") if p.strip()]
    weights = [float(w) for w in args.kinetic_weights.split(",") if w.strip()]

    state = load_state(args.state)

    out_rows = []
    extremes = []

    for product in products:
        for w in weights:
            label = f"product={product!r} w_kin={w:g}"
            print(f"\n=== {label} ===")

            # Per-weight v_max via biomass-max with soft kinetic loss.
            # This is the LP's voluntary stopping point — the envelope's
            # right boundary in soft Mode C. Sweep [0, v_max(w)] at this
            # weight; the right endpoint of each curve is on the Mode-B
            # Pareto front (product > 0) since the LP can rebalance to
            # admit product flux at this v_biomass. The plot adds a
            # vertical drop from that endpoint to (v_max(w), 0) to mark
            # the biomass-max corner explicitly.
            v_max, vmax_result = find_vmax(
                state,
                args.source,
                args.glucose_multiplier,
                w,
                args.efficiency_weight,
                args.solver,
                args.upper_flux_bound,
            )
            if v_max is None:
                print(f"  v_biomass_max solve failed: status={vmax_result['status']}")
                out_rows.append(
                    {
                        "product": product,
                        "kinetic_weight": w,
                        "v_biomass_user": None,
                        "product_counts_s": None,
                        "product_mmol_gdcw_h": None,
                        "status": f"vmax_{vmax_result['status']}",
                        "v_biomass_max": None,
                    }
                )
                continue
            print(f"  v_biomass_max = {v_max:.4f}  -- envelope right boundary")

            rows = sweep_envelope(
                state,
                args.source,
                product,
                args.glucose_multiplier,
                w,
                args.efficiency_weight,
                args.n_steps,
                v_max,
                args.solver,
                args.upper_flux_bound,
            )
            print(
                f"  swept {len(rows)} points; "
                f"product@v=0: {to_mmol(rows[0]['product_counts_s'])}, "
                f"product@v_max: {to_mmol(rows[-1]['product_counts_s'])}"
            )

            # Capture exchange profile at the two extremes for this (product, w)
            extremes.append(
                {
                    "product": product,
                    "kinetic_weight": w,
                    "v_biomass_max": float(v_max),
                    "extreme_v0": {
                        "v_biomass_user": rows[0]["v_biomass_user"],
                        "product_mmol_gdcw_h": rows[0]["product_mmol_gdcw_h"],
                        "e_near_cap": rows[0]["e_near_cap"],
                        "e_top_abs": rows[0]["e_top_abs"],
                        "n_v_near_cap": rows[0]["v_near_cap_count"],
                    },
                    "extreme_vmax": {
                        "v_biomass_user": rows[-1]["v_biomass_user"],
                        "product_mmol_gdcw_h": rows[-1]["product_mmol_gdcw_h"],
                        "e_near_cap": rows[-1]["e_near_cap"],
                        "e_top_abs": rows[-1]["e_top_abs"],
                        "n_v_near_cap": rows[-1]["v_near_cap_count"],
                    },
                }
            )

            for r in rows:
                out_rows.append(
                    {
                        "product": product,
                        "kinetic_weight": w,
                        "v_biomass_user": r["v_biomass_user"],
                        "product_counts_s": r["product_counts_s"],
                        "product_mmol_gdcw_h": r["product_mmol_gdcw_h"],
                        "status": r["status"],
                        "v_biomass_max": float(v_max),
                    }
                )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(
            f,
            fieldnames=[
                "product",
                "kinetic_weight",
                "v_biomass_user",
                "product_counts_s",
                "product_mmol_gdcw_h",
                "status",
                "v_biomass_max",
            ],
        )
        wr.writeheader()
        wr.writerows(out_rows)
    print(f"\nWrote {out_csv} ({len(out_rows)} rows)")

    out_ext = Path(args.out_extremes)
    out_ext.parent.mkdir(parents=True, exist_ok=True)
    with open(out_ext, "w") as f:
        json.dump(extremes, f, indent=2)
    print(f"Wrote {out_ext}")


if __name__ == "__main__":
    main()
