"""Production envelope sweep: max product secretion at fixed biomass flux.

Sweeps v_biomass_user from 0 to v_max in n_steps, maxing product secretion
at each fixed point. Outputs a CSV + a quick text plot.

Usage:
    uv run python notebooks/metabolic_engineering/_drivers/envelope_sweep.py \\
        --product 'ETOH[p] exchange rev' --glucose-multiplier 1.0 --n-steps 25
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from solve_biomass import load_state, solve


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", default="out/biomass_reaction/state.pkl")
    ap.add_argument("--source", choices=["target", "estimated"], default="estimated")
    ap.add_argument(
        "--product",
        required=True,
        help="exchange ID to maximize, e.g. 'ETOH[p] exchange rev'",
    )
    ap.add_argument("--glucose-multiplier", type=float, default=1.0)
    ap.add_argument("--n-steps", type=int, default=25)
    ap.add_argument(
        "--v-max",
        type=float,
        default=None,
        help="upper end of biomass sweep; if None, find via biomass-max solve",
    )
    ap.add_argument("--solver", default="HIGHS")
    ap.add_argument("--upper-flux-bound", type=float, default=1e7)
    ap.add_argument("--out-csv", default="out/biomass_reaction/envelope_ethanol.csv")
    args = ap.parse_args()

    state = load_state(args.state)

    # Bridge constants for unit conversion at output boundary
    c2m = 1.343e-6  # mM/count (empirical, from the steady-state window check)
    rho_dry = 330.0  # gDCW/L

    def to_mmol(counts_s):
        return counts_s * c2m * 3600 / rho_dry

    # 1) Find v_biomass_max if not given
    if args.v_max is None:
        print("Finding v_biomass_max under given conditions...")
        r = solve(
            state,
            source=args.source,
            kin_upper=False,
            drop_secretion=True,
            solver=args.solver,
            upper_flux_bound=args.upper_flux_bound,
            top_n_duals=0,
            glucose_multiplier=args.glucose_multiplier,
            objective="biomass",
        )
        if r["status"] != "optimal":
            print(f"biomass-max solve failed: {r}")
            return
        v_max = r["v_biomass_user"]
        print(f"  v_biomass_max = {v_max:.4f} (objective bound)")
    else:
        v_max = args.v_max

    # 2) Sweep from 0 → v_max, maxing product at each fixed v_biomass
    biomass_pts = np.linspace(0, v_max, args.n_steps)
    print(
        f"\nSweeping v_biomass over {args.n_steps} points in [0, {v_max:.4f}], "
        f"maxing {args.product!r}..."
    )

    rows = []
    for vb in biomass_pts:
        r = solve(
            state,
            source=args.source,
            kin_upper=False,
            drop_secretion=True,
            solver=args.solver,
            upper_flux_bound=args.upper_flux_bound,
            top_n_duals=0,
            glucose_multiplier=args.glucose_multiplier,
            objective=f"secrete:{args.product}",
            fix_biomass_user=float(vb),
        )
        if r["status"] != "optimal":
            rows.append(
                {
                    "v_biomass_user": float(vb),
                    "product_counts_s": None,
                    "product_mmol_gdcw_h": None,
                    "status": r["status"],
                }
            )
            continue
        prod = r["product_flux_counts_s"]
        rows.append(
            {
                "v_biomass_user": float(vb),
                "product_counts_s": prod,
                "product_mmol_gdcw_h": to_mmol(prod),
                "status": r["status"],
            }
        )

    # 3) Write CSV
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "v_biomass_user",
                "product_counts_s",
                "product_mmol_gdcw_h",
                "status",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_path}")

    # 4) Quick text table
    print(f"\n{'v_biomass':>12}  {'product (mmol/gDCW/h)':>25}  {'status':>10}")
    for r in rows:
        vb = r["v_biomass_user"]
        if r["product_mmol_gdcw_h"] is None:
            print(f"  {vb:>10.4f}  {'—':>23}  {r['status']:>10}")
        else:
            print(
                f"  {vb:>10.4f}  {r['product_mmol_gdcw_h']:>23.4f}  {r['status']:>10}"
            )


if __name__ == "__main__":
    main()
