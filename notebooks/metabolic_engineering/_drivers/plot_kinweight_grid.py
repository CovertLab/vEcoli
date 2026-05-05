"""Visualize the soft-kinetic-weight envelope grid.

Reads out/biomass_reaction/envelope_kinweight_grid.csv (long format) and
writes:
  - envelope_kinweight_<product>.html : family-of-curves per product
  - vmax_vs_kinweight.html             : v_biomass_max vs weight (both products)

Usage:
    uv run python notebooks/metabolic_engineering/_drivers/plot_kinweight_grid.py
"""

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--grid-csv", default="out/biomass_reaction/envelope_kinweight_grid.csv"
    )
    ap.add_argument("--out-dir", default="out/biomass_reaction")
    ap.add_argument(
        "--mu-per-vbiomass",
        type=float,
        default=0.945,
        help="μ (1/h) at v_biomass_user=1 — for the secondary x-axis. "
        "Default 0.945 is parameterized basal μ for M9 + glucose.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.grid_csv)
    df = df.dropna(subset=["v_biomass_user", "product_mmol_gdcw_h"])
    out_dir = Path(args.out_dir)

    products = sorted(df["product"].unique())
    weights = sorted(df["kinetic_weight"].unique())

    # 1) Family of curves per product
    palette = px.colors.sequential.Viridis
    color_for = {
        w: palette[int(round(i * (len(palette) - 1) / max(1, len(weights) - 1)))]
        for i, w in enumerate(weights)
    }
    for product in products:
        sub = df[df["product"] == product]
        fig = go.Figure()

        # Mode-B (w=0) full envelope as faint dashed backdrop
        if 0 in weights:
            backdrop = sub[sub["kinetic_weight"] == 0].sort_values("v_biomass_user")
            fig.add_trace(
                go.Scatter(
                    x=backdrop["v_biomass_user"] * args.mu_per_vbiomass,
                    y=backdrop["product_mmol_gdcw_h"],
                    mode="lines",
                    line=dict(color="lightgray", width=1.5, dash="dash"),
                    name="Mode-B envelope (backdrop)",
                    hoverinfo="skip",
                    showlegend=True,
                )
            )

        for w in weights:
            curve = sub[sub["kinetic_weight"] == w].sort_values("v_biomass_user")
            label = "Mode B (w=0)" if w == 0 else f"w_kin = {w:.0e}"

            x_pts = (curve["v_biomass_user"] * args.mu_per_vbiomass).tolist()
            y_pts = curve["product_mmol_gdcw_h"].tolist()

            # Append a vertical drop from the curve's right endpoint to
            # (v_max(w), 0), marking the biomass-max corner explicitly.
            # See note: this drop is procedural — it represents the
            # voluntary-stopping LP solution at v_max(w), product=0
            # (biomass-max with efficiency tie-break), distinct from the
            # product-max LP solution we sweep along the rest of the curve.
            x_with_drop = x_pts + [x_pts[-1]]
            y_with_drop = y_pts + [0.0]

            # Marker sizes/symbols: small circles on the swept points,
            # large diamond at the curve's product-max endpoint, large
            # square at the (v_max, 0) corner.
            marker_sizes = [4] * (len(curve) - 1) + [13, 11]
            marker_symbols = ["circle"] * (len(curve) - 1) + ["diamond", "square"]

            fig.add_trace(
                go.Scatter(
                    x=x_with_drop,
                    y=y_with_drop,
                    mode="lines+markers",
                    name=label,
                    line=dict(color=color_for[w], width=2.0),
                    marker=dict(
                        color=color_for[w],
                        size=marker_sizes,
                        symbol=marker_symbols,
                        line=dict(color="black", width=0.5),
                    ),
                )
            )
        product_short = product.split(" ")[0].replace("[p]", "").replace("[c]", "")
        fig.update_layout(
            title=(
                f"Production envelope: {product_short} "
                f"(soft kinetic weight family, GUR=10 mmol/gDCW/h, source=estimated)"
                "<br><sub>Diamond = max-product LP at v=v_max(w_kin); "
                "square = biomass-max LP corner (product≈0). "
                "Vertical drop is procedural, marking v_max(w) on the x-axis.</sub>"
            ),
            xaxis_title="Growth rate μ (1/h)",
            yaxis_title=f"{product_short} secretion (mmol/gDCW/h)",
            legend_title="Soft kinetic weight",
            template="plotly_white",
            width=950,
            height=600,
        )
        out_path = out_dir / f"envelope_kinweight_{product_short}.html"
        fig.write_html(out_path)
        print(f"wrote {out_path}")

    # 2) v_biomass_max vs kinetic weight (independent of product — pure biomass-max)
    vmax = (
        df.groupby("kinetic_weight")["v_biomass_max"]
        .first()
        .reset_index()
        .sort_values("kinetic_weight")
    )
    nonzero_min = vmax["kinetic_weight"][vmax["kinetic_weight"] > 0].min()
    w0_x = nonzero_min / 10.0  # placeholder for w=0 on log axis
    x = vmax["kinetic_weight"].replace(0, w0_x)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=vmax["v_biomass_max"] * args.mu_per_vbiomass,
            mode="lines+markers",
            line=dict(color="#2c3e50", width=2.5),
            marker=dict(size=9, color="#2c3e50"),
            showlegend=False,
        )
    )
    fig.update_xaxes(
        type="log",
        title="Soft kinetic weight w<sub>kin</sub>",
        exponentformat="power",
        showexponent="all",
        dtick=1,
    )
    fig.update_yaxes(title="μ<sub>max</sub> (1/h)")
    fig.update_layout(
        title=(
            "Biomass-max growth rate vs soft kinetic weight "
            "(phase transition; product-independent)"
        ),
        template="plotly_white",
        width=800,
        height=480,
        annotations=[
            dict(
                x=x.iloc[0],
                y=vmax["v_biomass_max"].iloc[0] * args.mu_per_vbiomass,
                text="w=0 (Mode B)",
                showarrow=True,
                arrowhead=2,
                ax=40,
                ay=-30,
                font=dict(size=11),
            )
        ],
    )
    out_path = out_dir / "vmax_vs_kinweight.html"
    fig.write_html(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
