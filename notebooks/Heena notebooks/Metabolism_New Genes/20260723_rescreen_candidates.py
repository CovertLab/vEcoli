"""
Phase A re-screen: select weight-combo candidates that sit in the real
homeostatic-kinetic trade-off region rather than the noise floor.

Background: `best_of_best.csv` (built from the old, stale sim snapshot) was
filtered on a narrow, near-zero `obj_homeo` band, which by construction
selected weight combos where the kinetics penalty is not binding at the
optimum — this is why a fraction_kinetic_target knockdown never moved
obj_homeo for any of those candidates (see notebooks/Heena notebooks/
Metabolism_New Genes/20260506_fraction_kinetic_target.ipynb and this
session's LP-theory + correlation analysis). This script instead selects
for weight combos with a large lambda_kin/lambda_hom ratio (closer to the
LP's basis-change boundary) while requiring obj_homeo to sit well above the
solver noise floor, from the freshly regenerated sweep in
pareto_results_jul_10000samples/pareto_results.csv (current process code,
widened kinetics weight range).

Usage:
    uvenv notebooks/Heena\ notebooks/Metabolism_New\ Genes/20260723_rescreen_candidates.py
"""

import os

import plotly.express.colors as pc
import plotly.graph_objects as go
import polars as pl

OUT_DIR = "notebooks/Heena notebooks/Metabolism_New Genes/pareto_results_relationship_sep_v1_10000samples"
TOYA_R2_MIN = 0.6
OBJ_HOMEO_NOISE_FLOOR = (
    1e-12  # empirically: obj_homeo deciles jump from ~1e-11 to ~2e-4
)
N_CANDIDATES = 1000


def rescreen() -> pl.DataFrame:
    os.chdir(os.path.expanduser("~/dev/vEcoli/"))

    df = pl.read_csv(f"{OUT_DIR}/pareto_results.csv").with_row_index("Index")
    df = df.filter(pl.col("toya_r_squared").is_finite())
    print(f"Loaded {df.height} finite-R2 solves.")

    feasible = df.filter(pl.col("toya_r_squared") > TOYA_R2_MIN)
    print(f"{feasible.height} solves pass toya_r_squared > {TOYA_R2_MIN}.")

    feasible = feasible.with_columns(
        (pl.col("lambda_kin") / pl.col("lambda_hom")).alias("kin_hom_ratio")
    )

    real_competition = feasible.filter(pl.col("obj_homeo") <= OBJ_HOMEO_NOISE_FLOOR)
    print(
        f"{real_competition.height} of those have obj_homeo <= "
        f"{OBJ_HOMEO_NOISE_FLOOR:.0e} (i.e. the kinetics penalty is "
        f"actually binding, not solver noise floor)."
    )

    shortlist = real_competition.sort("kin_hom_ratio", descending=True).head(
        N_CANDIDATES
    )
    out_path = f"{OUT_DIR}/best_of_best_low_toya.csv"
    shortlist.write_csv(out_path)
    print(f"Saved {shortlist.height}-candidate shortlist: {out_path}")
    print(
        shortlist.select(
            [
                "Index",
                "lambda_hom",
                "lambda_kin",
                "kin_hom_ratio",
                "lambda_eff",
                "lambda_sec",
                "lambda_div",
                "obj_homeo",
                "obj_kin",
                "toya_r_squared",
            ]
        )
    )

    plot_selection(df, feasible, shortlist)
    return shortlist


def plot_selection(
    df: pl.DataFrame, feasible: pl.DataFrame, shortlist: pl.DataFrame
) -> None:
    """
    Scatter of obj_kin vs obj_homeo (log-log), all solves in light gray,
    R2>0.5 solves colored by toya_r_squared, and the Phase A shortlist
    highlighted — showing the shortlist sits in the genuine trade-off
    region rather than the noise floor. Pastel palette per project
    convention; saved as svg with title/axis labels/legend.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["obj_kin"],
            y=df["obj_homeo"],
            mode="markers",
            name="All solves",
            marker=dict(size=4, color="lightgray", opacity=0.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=feasible["obj_kin"],
            y=feasible["obj_homeo"],
            mode="markers",
            name=f"toya_r² > {TOYA_R2_MIN}",
            marker=dict(
                size=5,
                color=feasible["toya_r_squared"],
                colorscale=[pc.qualitative.Pastel[0], pc.qualitative.Pastel[2]],
                colorbar=dict(title="toya R²"),
                opacity=0.7,
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=shortlist["obj_kin"],
            y=shortlist["obj_homeo"],
            mode="markers",
            name="Phase A shortlist (large λ_kin/λ_hom)",
            marker=dict(
                size=11,
                color=pc.qualitative.Pastel[4],
                line=dict(width=1.5, color="black"),
                symbol="diamond",
            ),
        )
    )
    fig.add_hline(
        y=OBJ_HOMEO_NOISE_FLOOR,
        line_dash="dot",
        line_color=pc.qualitative.Pastel[3],
        annotation_text="solver noise-floor cutoff",
    )

    fig.update_xaxes(type="log", title="Kinetic Objective (obj_kin, unweighted)")
    fig.update_yaxes(type="log", title="Homeostatic Objective (obj_homeo, unweighted)")
    fig.update_layout(
        title="Phase A re-screen: candidates in the real homeostatic-kinetic "
        "trade-off region",
        legend_title="Legend",
        template="plotly_white",
        width=900,
        height=650,
    )

    out_path = f"{OUT_DIR}/rescreen_candidates.svg"
    fig.write_image(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    rescreen()
