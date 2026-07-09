"""Compare FBA-simulated phenotype-microarray growth predictions against real
EcoCyc experimental growth calls, visualized as confusion matrices.

Predicted growth comes from `results.csv` (written by
202607_run_phenotypic_arrays.py): a well's `homeostatic_objective` is reduced
to a binary grows / no-growth call using the same Growth/No-Growth threshold
as 202607_plot_phenotypic_results.py's `compute_growth_threshold()` -- the
lowest homeostatic_objective among the four negative controls, computed once
from all wells combined.

Experimental ("ground truth") growth comes from the `aerobic_growth_call`
field in `phenotypic_array_wells.json` (fetched from EcoCyc): "Growth" and
"Low Growth" both count as "grows", "No Growth" counts as "no growth", and
"Indeterminate" or missing calls are excluded entirely.

Usage:
    uv run --env-file .env --project . python3 \
        "notebooks/Heena notebooks/Metabolism_New Genes/plot_confusion_matrices.py"
"""

import argparse
import importlib.util
import json
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_CSV = SCRIPT_DIR / "out" / "phenotypic_arrays" / "results.csv"
DEFAULT_WELLS_JSON = SCRIPT_DIR / "phenotypic_array_wells.json"
DEFAULT_OUT_DIR = SCRIPT_DIR / "out" / "phenotypic_arrays" / "plots"

WELL_RE = re.compile(r"^([A-H])(\d{1,2})$")

CORRECT_COLOR = "#ddeee2"
INCORRECT_COLOR = "#f8ddd8"
FIGURE_BGCOLOR = "#e8e8e8"

GROWS_CALLS = {"Growth", "Low Growth"}
NO_GROWTH_CALLS = {"No Growth"}


def load_plot_phenotypic_results():
    """Import 202607_plot_phenotypic_results.py as a module despite the space
    in its directory path, to reuse compute_band_boundaries() unmodified."""
    path = SCRIPT_DIR / "202607_plot_phenotypic_results.py"
    spec = importlib.util.spec_from_file_location("plot_phenotypic_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ground_truth(wells_json_path):
    """Build a (plate, well) -> aerobic_growth_call lookup from the EcoCyc
    phenotypic_array_wells.json export."""
    with open(wells_json_path) as handle:
        wells = json.load(handle)
    ground_truth = {}
    for entry in wells:
        well = f"{entry['well_row']}{entry['well_col']}"
        ground_truth[(entry["plate"], well)] = entry.get("aerobic_growth_call")
    return ground_truth


def experimental_grows(call):
    """Map an aerobic_growth_call to True (grows), False (no growth), or
    None (excluded: Indeterminate or missing)."""
    if call in GROWS_CALLS:
        return True
    if call in NO_GROWTH_CALLS:
        return False
    return None


def build_comparison_frame(results_df, ground_truth, q_hi):
    """Attach predicted_grows and experimental_grows columns, keeping only
    wells where both a valid prediction and valid ground truth exist."""
    df = results_df.copy()
    df["experimental_call"] = df.apply(
        lambda row: ground_truth.get((row["plate"], row["well"])), axis=1
    )
    df["experimental_grows"] = df["experimental_call"].map(experimental_grows)
    df["predicted_grows"] = df["homeostatic_objective"].apply(
        lambda value: value <= q_hi if pd.notna(value) else None
    )
    df = df[df["predicted_grows"].notna() & df["experimental_grows"].notna()].copy()
    df["predicted_grows"] = df["predicted_grows"].astype(bool)
    df["experimental_grows"] = df["experimental_grows"].astype(bool)
    return df


def confusion_counts(df):
    """Return (tp, fp, fn, tn) counts from a frame with predicted_grows and
    experimental_grows boolean columns."""
    tp = int(((df["predicted_grows"]) & (df["experimental_grows"])).sum())
    fp = int(((df["predicted_grows"]) & (~df["experimental_grows"])).sum())
    fn = int(((~df["predicted_grows"]) & (df["experimental_grows"])).sum())
    tn = int(((~df["predicted_grows"]) & (~df["experimental_grows"])).sum())
    return tp, fp, fn, tn


def add_confusion_matrix(fig, tp, fp, fn, tn, row=None, col=None):
    """Draw one 2x2 confusion matrix (as a heatmap + annotations) onto fig,
    either standalone (row/col=None) or into a subplot cell."""
    trace_kwargs = {"row": row, "col": col} if row is not None else {}
    axis_kwargs = {"row": row, "col": col} if row is not None else {}

    heatmap = go.Heatmap(
        z=[[1, 0], [0, 1]],
        x=[0, 1],
        y=[0, 1],
        colorscale=[[0, INCORRECT_COLOR], [1, CORRECT_COLOR]],
        showscale=False,
        hoverinfo="skip",
        xgap=3,
        ygap=3,
    )
    fig.add_trace(heatmap, **trace_kwargs)

    fig.update_xaxes(
        tickvals=[0, 1],
        ticktext=["Experimental: grows", "Experimental: no growth"],
        showgrid=False,
        zeroline=False,
        **axis_kwargs,
    )
    fig.update_yaxes(
        tickvals=[0, 1],
        ticktext=["Predicted: grows", "Predicted: no growth"],
        autorange="reversed",
        showgrid=False,
        zeroline=False,
        **axis_kwargs,
    )

    cells = [
        (0, 0, tp, "true positive"),
        (1, 0, fp, "false positive"),
        (0, 1, fn, "false negative"),
        (1, 1, tn, "true negative"),
    ]
    for x, y, count, label in cells:
        annotation_kwargs = {"row": row, "col": col} if row is not None else {}
        fig.add_annotation(
            x=x,
            y=y - 0.18,
            text=f"<b>{count}</b>",
            showarrow=False,
            font=dict(size=34),
            **annotation_kwargs,
        )
        fig.add_annotation(
            x=x,
            y=y + 0.2,
            text=label,
            showarrow=False,
            font=dict(size=12),
            **annotation_kwargs,
        )


def build_overall_figure(tp, fp, fn, tn, n):
    fig = go.Figure()
    add_confusion_matrix(fig, tp, fp, fn, tn)
    fig.update_layout(
        title=f"Overall: Predicted vs. Experimental Growth (n={n})",
        paper_bgcolor=FIGURE_BGCOLOR,
        plot_bgcolor=FIGURE_BGCOLOR,
        width=560,
        height=560,
        margin=dict(t=80, b=40, l=120, r=40),
    )
    return fig


def source_category_masks(df):
    row = df["well"].str.extract(WELL_RE)[0]
    return {
        "Carbon": df["plate"].isin(["PM1", "PM2"]),
        "Nitrogen": df["plate"] == "PM3",
        "Phosphorus": (df["plate"] == "PM4") & row.isin(list("ABCDE")),
        "Sulfur": (df["plate"] == "PM4") & row.isin(list("FGH")),
    }


def build_by_source_figure(df):
    categories = ["Carbon", "Nitrogen", "Phosphorus", "Sulfur"]
    masks = source_category_masks(df)

    counts_by_category = {}
    for category in categories:
        sub = df[masks[category]]
        counts_by_category[category] = (confusion_counts(sub), len(sub))

    subplot_titles = [
        f"{category} (n={counts_by_category[category][1]})" for category in categories
    ]
    fig = make_subplots(rows=2, cols=2, subplot_titles=subplot_titles)

    positions = {
        "Carbon": (1, 1),
        "Nitrogen": (1, 2),
        "Phosphorus": (2, 1),
        "Sulfur": (2, 2),
    }
    for category, (row, col) in positions.items():
        (tp, fp, fn, tn), _n = counts_by_category[category]
        add_confusion_matrix(fig, tp, fp, fn, tn, row=row, col=col)

    fig.update_layout(
        title="Predicted vs. Experimental Growth by Nutrient Source",
        paper_bgcolor=FIGURE_BGCOLOR,
        plot_bgcolor=FIGURE_BGCOLOR,
        width=1000,
        height=1000,
        margin=dict(t=100, b=40, l=80, r=40),
        showlegend=False,
    )
    return fig, counts_by_category


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help="Path to results.csv written by 202607_run_phenotypic_arrays.py",
    )
    parser.add_argument(
        "--wells-json",
        type=Path,
        default=DEFAULT_WELLS_JSON,
        help="Path to phenotypic_array_wells.json (EcoCyc ground-truth export)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory to write the confusion matrix HTML files",
    )
    parser.add_argument(
        "--show", action="store_true", help="Also open each figure in a browser"
    )
    args = parser.parse_args()

    ppr = load_plot_phenotypic_results()

    results_df = pd.read_csv(args.results_csv)
    ground_truth = load_ground_truth(args.wells_json)

    all_values = results_df["homeostatic_objective"].dropna().tolist()
    growth_threshold = ppr.compute_growth_threshold(results_df, ppr.NEG_CONTROL_WELLS)
    boundaries = ppr.compute_band_boundaries(all_values, growth_threshold)
    q_hi = boundaries[1]
    print(f"Global band boundaries (min, threshold, max) = {boundaries}")
    print(f"Predicted grows <= threshold = {q_hi}")

    df = build_comparison_frame(results_df, ground_truth, q_hi)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    tp, fp, fn, tn = confusion_counts(df)
    n = len(df)
    print(f"Overall: n={n} wells with valid prediction and ground truth")
    print(f"Overall: TP={tp} FP={fp} FN={fn} TN={tn}")

    overall_fig = build_overall_figure(tp, fp, fn, tn, n)
    overall_path, overall_svg_path = ppr.write_figure(
        overall_fig, args.out_dir, "confusion_matrix_overall"
    )
    print(f"wrote {overall_path}")
    print(f"wrote {overall_svg_path}")
    if args.show:
        overall_fig.show()

    by_source_fig, counts_by_category = build_by_source_figure(df)
    for category, ((c_tp, c_fp, c_fn, c_tn), c_n) in counts_by_category.items():
        print(f"{category}: n={c_n} wells with valid prediction and ground truth")
        print(f"{category}: TP={c_tp} FP={c_fp} FN={c_fn} TN={c_tn}")

    by_source_path, by_source_svg_path = ppr.write_figure(
        by_source_fig, args.out_dir, "confusion_matrix_by_source"
    )
    print(f"wrote {by_source_path}")
    print(f"wrote {by_source_svg_path}")
    if args.show:
        by_source_fig.show()


if __name__ == "__main__":
    main()
