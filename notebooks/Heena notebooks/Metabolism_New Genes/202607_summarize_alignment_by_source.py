"""Summarize, per nutrient-source category (Carbon/Nitrogen/Phosphorus/
Sulfur), how many phenotype-microarray wells fall into each of the 4
OLD-vs-NEW model alignment categories from 202607_plot_alignment_heatmap.py:

    A - neither model's prediction matches experimental ground truth
    B - only the OLD model matches
    C - only the NEW model matches
    D - both models match
    Missing - infeasible/unavailable prediction in one or both models

Unlike 202607_plot_alignment_heatmap.py (which plots one plate at a time),
this covers all four PM plates so the counts/percentages here are the
one-glance summary of how model alignment differs across nutrient sources.

Usage:
    uv run --env-file .env --project . python3 \
        "notebooks/Heena notebooks/Metabolism_New Genes/202607_summarize_alignment_by_source.py"
"""

import argparse
import importlib.util
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OLD_RESULTS_CSV = (
    SCRIPT_DIR
    / "out"
    / "phenotypic_arrays"
    / "results_no_new_reactions_original_weights.csv"
)
DEFAULT_NEW_RESULTS_CSV = (
    SCRIPT_DIR
    / "out"
    / "phenotypic_arrays"
    / "results_new_reactions_original_weights.csv"
)
DEFAULT_WELLS_JSON = SCRIPT_DIR / "phenotypic_array_wells.json"
DEFAULT_OUT_DIR = SCRIPT_DIR / "out" / "phenotypic_arrays" / "plots"

CATEGORY_ORDER = ["A", "B", "C", "D", "Missing"]
CATEGORY_LABELS = {
    "A": "neither correct",
    "B": "only OLD correct",
    "C": "only NEW correct",
    "D": "both correct",
}
SOURCE_ORDER = ["Carbon", "Nitrogen", "Phosphorus", "Sulfur"]


def load_plot_phenotypic_results():
    """Import 202607_plot_phenotypic_results.py as a module despite the space
    in its directory path, to reuse build_dual_comparison_frame/
    well_source_category/write_figure unmodified."""
    path = SCRIPT_DIR / "202607_plot_phenotypic_results.py"
    spec = importlib.util.spec_from_file_location("plot_phenotypic_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize_alignment_by_source(merged, ppr):
    """Return (counts, percentages) DataFrames: rows = alignment_category
    (A/B/C/D/Missing), columns = nutrient source (Carbon/Nitrogen/
    Phosphorus/Sulfur), percentages are each cell's share of that source's
    total well count (source well counts differ a lot -- Carbon spans two
    plates, Sulfur is only PM4's F-H rows -- so raw counts alone aren't
    comparable across sources)."""
    df = merged.copy()
    df["source"] = [
        ppr.well_source_category(plate, well)
        for plate, well in zip(df["plate"], df["well"])
    ]
    df = df[df["source"].notna()]

    counts = (
        df.groupby(["alignment_category", "source"])
        .size()
        .unstack("source", fill_value=0)
        .reindex(index=CATEGORY_ORDER, columns=SOURCE_ORDER, fill_value=0)
        .astype(int)
    )
    percentages = counts.div(counts.sum(axis=0), axis=1).mul(100).round(1)
    return counts, percentages


def build_summary_table_figure(counts, percentages):
    cell_text = counts.astype(str) + " (" + percentages.astype(str) + "%)"
    row_labels = [CATEGORY_LABELS.get(cat, cat) for cat in counts.index]
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Alignment category", *SOURCE_ORDER],
                    fill_color="#e5ecf6",
                    align="left",
                ),
                cells=dict(
                    values=[row_labels, *[cell_text[col] for col in SOURCE_ORDER]],
                    align="left",
                ),
            )
        ]
    )
    fig.update_layout(
        title="OLD-vs-NEW Model Alignment by Nutrient Source (count (% of source total))",
        width=800,
        height=280,
        margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-old-csv",
        type=Path,
        default=DEFAULT_OLD_RESULTS_CSV,
        help="Path to the OLD model's results.csv (no 2022 gene-expansion reactions)",
    )
    parser.add_argument(
        "--results-new-csv",
        type=Path,
        default=DEFAULT_NEW_RESULTS_CSV,
        help="Path to the NEW model's results.csv (with 2022 gene-expansion reactions)",
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
        help="Directory to write the summary CSV and table HTML/SVG files",
    )
    parser.add_argument(
        "--show", action="store_true", help="Also open the table figure in a browser"
    )
    args = parser.parse_args()

    ppr = load_plot_phenotypic_results()

    results_old = pd.read_csv(args.results_old_csv)
    results_new = pd.read_csv(args.results_new_csv)
    ground_truth = ppr.load_ground_truth(args.wells_json)

    threshold_old = ppr.compute_growth_threshold(results_old, ppr.NEG_CONTROL_WELLS)
    threshold_new = ppr.compute_growth_threshold(results_new, ppr.NEG_CONTROL_WELLS)
    print(
        f"OLD growth threshold = {threshold_old}, NEW growth threshold = {threshold_new}"
    )

    merged = ppr.build_dual_comparison_frame(
        results_old, results_new, ground_truth, threshold_old, threshold_new
    )

    counts, percentages = summarize_alignment_by_source(merged, ppr)
    print("Counts:")
    print(counts)
    print("Percentages (of each source's total):")
    print(percentages)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.out_dir / "alignment_summary_by_source.csv"
    counts.to_csv(csv_path)
    print(f"wrote {csv_path}")

    fig = build_summary_table_figure(counts, percentages)
    fig_path, svg_path = ppr.write_figure(
        fig, args.out_dir, "alignment_summary_by_source"
    )
    print(f"wrote {fig_path}")
    print(f"wrote {svg_path}")
    if args.show:
        fig.show()


if __name__ == "__main__":
    main()
