"""Plot a phenotype-microarray plate heatmap showing, per well, whether the
OLD model (no 2022 gene-expansion reactions) and the NEW model (with the
2022 gene-expansion reactions) each correctly predict the experimental
growth call.

Each well is colored with one of four discrete categories:
    A - neither model's prediction matches experimental ground truth
    B - only the OLD model matches
    C - only the NEW model matches
    D - both models match

A well with no ground truth call, or infeasible/missing in either model's
results, is left blank (grey).

Usage:
    uv run --env-file .env --project . python3 \
        "notebooks/Heena notebooks/Metabolism_New Genes/202607_plot_alignment_heatmap.py"
"""

import argparse
import importlib.util
from pathlib import Path

import altair as alt
import pandas as pd
import plotly.colors as pc

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

CATEGORY_ORDER = ["A", "B", "C", "D"]
CATEGORY_LABELS = {
    "A": "A: neither correct",
    "B": "B: only OLD correct",
    "C": "C: only NEW correct",
    "D": "D: both correct",
}
CATEGORY_COLORS = pc.qualitative.Pastel[:4]
BLANK_COLOR = pc.qualitative.Pastel[-1]

CELL_WIDTH = 130
CELL_HEIGHT = 100


def load_plot_phenotypic_results():
    """Import 202607_plot_phenotypic_results.py as a module despite the space
    in its directory path, to reuse build_comparison_frame/build_dual_
    comparison_frame unmodified."""
    path = SCRIPT_DIR / "202607_plot_phenotypic_results.py"
    spec = importlib.util.spec_from_file_location("plot_phenotypic_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_altair_figure(chart, out_dir, stem):
    """Write both an interactive out_dir/stem.html and a static
    out_dir/svg/stem.svg (vl-convert) version of an Altair chart, matching
    the convention used by write_figure() in 202607_plot_phenotypic_results.py."""
    html_path = out_dir / f"{stem}.html"
    chart.save(html_path)
    svg_dir = out_dir / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    svg_path = svg_dir / f"{stem}.svg"
    chart.save(svg_path)
    return html_path, svg_path


def build_alignment_heatmap(plate, merged, ppr):
    """8x12 plate heatmap of the 4-category alignment_category for one
    plate: a colored rect per well (discrete category color, blank/grey if
    no ground truth or infeasible in either model) with the compound name
    and category letter overlaid as wrapped multi-line text."""
    sub = merged[merged["plate"] == plate]
    rows = list("ABCDEFGH")
    cols = list(range(1, 13))
    by_well = sub.set_index("well")

    records = []
    for r in rows:
        for c in cols:
            key = f"{r}{c}"
            category, compound = None, None
            if key in by_well.index:
                category = by_well.loc[key, "alignment_category"]
                compound = by_well.loc[key, "compound_name"]
            lines = str(compound).split(" ") if pd.notna(compound) else []
            if category in CATEGORY_ORDER:
                lines = [*lines, category]
            records.append(
                {
                    "row": r,
                    "col": str(c),
                    "category": category if category in CATEGORY_ORDER else "Missing",
                    "label_lines": lines,
                }
            )
    plate_df = pd.DataFrame.from_records(records)

    domain = [*CATEGORY_ORDER, "Missing"]
    colors = [*CATEGORY_COLORS, BLANK_COLOR]
    labels = {**CATEGORY_LABELS, "Missing": "Missing: no ground truth / infeasible"}

    base = alt.Chart(plate_df).encode(
        x=alt.X(
            "col:O",
            sort=[str(c) for c in cols],
            title="Column",
            axis=alt.Axis(labelAngle=0),
        ),
        y=alt.Y("row:O", sort=rows, title="Row"),
    )

    rects = base.mark_rect(stroke="white", strokeWidth=2).encode(
        color=alt.Color(
            "category:N",
            sort=domain,
            scale=alt.Scale(domain=domain, range=colors),
            legend=alt.Legend(
                title=None,
                labelExpr="{"
                + ", ".join(f"'{k}': '{v}'" for k, v in labels.items())
                + "}[datum.value]",
            ),
        ),
    )

    text = base.mark_text(baseline="middle", fontSize=12, lineHeight=14).encode(
        text="label_lines:N",
        color=alt.value("black"),
    )

    chart = (
        (rects + text)
        .properties(
            width=CELL_WIDTH * len(cols),
            height=CELL_HEIGHT * len(rows),
            title=f"{plate}: Predicted-vs-Experimental Alignment (OLD vs NEW model)",
        )
        .configure_view(strokeWidth=0)
    )
    return chart


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
        "--plate",
        default="PM1",
        help="Which plate to plot (default PM1); pass e.g. PM2/PM3/PM4 for others",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory to write the alignment heatmap HTML/SVG files",
    )
    parser.add_argument(
        "--show", action="store_true", help="Also open the figure in a browser"
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

    plate_counts = merged[merged["plate"] == args.plate][
        "alignment_category"
    ].value_counts()
    print(f"{args.plate} alignment counts:\n{plate_counts}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    chart = build_alignment_heatmap(args.plate, merged, ppr)
    fig_path, svg_path = write_altair_figure(
        chart, args.out_dir, f"{args.plate.lower()}_alignment_heatmap"
    )
    print(f"wrote {fig_path}")
    print(f"wrote {svg_path}")
    if args.show:
        chart.show()


if __name__ == "__main__":
    main()
