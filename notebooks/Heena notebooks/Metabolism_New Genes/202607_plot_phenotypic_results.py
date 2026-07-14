"""Recreate the phenotype-microarray histogram and heatmap plots from
20250616_test_carbon_source_cp3.ipynb (cells 35-42), reading from the CSV
written by 202607_run_phenotypic_arrays.py instead of live notebook state,
and compare FBA-simulated growth predictions against real EcoCyc
experimental growth calls as confusion matrices.

The histogram combines all four PM plates' wells into one shared plot, with
dotted reference lines for each plate's own negative control (PM1 carbon,
PM3 nitrogen, PM4 phosphorus, PM4 sulfur) plus an optional minimal-media
("Basal") reference line computed by actually re-running FBA with no
Add/Remove at all (true minimal media, i.e. normal growth) via
202607_run_phenotypic_arrays.test_NetworkFlowModel. Heatmaps stay per-plate
(an 8x12 grid only makes sense per plate) and are strictly binary
Growth/No-Growth, each well judged against its OWN nutrient-source
category's negative control (not one plate-wide or global cutoff) --
PM4 mixes Phosphorus (rows A-E) and Sulfur (rows F-H) wells with different
negative controls, so a single threshold would misclassify wells that sit
between the two.

Confusion matrices classify a well's predicted_grows using the same
Growth/No-Growth threshold as the heatmaps (value < growth_threshold), and
compare it against the `aerobic_growth_call` field in
phenotypic_array_wells.json (fetched from EcoCyc): "Growth" and "Low
Growth" both count as "grows", "No Growth" counts as "no growth", and
"Indeterminate" or missing calls are excluded entirely.

Usage:
    uv run --env-file .env --project . python3 \
        "notebooks/Heena notebooks/Metabolism_New Genes/202607_plot_phenotypic_results.py"
"""

import argparse
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_CSV = SCRIPT_DIR / "out" / "phenotypic_arrays" / "results.csv"
DEFAULT_WELLS_JSON = SCRIPT_DIR / "phenotypic_array_wells.json"
DEFAULT_OUT_DIR = SCRIPT_DIR / "out" / "phenotypic_arrays" / "plots"
DEFAULT_SIM_FOLDER = "out/phenotypic/basal_min_weights_2500_2026-07-06/"

WELL_RE = re.compile(r"^([A-H])(\d{1,2})$")

# (plate, well) -> reference-line label. Each PM plate's own negative control
# removes a different nutrient category, so these are distinct conditions,
# not all the same as minimal-media "Basal" below.
NEG_CONTROL_WELLS = [
    ("PM1", "A1", "C Neg Control"),
    ("PM3", "A1", "N Neg Control"),
    ("PM4", "A1", "P Neg Control"),
    ("PM4", "F1", "S Neg Control"),
]

# (label, (plate, well) -> bool predicate, output filename, figure title). The
# same predicate is used both to select a df subset and to filter
# NEG_CONTROL_WELLS down to the reference lines relevant to that category.
# Predicates delegate to well_source_category (defined below) so there's one
# source of truth for the plate/well -> nutrient-source-category mapping.
SOURCE_CATEGORIES = [
    (
        "All Plates Combined",
        lambda plate, well: True,
        "combined_histogram.html",
        "All PM Plates Combined: Homeostatic Objective Distribution",
    ),
    (
        "Carbon",
        lambda plate, well: well_source_category(plate, well) == "Carbon",
        "carbon_histogram.html",
        "Carbon Sources: Homeostatic Objective Distribution",
    ),
    (
        "Nitrogen",
        lambda plate, well: well_source_category(plate, well) == "Nitrogen",
        "nitrogen_histogram.html",
        "Nitrogen Sources: Homeostatic Objective Distribution",
    ),
    (
        "Phosphorus",
        lambda plate, well: well_source_category(plate, well) == "Phosphorus",
        "phosphorus_histogram.html",
        "Phosphorus Sources: Homeostatic Objective Distribution",
    ),
    (
        "Sulfur",
        lambda plate, well: well_source_category(plate, well) == "Sulfur",
        "sulfur_histogram.html",
        "Sulfur Sources: Homeostatic Objective Distribution",
    ),
]

BIN_COLORS = ["#dda0dd", "#fdf4bf"]
VRECT_COLORS = {
    "Growth": "aqua",
    "No Growth": "yellow",
}
# Fixed per-source-category colors (Plotly's Pastel qualitative palette) so a
# given nutrient source is always the same color across every figure it
# appears in.
SOURCE_COLORS = {
    "Carbon": pc.qualitative.Pastel[0],
    "Nitrogen": pc.qualitative.Pastel[1],
    "Phosphorus": pc.qualitative.Pastel[2],
    "Sulfur": pc.qualitative.Pastel[3],
}

CONFUSION_CORRECT_COLOR = "#ddeee2"
CONFUSION_INCORRECT_COLOR = "#f8ddd8"
CONFUSION_BGCOLOR = "#e8e8e8"

GROWS_CALLS = {"Growth", "Low Growth"}
NO_GROWTH_CALLS = {"No Growth"}


def _fmt(x: float) -> str:
    """Compact number formatter for bin-edge labels."""
    if x == 0:
        return "0"
    if abs(x) >= 1e3 or abs(x) < 1e-2:
        return f"{x:.0E}"
    return f"{x:.4g}"


def load_run_phenotypic_arrays():
    """Import 202607_run_phenotypic_arrays.py as a module despite the space in
    its directory path (can't just `import` it normally)."""
    path = SCRIPT_DIR / "202607_run_phenotypic_arrays.py"
    spec = importlib.util.spec_from_file_location("run_phenotypic_arrays", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_basal_homeostatic_objective(rpa, sim_folder, objective_weights):
    """Re-run FBA with no uptake Add/Remove at all -- true minimal media,
    i.e. normal growth -- and return its homeostatic_objective."""
    folder = str(sim_folder)
    if not folder.endswith("/"):
        folder += "/"
    fba, _bulk, metabolism, _output = rpa.load_sim(folder)
    result = rpa.test_NetworkFlowModel(
        objective_weights, metabolism=metabolism, fba=fba
    )
    solution = result[6]
    return float(solution.homeostatic_term)


def parse_well(well):
    match = WELL_RE.match(well)
    if not match:
        raise ValueError(f"Unrecognized well label: {well!r}")
    return match.group(1), int(match.group(2))


def well_source_category(plate, well):
    """Map a (plate, well) to its nutrient-source category: PM1/PM2 ->
    Carbon, PM3 -> Nitrogen, PM4 -> Phosphorus or Sulfur depending on row
    (matches the SOURCE_CATEGORIES groupings). Returns None for anything
    else (e.g. the synthetic BASAL row)."""
    if plate in ("PM1", "PM2"):
        return "Carbon"
    if plate == "PM3":
        return "Nitrogen"
    if plate == "PM4":
        return "Phosphorus" if parse_well(well)[0] in "ABCDE" else "Sulfur"
    return None


def discrete_colorscale(bvals, colors):
    """bvals - boundary values delimiting len(colors) intervals of interest.
    Returns a plotly discrete colorscale."""
    if len(bvals) != len(colors) + 1:
        raise ValueError("len(boundary values) should be equal to len(colors)+1")
    bvals = sorted(bvals)
    nvals = [(v - bvals[0]) / (bvals[-1] - bvals[0]) for v in bvals]
    dcolorscale = []
    for k in range(len(colors)):
        dcolorscale.extend([[nvals[k], colors[k]], [nvals[k + 1], colors[k]]])
    return dcolorscale


def compute_growth_threshold(df, neg_controls=NEG_CONTROL_WELLS):
    """Growth/No-Growth cutoff = the LOWEST homeostatic_objective among the
    four negative controls (each removes a different nutrient category, so
    they don't all land at the same value). Using the lowest of the four
    means any well scoring above the "healthiest-looking" negative control is
    unambiguously called No Growth -- e.g. if the carbon neg control is 0.05
    and the nitrogen neg control is 0.04, everything above 0.04 is No Growth."""
    by_plate_well = df.set_index(["plate", "well"])["homeostatic_objective"]
    values = []
    for plate, well, _label in neg_controls:
        if (plate, well) in by_plate_well.index:
            value = by_plate_well.loc[(plate, well)]
            if pd.notna(value):
                values.append(float(value))
    if not values:
        raise ValueError("No negative control values found to compute growth threshold")
    return min(values)


def compute_category_thresholds(df, neg_controls=NEG_CONTROL_WELLS):
    """Growth/No-Growth cutoff PER nutrient-source category: each category's
    own negative control, rather than the single global minimum from
    compute_growth_threshold. PM4 mixes two categories on one plate
    (Phosphorus rows A-E, Sulfur rows F-H) with different negative controls,
    so judging every well against one plate-wide (or globally-lowest)
    threshold misclassifies any well that sits between its own category's
    negative control and the other category's (usually lower) one."""
    by_plate_well = df.set_index(["plate", "well"])["homeostatic_objective"]
    thresholds = {}
    for plate, well, _label in neg_controls:
        if (plate, well) not in by_plate_well.index:
            continue
        value = by_plate_well.loc[(plate, well)]
        if pd.notna(value):
            thresholds[well_source_category(plate, well)] = float(value)
    return thresholds


def value_to_bin_position(value, edges):
    """Map a raw value to an x-position on an equal-width-bin axis: the
    integer part is the bin index and the fractional part is the value's
    relative position within that bin's true (possibly very different)
    numeric span. This lets every bin render the same width on screen
    regardless of its true width -- e.g. a few narrow bins carved out
    around a region of interest don't get visually squeezed next to one
    wide catch-all bin. Extrapolates linearly past the first/last edge
    using that edge bin's width."""
    n_bins = len(edges) - 1
    i = int(np.clip(np.searchsorted(edges, value, side="right") - 1, 0, n_bins - 1))
    bin_width = edges[i + 1] - edges[i]
    frac = (value - edges[i]) / bin_width if bin_width else 0.0
    return i + frac


def build_combined_histogram(
    df,
    threshold,
    basal_value,
    title,
    neg_controls=NEG_CONTROL_WELLS,
    bins=None,
    nbins=50,
):
    """One histogram across every well in the given (sub)set of plates,
    stacked and colored by PM plate, with a dotted reference line per
    relevant plate's own negative control plus an optional minimal-media
    Basal line. Also annotates the single lowest homeostatic_objective well
    in this (sub)set, since with many bins it's otherwise not obvious which
    condition a given bar corresponds to.

    `threshold` is the Growth/No-Growth cutoff for the shaded bands -- pass
    the relevant category's own negative control (see
    compute_category_thresholds) so the shading agrees with the
    category-specific reference line already drawn on the chart, not some
    other category's (possibly lower) negative control.

    `bins` (explicit, possibly non-uniform bin edges) takes precedence over
    `nbins` (an equal-width bin count) when both are given. Bins are always
    rendered at equal width on screen regardless of their true numeric
    span (via value_to_bin_position) -- so a few deliberately narrow bins
    carved out around a region of interest read just as clearly as one wide
    catch-all bin -- with the true edge values shown as x-axis tick labels."""
    # Exclude the synthetic BASAL row (present in the "All Plates Combined"
    # subset) -- it's already drawn as its own dedicated reference line, not
    # a regular well to be counted/colored as part of a source's bars.
    df = df[df["plate"] != "BASAL"]
    df = df.assign(
        source=[well_source_category(p, w) for p, w in zip(df["plate"], df["well"])]
    )
    values = df["homeostatic_objective"].dropna().tolist()
    edges = np.asarray(
        sorted(bins)
        if bins is not None
        else np.histogram_bin_edges(values, bins=nbins),
        dtype=float,
    )
    n_bins = len(edges) - 1
    centers = np.arange(n_bins) + 0.5

    traces = []
    total_counts = np.zeros(n_bins)
    for source in sorted(df["source"].dropna().unique()):
        source_values = df.loc[df["source"] == source, "homeostatic_objective"].dropna()
        counts, _ = np.histogram(source_values, bins=edges)
        total_counts += counts
        traces.append(
            go.Bar(
                x=centers,
                y=counts,
                name=source,
                marker_color=SOURCE_COLORS.get(source, "lightblue"),
            )
        )
    max_count = max(total_counts.max(), 1)

    by_plate_well = df.set_index(["plate", "well"])["homeostatic_objective"]
    reference_lines = []
    for plate, well, label in neg_controls:
        if (plate, well) not in by_plate_well.index:
            continue
        value = by_plate_well.loc[(plate, well)]
        if pd.isna(value):
            continue
        reference_lines.append((float(value), label, "Navy"))
    if basal_value is not None:
        reference_lines.append((float(basal_value), "Basal", "red"))

    for value, _label, color in reference_lines:
        pos = value_to_bin_position(value, edges)
        traces.append(
            go.Scatter(
                x=[pos, pos],
                y=[0, max_count],
                mode="lines",
                line=dict(color=color, width=3, dash="dot"),
                showlegend=False,
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(barmode="stack")

    for value, label, _color in reference_lines:
        fig.add_annotation(
            x=value_to_bin_position(value, edges),
            y=max_count,
            xref="x",
            yref="y",
            text=f"{label}: {value:.4g}",
            textangle=-45,
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
        )

    band_edges = [
        value_to_bin_position(edges[0], edges),
        value_to_bin_position(threshold, edges),
        value_to_bin_position(edges[-1], edges),
    ]
    pad = 0.3
    band_names = ["Growth", "No Growth"]
    for k, name in enumerate(band_names):
        fig.add_vrect(
            x0=band_edges[k],
            x1=band_edges[k + 1],
            line_width=0,
            fillcolor=VRECT_COLORS[name],
            opacity=0.1,
            annotation_text=name,
            annotation_position="top",
            annotation_font_size=16,
            layer="below",
        )
    fig.update_xaxes(
        range=[band_edges[0] - pad, band_edges[-1] + pad],
        tickmode="array",
        tickvals=centers,
        ticktext=[
            f"[{_fmt(edges[i])}, {_fmt(edges[i + 1])})" for i in range(len(edges) - 1)
        ],
        tickangle=-45,
    )

    if not df["homeostatic_objective"].dropna().empty:
        min_idx = df["homeostatic_objective"].idxmin()
        min_row = df.loc[min_idx]
        fig.add_annotation(
            x=value_to_bin_position(min_row["homeostatic_objective"], edges),
            y=0.85,
            xref="x",
            yref="paper",
            text=(
                f"Lowest: {min_row['compound_name']}"
                f"<br>({min_row['plate']} {min_row['well']}, "
                f"{min_row['homeostatic_objective']:.4g})"
            ),
            bgcolor="white",
            bordercolor="black",
        )

    fig.update_layout(
        width=1000,
        height=800,
        paper_bgcolor="rgba(0,0,0,0)",
        title=title,
        xaxis_title="Homeostatic Objective",
        yaxis_title="Count",
        bargap=0.05,
        legend=dict(
            title="Sole Source",
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
        ),
    )
    return fig


def build_heatmap(plate, sub, growth_threshold):
    """Build the plate heatmap as a strictly binary Growth/No-Growth call
    per well, each judged one plate-wide cutoff -- lowest homeostatic
    objective amongst all negative controls"""
    rows = list("ABCDEFGH")
    cols = list(range(1, 13))
    by_well = sub.set_index("well")

    matrix, label = [], []
    for r in rows:
        row_data, row_text = [], []
        for c in cols:
            key = f"{r}{c}"
            if key in by_well.index:
                value = by_well.loc[key, "homeostatic_objective"]
                compound = by_well.loc[key, "compound_name"]
                text = (
                    "<br>".join(str(compound).split(" ")) if pd.notna(compound) else ""
                )
                if pd.notna(value):
                    text += f"<br>{value:.4g}"
                    # A well's own negative control is, by construction,
                    # exactly equal to the threshold -- classify ties as No
                    # Growth so a negative control never reads as "Growth".
                    row_data.append(
                        np.nan
                        if growth_threshold is None
                        else float(value >= growth_threshold)
                    )
                else:
                    row_data.append(np.nan)
                row_text.append(text)
            else:
                row_data.append(np.nan)
                row_text.append("")
        matrix.append(row_data)
        label.append(row_text)

    heatmap = go.Heatmap(
        z=matrix,
        zmin=0,
        zmax=1,
        x=[str(c) for c in cols],
        y=rows,
        text=label,
        texttemplate="%{text}",
        textfont={"size": 10},
        colorscale=discrete_colorscale([0, 0.5, 1], BIN_COLORS),
        colorbar=dict(
            thickness=25,
            tickvals=[0.25, 0.75],
            ticktext=[f"Growth  < {growth_threshold:0.5f}", "No Growth"],
        ),
    )
    fig = go.Figure(data=[heatmap])

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        title=f"{plate}: Plate Reader Heatmap (Homeostatic Objective)",
        xaxis_title="Column",
        yaxis_title="Row",
        yaxis_autorange="reversed",
    )
    return fig


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


def build_comparison_frame(results_df, ground_truth, growth_threshold):
    """Attach predicted_grows and experimental_grows columns, keeping only
    wells where both a valid prediction and valid ground truth exist.

    predicted_grows uses value < growth_threshold (strictly less than) to
    match build_heatmap(), which classifies a well exactly at its own
    negative control's value (i.e. value == growth_threshold) as No Growth,
    not Growth."""
    df = results_df.copy()
    df["experimental_call"] = df.apply(
        lambda row: ground_truth.get((row["plate"], row["well"])), axis=1
    )
    df["experimental_grows"] = df["experimental_call"].map(experimental_grows)
    df["predicted_grows"] = df["homeostatic_objective"].apply(
        lambda value: value < growth_threshold if pd.notna(value) else None
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
        colorscale=[[0, CONFUSION_INCORRECT_COLOR], [1, CONFUSION_CORRECT_COLOR]],
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
        paper_bgcolor=CONFUSION_BGCOLOR,
        plot_bgcolor=CONFUSION_BGCOLOR,
        width=560,
        height=560,
        margin=dict(t=80, b=40, l=120, r=40),
    )
    return fig


def build_by_source_figure(df):
    categories = ["Carbon", "Nitrogen", "Phosphorus", "Sulfur"]
    source = df.apply(
        lambda row: well_source_category(row["plate"], row["well"]), axis=1
    )

    counts_by_category = {}
    for category in categories:
        sub = df[source == category]
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
        paper_bgcolor=CONFUSION_BGCOLOR,
        plot_bgcolor=CONFUSION_BGCOLOR,
        width=1000,
        height=1000,
        margin=dict(t=100, b=40, l=80, r=40),
        showlegend=False,
    )
    return fig, counts_by_category


def write_figure(fig, out_dir, stem):
    """Write both an interactive out_dir/stem.html and a static
    out_dir/svg/stem.svg (kaleido) version of a figure."""
    html_path = out_dir / f"{stem}.html"
    fig.write_html(html_path)
    svg_dir = out_dir / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    svg_path = svg_dir / f"{stem}.svg"
    fig.write_image(svg_path)
    return html_path, svg_path


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
        help="Path to phenotypic_array_wells.json (EcoCyc ground-truth "
        "export), used for the confusion matrices",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory to write the histogram, heatmap, and confusion "
        "matrix HTML files",
    )
    parser.add_argument(
        "--basal-value",
        type=float,
        default=None,
        help="Manually supply the minimal-media homeostatic_objective instead "
        "of recomputing it live (skips loading the simulation checkpoint).",
    )
    parser.add_argument(
        "--skip-basal",
        action="store_true",
        help="Don't compute or draw the Basal (minimal media) reference line at all.",
    )
    parser.add_argument(
        "--sim-folder",
        default=DEFAULT_SIM_FOLDER,
        help="Simulation checkpoint folder used to compute the live Basal "
        "reference line. Must match whatever checkpoint produced "
        "--results-csv for the comparison to be meaningful.",
    )
    parser.add_argument(
        "--nbins",
        type=int,
        default=50,
        help="Number of histogram bins, default 50",
    )
    parser.add_argument(
        "--bins",
        type=lambda s: [float(x) for x in s.split(",")],
        default=None,
        help="Comma-separated list of explicit (possibly non-uniform) bin "
        'edges, e.g. --bins "0,0.5,1,1.5,2,3,5". Overrides --nbins.',
    )
    parser.add_argument(
        "--show", action="store_true", help="Also open each figure in a browser"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    growth_threshold = compute_growth_threshold(df, NEG_CONTROL_WELLS)
    print(
        f"Global Growth/No-Growth threshold (lowest negative control) = {growth_threshold}"
    )
    category_thresholds = compute_category_thresholds(df, NEG_CONTROL_WELLS)
    print(f"Per-category Growth/No-Growth thresholds = {category_thresholds}")

    basal_value = args.basal_value
    if basal_value is None and not args.skip_basal:
        # Prefer a "BASAL" row written directly into results.csv by
        # 202607_run_phenotypic_arrays.py (fast: no checkpoint reload) --
        # only fall back to a live re-solve for older results.csv files that
        # predate that row being added.
        basal_rows = df[df["plate"] == "BASAL"]
        if not basal_rows.empty and pd.notna(
            basal_rows.iloc[0]["homeostatic_objective"]
        ):
            basal_value = float(basal_rows.iloc[0]["homeostatic_objective"])
            print(f"Basal homeostatic_objective (from results.csv) = {basal_value}")
        else:
            print(
                f"No BASAL row in results.csv; computing live from {args.sim_folder} ..."
            )
            rpa = load_run_phenotypic_arrays()
            basal_value = compute_basal_homeostatic_objective(
                rpa, args.sim_folder, rpa.DEFAULT_OBJECTIVE_WEIGHTS
            )
            print(f"Basal homeostatic_objective = {basal_value}")

    for label, predicate, filename, title in SOURCE_CATEGORIES:
        mask = df.apply(lambda row: predicate(row["plate"], row["well"]), axis=1)
        sub = df[mask]
        neg_controls = [nc for nc in NEG_CONTROL_WELLS if predicate(nc[0], nc[1])]
        # "All Plates Combined" spans multiple categories in one shared band,
        # so it falls back to the single conservative global threshold; every
        # other (single-category) histogram uses its own category's negative
        # control, matching the heatmaps and the reference line already
        # drawn on that chart.
        threshold = category_thresholds.get(label, growth_threshold)
        print(f"{label}: {len(sub)} wells, threshold = {threshold}")
        hist_fig = build_combined_histogram(
            sub,
            threshold,
            basal_value,
            title,
            neg_controls,
            bins=args.bins,
            nbins=args.nbins,
        )
        stem = Path(filename).stem
        hist_path, hist_svg_path = write_figure(hist_fig, args.out_dir, stem)
        print(f"wrote {hist_path}")
        print(f"wrote {hist_svg_path}")
        if args.show:
            hist_fig.show()

    for plate in sorted(df["plate"].unique()):
        if plate == "BASAL":
            continue
        sub = df[df["plate"] == plate]
        if sub["homeostatic_objective"].dropna().empty:
            print(
                f"{plate}: no feasible homeostatic_objective values, skipping heatmap"
            )
            continue
        heatmap_fig = build_heatmap(plate, sub, growth_threshold)
        heatmap_path, heatmap_svg_path = write_figure(
            heatmap_fig, args.out_dir, f"{plate}_heatmap"
        )
        print(f"{plate}: wrote {heatmap_path}")
        print(f"{plate}: wrote {heatmap_svg_path}")
        if args.show:
            heatmap_fig.show()

    print(f"Predicted grows if homeostatic_objective < {growth_threshold}")
    ground_truth = load_ground_truth(args.wells_json)
    comparison_df = build_comparison_frame(df, ground_truth, growth_threshold)

    tp, fp, fn, tn = confusion_counts(comparison_df)
    n = len(comparison_df)
    print(f"Overall: n={n} wells with valid prediction and ground truth")
    print(f"Overall: TP={tp} FP={fp} FN={fn} TN={tn}")

    overall_fig = build_overall_figure(tp, fp, fn, tn, n)
    overall_path, overall_svg_path = write_figure(
        overall_fig, args.out_dir, "confusion_matrix_overall"
    )
    print(f"wrote {overall_path}")
    print(f"wrote {overall_svg_path}")
    if args.show:
        overall_fig.show()

    by_source_fig, counts_by_category = build_by_source_figure(comparison_df)
    for category, ((c_tp, c_fp, c_fn, c_tn), c_n) in counts_by_category.items():
        print(f"{category}: n={c_n} wells with valid prediction and ground truth")
        print(f"{category}: TP={c_tp} FP={c_fp} FN={c_fn} TN={c_tn}")

    by_source_path, by_source_svg_path = write_figure(
        by_source_fig, args.out_dir, "confusion_matrix_by_source"
    )
    print(f"wrote {by_source_path}")
    print(f"wrote {by_source_svg_path}")
    if args.show:
        by_source_fig.show()


if __name__ == "__main__":
    main()
