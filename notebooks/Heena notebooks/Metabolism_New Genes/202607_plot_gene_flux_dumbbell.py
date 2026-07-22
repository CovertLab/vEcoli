"""Plot a horizontal dumbbell chart of new-metabolic-gene usage: for each of
the ~307 genes added to the model in the 2022 metabolic-gene expansion, what
fraction of phenotype-microarray conditions gave that gene's enzyme(s) any
flux, comparing the OLD model (no 2022 reactions) against the NEW model
(with 2022 reactions).

Requires per-well base-reaction flux data captured by
202607_run_phenotypic_arrays.py --capture-fluxes (not present in the
regular results.csv), plus the accompanying reaction_catalysts/
fba_to_base_reactions side-car JSONs it writes alongside the flux parquet.

Gene -> reaction mapping is enzyme-mediated (matches the user's own prior
analysis in 20250616_test_carbon_source_cp3.ipynb cell 24, inherited from
20250307_track_reaction_usage.ipynb): each gene's "Enzyme encoded" value(s)
[new_metabolic_gene_annotation.csv] are looked up in reaction_catalysts
(reaction -> catalyzing enzyme ids) to find every fba-level reaction that
enzyme catalyzes, which is then collapsed to base reaction ids via
fba_reaction_ids_to_base_reaction_ids -- the same base-reaction id space the
flux parquet is indexed by.

A gene's flux is "present" in a well if any of its mapped base-reaction ids
has |flux| > --epsilon there (binary presence, not magnitude) -- this
directly answers "did adding the 2022 reactions get this gene used", the
question motivating the whole analysis; it also matches the truthiness-only
metric already used in the user's prior notebook precedent.

Note: a gene showing 0% flux in the OLD model is not necessarily because its
reactions are structurally missing from the OLD network -- most 2022-tagged
reactions are present in both checkpoints' base_reaction_ids (the OLD/NEW
gap mostly comes from which reactions the solver actually routes flux
through in each checkpoint's frozen catalyst/kinetic state, not from S-matrix
column presence/absence).

CAVEAT (checked 2026-07-13): re-solving the OLD checkpoint (--capture-fluxes)
to get flux data does NOT reproduce the already-published
results_no_new_reactions_original_weights.csv's homeostatic_objective --
every one of the 384 wells differs, many by 1.5-2.7 in absolute terms (e.g.
published oofv ~3.44 for several PM1 wells vs. ~0.7-1.4 on re-solve). This is
not floating-point noise (the NEW-model re-solve matches its own published
CSV to ~1e-12 on the same wells) -- something about the OLD checkpoint's
solve behavior has drifted, most likely from code changes to
metabolism_redux_classic.py/NetworkFlowModel between whenever that CSV was
generated and the current HEAD. This script's flux data (and therefore its
OLD-model usage fractions) reflects CURRENT-HEAD solve behavior, not
whatever produced the published OLD-model results.csv -- treat the OLD-model
side of this plot as a snapshot of current code, not as consistent with
prior OLD-model analyses, until that drift is root-caused.

With ~307 genes, the output SVG/HTML is intentionally a tall image -- this
matches the reference dumbbell-chart style, not a layout bug. By default
every gene is plotted; pass --top-n N to only plot the top N genes by
NEW-model fraction. The summary CSV always contains all genes regardless of
this flag.

Usage:
    uv run --env-file .env --project . python3 \
        "notebooks/Heena notebooks/Metabolism_New Genes/202607_plot_gene_flux_dumbbell.py"
"""

import argparse
import ast
import json
from pathlib import Path

import altair as alt
import pandas as pd
import plotly.colors as pc

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = SCRIPT_DIR / "out" / "phenotypic_arrays"
DEFAULT_PLOTS_DIR = DEFAULT_OUT_DIR / "plots"
DEFAULT_GENE_ANNOTATION_CSV = SCRIPT_DIR / "new_metabolic_gene_annotation.csv"

OLD_COLOR = "#b0b0b0"
NEW_COLOR = pc.qualitative.Pastel[2]
LINE_COLOR = "#dddddd"
OLD_LABEL = "Old model (no 2022 reactions)"
NEW_LABEL = "New model (with 2022 reactions)"


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


def _parse_enzyme_list(value):
    """Parse an "Enzyme encoded" cell: a bracket-list-string (e.g.
    "['PYRUVATEDEH-CPLX', 'E1P-CPLX']") -> list; a bare non-null string
    (e.g. "AAS-MONOMER") -> single-item list; NaN -> empty list."""
    if pd.isna(value):
        return []
    value = str(value).strip()
    if value.startswith("["):
        return ast.literal_eval(value)
    return [value]


def load_gene_to_base_reactions(gene_annotation_csv, reaction_catalysts, fba_to_base):
    """Map each new gene (keyed by "Gene ID (EcoCyc)") to the set of base
    reaction ids its enzyme(s) catalyze: gene -> enzyme(s) ["Enzyme
    encoded"] -> fba-level reactions [reaction_catalysts, already
    tag-stripped by --capture-fluxes] -> base reaction ids [fba_to_base].
    Returns (genes_df, gene_id -> set-of-base-reaction-ids)."""
    genes_df = pd.read_csv(gene_annotation_csv)
    genes_to_enzymes = genes_df.set_index("Gene ID (EcoCyc)")[
        "Enzyme encoded"
    ].to_dict()

    enzyme_to_fba_reactions = {}
    for rxn, enzymes in reaction_catalysts.items():
        for enzyme in enzymes:
            enzyme_to_fba_reactions.setdefault(enzyme, []).append(rxn)

    gene_to_base_reactions = {}
    for gene_id, enzyme_value in genes_to_enzymes.items():
        fba_reactions = set()
        for enzyme in _parse_enzyme_list(enzyme_value):
            fba_reactions.update(enzyme_to_fba_reactions.get(enzyme, []))
        gene_to_base_reactions[gene_id] = {fba_to_base.get(r, r) for r in fba_reactions}
    return genes_df, gene_to_base_reactions


def compute_has_flux_fraction(flux_df, gene_to_base_reactions, epsilon):
    """Per gene, the fraction of wells (rows of flux_df) where at least one
    of its mapped base-reaction columns has |flux| > epsilon. Reaction ids
    not present in this model's flux_df columns are dropped -- treated as
    no-flux by construction, not an error (a handful of base reaction ids
    differ between the OLD/NEW checkpoints' networks).

    Returns (gene_id -> fraction, gene_id -> n_mapped_reactions_present)."""
    columns = set(flux_df.columns)
    fractions, coverage = {}, {}
    for gene_id, base_reactions in gene_to_base_reactions.items():
        present = base_reactions & columns
        coverage[gene_id] = len(present)
        if not present:
            fractions[gene_id] = 0.0
            continue
        has_flux = (flux_df[list(present)].abs() > epsilon).any(axis=1)
        fractions[gene_id] = float(has_flux.mean())
    return fractions, coverage


def build_dumbbell_figure(result_df):
    """Horizontal dumbbell/lollipop chart: one row per gene, a grey dot for
    the OLD-model fraction and a colored dot for the NEW-model fraction,
    connected by a thin line. result_df is expected pre-sorted; genes are
    plotted top-to-bottom in that order (first row on top)."""
    labels = result_df["gene_label"].tolist()
    n_genes = len(result_df)
    height = max(600, 22 * n_genes)

    y_axis = alt.Y("gene_label:N", sort=labels, title="Gene")
    x_scale = alt.Scale(domain=[-0.02, 1.02])

    rules = (
        alt.Chart(result_df)
        .mark_rule(color=LINE_COLOR)
        .encode(
            y=y_axis,
            x=alt.X("fraction_old:Q", scale=x_scale),
            x2="fraction_new:Q",
        )
    )

    long_df = pd.melt(
        result_df,
        id_vars=["gene_label"],
        value_vars=["fraction_old", "fraction_new"],
        var_name="model",
        value_name="fraction",
    )
    long_df["model"] = long_df["model"].map(
        {"fraction_old": OLD_LABEL, "fraction_new": NEW_LABEL}
    )

    points = (
        alt.Chart(long_df)
        .mark_circle(size=80)
        .encode(
            y=y_axis,
            x=alt.X(
                "fraction:Q",
                scale=x_scale,
                title="Fraction of conditions with flux",
            ),
            color=alt.Color(
                "model:N",
                sort=[OLD_LABEL, NEW_LABEL],
                scale=alt.Scale(
                    domain=[OLD_LABEL, NEW_LABEL], range=[OLD_COLOR, NEW_COLOR]
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=["gene_label:N", "model:N", "fraction:Q"],
        )
    )

    chart = (
        (rules + points)
        .properties(
            width=400,
            height=height,
            title="New Metabolic Gene Usage: Fraction of PM Conditions With Flux (Old vs New Model)",
        )
        .configure_axis(labelLimit=200)
    )
    return chart


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-name",
        default="no_new_reactions_original_weights",
        help="--out-name used when running 202607_run_phenotypic_arrays.py "
        "--capture-fluxes for the OLD model (no 2022 reactions)",
    )
    parser.add_argument(
        "--new-name",
        default="new_reactions_original_weights",
        help="--out-name used when running 202607_run_phenotypic_arrays.py "
        "--capture-fluxes for the NEW model (with 2022 reactions)",
    )
    parser.add_argument(
        "--flux-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory containing the {name}_fluxes.parquet + "
        "{name}_reaction_catalysts.json + {name}_fba_to_base_reactions.json "
        "side-cars written by --capture-fluxes",
    )
    parser.add_argument(
        "--gene-annotation-csv",
        type=Path,
        default=DEFAULT_GENE_ANNOTATION_CSV,
        help="Path to new_metabolic_gene_annotation.csv (the ~307 new genes, "
        "with Gene ID (EcoCyc)/Gene name/Enzyme encoded columns)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-7,
        help="Absolute flux threshold above which a base reaction counts as "
        "'has flux' in a well, default 1e-7",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help="Directory to write the dumbbell chart HTML/SVG and a summary CSV",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Only plot the top N genes by NEW-model fraction (default: None, "
        "plot every gene). The summary CSV always contains all genes "
        "regardless of this flag.",
    )
    parser.add_argument(
        "--show", action="store_true", help="Also open the figure in a browser"
    )
    args = parser.parse_args()

    print(
        "CAVEAT: the OLD-model flux data behind this plot comes from a "
        "current-HEAD re-solve of the checkpoint, not from whatever "
        "produced results_no_new_reactions_original_weights.csv -- the two "
        "diverge substantially (see module docstring). Treat the OLD-model "
        "side of this plot as current-code behavior, not as consistent "
        "with prior OLD-model analyses."
    )

    old_flux_df = pd.read_parquet(args.flux_dir / f"{args.old_name}_fluxes.parquet")
    new_flux_df = pd.read_parquet(args.flux_dir / f"{args.new_name}_fluxes.parquet")
    print(
        f"OLD flux matrix: {old_flux_df.shape[0]} wells x {old_flux_df.shape[1]} reactions"
    )
    print(
        f"NEW flux matrix: {new_flux_df.shape[0]} wells x {new_flux_df.shape[1]} reactions"
    )

    with open(args.flux_dir / f"{args.old_name}_reaction_catalysts.json") as handle:
        old_reaction_catalysts = json.load(handle)
    with open(args.flux_dir / f"{args.old_name}_fba_to_base_reactions.json") as handle:
        old_fba_to_base = json.load(handle)
    with open(args.flux_dir / f"{args.new_name}_reaction_catalysts.json") as handle:
        new_reaction_catalysts = json.load(handle)
    with open(args.flux_dir / f"{args.new_name}_fba_to_base_reactions.json") as handle:
        new_fba_to_base = json.load(handle)

    genes_df, gene_to_base_old = load_gene_to_base_reactions(
        args.gene_annotation_csv, old_reaction_catalysts, old_fba_to_base
    )
    _, gene_to_base_new = load_gene_to_base_reactions(
        args.gene_annotation_csv, new_reaction_catalysts, new_fba_to_base
    )

    fraction_old, coverage_old = compute_has_flux_fraction(
        old_flux_df, gene_to_base_old, args.epsilon
    )
    fraction_new, coverage_new = compute_has_flux_fraction(
        new_flux_df, gene_to_base_new, args.epsilon
    )

    gene_names = genes_df.set_index("Gene ID (EcoCyc)")["Gene name"].to_dict()
    gene_ids = list(gene_to_base_new.keys())
    result_df = pd.DataFrame(
        {
            "gene_id": gene_ids,
            "gene_name": [gene_names.get(g) for g in gene_ids],
            "fraction_old": [fraction_old.get(g, 0.0) for g in gene_ids],
            "fraction_new": [fraction_new.get(g, 0.0) for g in gene_ids],
            "n_reactions_old": [coverage_old.get(g, 0) for g in gene_ids],
            "n_reactions_new": [coverage_new.get(g, 0) for g in gene_ids],
        }
    )
    result_df["gene_label"] = result_df["gene_name"].fillna(result_df["gene_id"])

    n_covered_new = (result_df["n_reactions_new"] > 0).sum()
    n_covered_old = (result_df["n_reactions_old"] > 0).sum()
    print(
        f"{n_covered_new}/{len(result_df)} genes have >=1 mapped reaction present "
        "in the NEW model's base_reaction_ids universe"
    )
    print(
        f"{n_covered_old}/{len(result_df)} genes have >=1 mapped reaction present "
        "in the OLD model's base_reaction_ids universe"
    )

    result_df = result_df.sort_values("fraction_new", ascending=False).reset_index(
        drop=True
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "gene_flux_usage.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")

    if args.top_n is None:
        plot_df = result_df
    else:
        plot_df = result_df.head(args.top_n).reset_index(drop=True)
    print(f"plotting {len(plot_df)}/{len(result_df)} genes (top_n={args.top_n})")

    chart = build_dumbbell_figure(plot_df)
    fig_path, svg_path = write_altair_figure(chart, args.out_dir, "gene_flux_dumbbell")
    print(f"wrote {fig_path}")
    print(f"wrote {svg_path}")
    if args.show:
        chart.show()


if __name__ == "__main__":
    main()
