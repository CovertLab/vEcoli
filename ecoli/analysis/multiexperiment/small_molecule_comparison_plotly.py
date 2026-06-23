"""
Multiexperiment comparison plotly scatter

Plots the total small molecule counts from two simulation experiments against
each other. Note this plot is very similar to the multiseed scatter, but
compares two different experiments instead of the free vs. total counts from
the same experiment.

Produces TWO HTML plots:
  1. <name>_<exp1>_vs_<exp2>.html: Plots all small molecules by how they are
    sequestered within other molecules in the simulation
     (category determined by Experiment 1's sequestration state, which ideally
     should be identical to Experiment 2 if nothing has changed in the
     complexation reactions or actively modeled transcription factors that use
     small molecules between the simulations).
  2. <name>_highlighted_<exp1>_vs_<exp2>.html: all small molecules are colored
     the same except for a user-specified list of small molecules.

Since there are a large number of small molecules that will have count values
of zero in most simulations, this plot chooses which molecules to plot by finding
species that are nonzero in either experiment and appear across both
experiments' tracked-species lists (in case the two ParCas differ
because new molecules are added to one simulation but not the other).

Both are log-log scatters with:
  - X-axis: average total count from Sim 1, Y-axis: average total count from Sim 2
  - y=x reference line (indicating equal counts between experiments)
  - hover text with the full free/sequestration breakdown for both experiments
  - +1 pseudocount added to all values so species that if the counts are zero in one
    experiment (but nonzero in the other), they will still appear on the log axes.

Per-experiment averaging reuses the same in-DB helpers as the multiseed scatter
(read_avg_listener_list / read_avg_free_from_bulk / compute_avg_sequestration),
each fed an experiment-filtered history subquery.

Params (all optional):
    {
        # Initial generations to skip from every average (default 0)
        "skip_n_gens": int,
        # Small molecules to highlight in plot 2 (default ["ATP", "Pi"])
        "highlight_metabolites": list[str],
    }
"""

import os
import pickle
from typing import Any

import numpy as np
import plotly.graph_objects as go
from duckdb import DuckDBPyConnection

from ecoli.library.parquet_emitter import (
    field_metadata,
    open_arbitrary_sim_data,
)
from ecoli.library.small_molecule_sequestration import compute_avg_sequestration
from ecoli.library.small_molecule_analysis import (
    count_cells,
    read_avg_free_from_bulk,
    read_avg_listener_list,
    resolve_highlights,
)

# PLOT DEFAULTS:
# Default highlight list if "highlight_molecules" is not given in params:
DEFAULT_HIGHLIGHT_MOLECULES = ["ATP", "Pi"]

# 8-way categorization (key, color, label, marker_size, opacity), keyed off the
# Sim 1 sequestration state (mirrors the multiseed scatter's CATEGORY_COLORS):
CATEGORY_SPECS = [
    ("none", "lightseagreen", "No sequestration", 5, 0.45),
    ("eq", "darkorange", "In eq complex", 7, 0.8),
    ("tcs", "magenta", "In TCS", 7, 0.8),
    ("btf", "royalblue", "In DNA-bound TF", 7, 0.8),
    ("eq_tcs", "mediumpurple", "In eq + TCS", 9, 0.9),
    ("eq_btf", "green", "In eq + bound TF", 9, 0.9),
    ("tcs_btf", "crimson", "In TCS + bound TF", 9, 0.9),
    ("eq_tcs_btf", "black", "In eq + TCS + bound TF", 9, 0.95),
]


def load_experiment(conn, history_sql, config_sql, sim_data_dict, exp_id, skip_n_gens):
    """Load average totals + sequestration breakdown for one experiment.

    Returns a dict with keys: sm_ids, avg_total, avg_free, avg_eq, avg_tcs_pi,
    avg_tcs_cplx, avg_btf, n_cells. Returns None if the experiment has no cells.
    """
    listener_col = "listeners__small_molecule_counts__totalSmallMoleculeCounts"

    exp_hist = f"SELECT * FROM ({history_sql}) WHERE experiment_id = '{exp_id}'"
    if skip_n_gens:
        exp_hist = f"SELECT * FROM ({exp_hist}) WHERE generation >= {skip_n_gens}"
    exp_conf = f"SELECT * FROM ({config_sql}) WHERE experiment_id = '{exp_id}'"

    sm_ids = field_metadata(conn, exp_conf, listener_col)

    n_cells = count_cells(conn, exp_hist)
    if n_cells == 0:
        return None

    avg_total = read_avg_listener_list(conn, exp_hist, listener_col)

    with open_arbitrary_sim_data({exp_id: sim_data_dict[exp_id]}) as f:
        sim_data = pickle.load(f)

    avg_free = read_avg_free_from_bulk(conn, exp_hist, sim_data, sm_ids)
    seq = compute_avg_sequestration(conn, exp_hist, sim_data, sm_ids)

    return {
        "sm_ids": np.array(sm_ids),
        "avg_total": avg_total,
        "avg_free": avg_free,
        "avg_eq": seq["avg_eq"],
        "avg_tcs_pi": seq["avg_tcs_pi"],
        "avg_tcs_cplx": seq["avg_tcs_complex"],
        "avg_btf": seq["avg_bound_tf"],
        "n_cells": n_cells,
    }


def categorize(eq_bound, tcs_bound, btf):
    """Return an array of category keys (one per species)."""
    has_eq = eq_bound > 0
    has_tcs = tcs_bound > 0
    has_btf = btf > 0
    keys = np.empty(len(eq_bound), dtype=object)
    for i in range(len(eq_bound)):
        e, t, b = has_eq[i], has_tcs[i], has_btf[i]
        if e and t and b:
            keys[i] = "eq_tcs_btf"
        elif e and t:
            keys[i] = "eq_tcs"
        elif e and b:
            keys[i] = "eq_btf"
        elif t and b:
            keys[i] = "tcs_btf"
        elif e:
            keys[i] = "eq"
        elif t:
            keys[i] = "tcs"
        elif b:
            keys[i] = "btf"
        else:
            keys[i] = "none"
    return keys


def make_hover_comparison(ids, total1_raw, total2_raw, data1, data2, label1, label2):
    """Hover text showing both experiments' breakdown (note it shows the raw
    counts, not psuedo-count that is calculated to help the plot's appearance)."""
    texts = []
    for i in range(len(ids)):
        bound_total1 = total1_raw[i] - data1["avg_free"][i]
        tcs1 = data1["avg_tcs_pi"][i] + data1["avg_tcs_cplx"][i]
        frac1 = data1["avg_free"][i] / total1_raw[i] if total1_raw[i] > 0 else 0

        bound_total2 = total2_raw[i] - data2["avg_free"][i]
        tcs2 = data2["avg_tcs_pi"][i] + data2["avg_tcs_cplx"][i]
        frac2 = data2["avg_free"][i] / total2_raw[i] if total2_raw[i] > 0 else 0

        if total1_raw[i] > 0:
            ratio_text = (
                f"<b>Ratio (Sim 2/Sim 1):</b> {total2_raw[i] / total1_raw[i]:.3f}"
            )
        else:
            ratio_text = "<b>Ratio:</b> undefined (Sim 1 = 0)"

        texts.append(
            f"<b>{ids[i]}</b><br><br>"
            f"<b>{label1}:</b><br>"
            f"  Avg. total: {total1_raw[i]:.1f}<br>"
            f"  Avg. free: {data1['avg_free'][i]:.1f}<br>"
            f"  Avg. bound (total): {bound_total1:.1f}<br>"
            f"    in equilibrium complexes: {data1['avg_eq'][i]:.1f}<br>"
            f"    in TCS: {tcs1:.1f}<br>"
            f"    in DNA-bound TFs: {data1['avg_btf'][i]:.1f}<br>"
            f"  Fraction free: {frac1:.3f}<br><br>"
            f"<b>{label2}:</b><br>"
            f"  Avg. total: {total2_raw[i]:.1f}<br>"
            f"  Avg. free: {data2['avg_free'][i]:.1f}<br>"
            f"  Avg. bound (total): {bound_total2:.1f}<br>"
            f"    in equilibrium complexes: {data2['avg_eq'][i]:.1f}<br>"
            f"    in TCS: {tcs2:.1f}<br>"
            f"    in DNA-bound TFs: {data2['avg_btf'][i]:.1f}<br>"
            f"  Fraction free: {frac2:.3f}<br><br>"
            f"{ratio_text}"
        )
    return texts


def _square_loglog_layout(fig, title, xlabel, ylabel):
    """Set up a square log-log layout with the given title and axis labels"""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=11),
            x=0.5,
            xanchor="center",
            y=0.97,
            yanchor="top",
        ),
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=750,
        height=860,
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.18,
            yanchor="top",
            font=dict(size=11),
        ),
        margin=dict(l=80, r=80, t=190, b=120),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(
        type="log",
        showgrid=True,
        gridcolor="lightgrey",
        linecolor="black",
        mirror=True,
    )
    fig.update_yaxes(
        type="log",
        scaleanchor="x",
        scaleratio=1,
        showgrid=True,
        gridcolor="lightgrey",
        linecolor="black",
        mirror=True,
    )


def build_categorized_figure(
    ids,
    total1_plot,
    total2_plot,
    total1_raw,
    total2_raw,
    data1,
    data2,
    label1,
    label2,
    title,
):
    """Scatter colored by Sim 1's sequestration category."""
    if len(ids) == 0:
        return None

    tcs_bound1 = data1["avg_tcs_pi"] + data1["avg_tcs_cplx"]
    cat_keys = categorize(data1["avg_eq"], tcs_bound1, data1["avg_btf"])
    hover = make_hover_comparison(
        ids, total1_raw, total2_raw, data1, data2, label1, label2
    )

    all_counts = np.concatenate([total1_plot, total2_plot])
    line_x = np.array([float(all_counts.min()) * 0.8, float(all_counts.max()) * 1.25])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_x,
            mode="lines",
            line=dict(color="black", width=1, dash="dash"),
            name="y=x",
            showlegend=True,
        )
    )

    for key, color, name, size, opacity in CATEGORY_SPECS:
        mask = cat_keys == key
        if mask.sum() == 0:
            continue
        hover_subset = [hover[i] for i in np.where(mask)[0]]
        fig.add_trace(
            go.Scatter(
                x=total1_plot[mask],
                y=total2_plot[mask],
                mode="markers",
                marker=dict(size=size, color=color, opacity=opacity),
                hovertext=hover_subset,
                hoverinfo="text",
                name=f"{name} ({int(mask.sum())})",
            )
        )

    _square_loglog_layout(
        fig,
        title,
        f"{label1}: Log Average Total Count + 1",
        f"{label2}: Log Average Total Count + 1",
    )
    return fig


def build_highlighted_figure(
    ids,
    total1_plot,
    total2_plot,
    total1_raw,
    total2_raw,
    data1,
    data2,
    highlight_mask,
    label1,
    label2,
    title,
):
    """Scatter with highlighted species in red and all others in lightseagreen."""
    if len(ids) == 0:
        return None

    hover = make_hover_comparison(
        ids, total1_raw, total2_raw, data1, data2, label1, label2
    )

    all_counts = np.concatenate([total1_plot, total2_plot])
    line_x = np.array([float(all_counts.min()) * 0.8, float(all_counts.max()) * 1.25])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_x,
            mode="lines",
            line=dict(color="black", width=1, dash="dash"),
            name="y = x",
            showlegend=True,
        )
    )

    bg = ~highlight_mask
    if bg.sum() > 0:
        fig.add_trace(
            go.Scatter(
                x=total1_plot[bg],
                y=total2_plot[bg],
                mode="markers",
                marker=dict(size=5, color="lightseagreen", opacity=0.45),
                hovertext=[hover[i] for i in np.where(bg)[0]],
                hoverinfo="text",
                name=f"Small molecules ({int(bg.sum())})",
            )
        )

    hl = highlight_mask
    if hl.sum() > 0:
        fig.add_trace(
            go.Scatter(
                x=total1_plot[hl],
                y=total2_plot[hl],
                mode="markers",
                marker=dict(
                    size=11,
                    color="red",
                    opacity=0.95,
                    line=dict(width=1, color="black"),
                ),
                hovertext=[hover[i] for i in np.where(hl)[0]],
                hoverinfo="text",
                name=f"Highlighted ({int(hl.sum())})",
            )
        )

    _square_loglog_layout(
        fig,
        title,
        f"{label1}: Log Average Total Count + 1",
        f"{label2}: Log Average Total Count + 1",
    )
    return fig


def plot(
    params: dict[str, Any],
    conn: DuckDBPyConnection,
    history_sql: str,
    config_sql: str,
    success_sql: str,
    sim_data_dict: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_names: dict[str, str],
):
    skip_n_gens = params.get("skip_n_gens", 0)
    highlight_list = params.get("highlight_molecules", DEFAULT_HIGHLIGHT_MOLECULES)

    # Determine the two experiments to compare:
    present = set(
        conn.sql(f"SELECT DISTINCT experiment_id FROM ({history_sql})")
        .pl()["experiment_id"]
        .to_list()
    )
    exp_ids = [e for e in sim_data_dict.keys() if e in present]
    if len(exp_ids) < 2:
        raise ValueError(
            f"Expected 2 experiments but found {len(exp_ids)}: {exp_ids}. "
            f"Make sure both experiment_ids are in the analysis config."
        )
    exp1, exp2 = exp_ids[0], exp_ids[1]
    label1, label2 = f"Sim 1 ({exp1})", f"Sim 2 ({exp2})"

    print(f"Loading data from {label1} ...")
    data1 = load_experiment(
        conn, history_sql, config_sql, sim_data_dict, exp1, skip_n_gens
    )
    if data1 is None:
        print(f"No cells found in {label1}, skipping plot.")
        return

    print(f"Loading data from {label2} ...")
    data2 = load_experiment(
        conn, history_sql, config_sql, sim_data_dict, exp2, skip_n_gens
    )
    if data2 is None:
        print(f"No cells found in {label2}, skipping plot.")
        return

    # Align on the species tracked by both experiments:
    ids1 = list(data1["sm_ids"])
    ids2 = list(data2["sm_ids"])
    common_ids = sorted(set(ids1) & set(ids2))
    if not common_ids:
        print("No common small molecules between the two experiments, skipping.")
        return

    only_in_1 = len(set(ids1) - set(ids2))
    only_in_2 = len(set(ids2) - set(ids1))
    print(f"Common small molecules: {len(common_ids)}")
    if only_in_1:
        print(f"  {only_in_1} only in {label1}")
    if only_in_2:
        print(f"  {only_in_2} only in {label2}")

    pos1 = {m: i for i, m in enumerate(ids1)}
    pos2 = {m: i for i, m in enumerate(ids2)}
    idx1 = [pos1[m] for m in common_ids]
    idx2 = [pos2[m] for m in common_ids]

    ids = np.array(common_ids)
    total1_raw = data1["avg_total"][idx1]
    total2_raw = data2["avg_total"][idx2]

    aligned1 = {
        k: data1[k][idx1]
        for k in ("avg_free", "avg_eq", "avg_tcs_pi", "avg_tcs_cplx", "avg_btf")
    }
    aligned2 = {
        k: data2[k][idx2]
        for k in ("avg_free", "avg_eq", "avg_tcs_pi", "avg_tcs_cplx", "avg_btf")
    }

    # Plot the union of species nonzero in either experiment:
    nonzero1 = total1_raw > 0
    nonzero2 = total2_raw > 0
    plot_mask = nonzero1 | nonzero2

    n_zero_both = int((~plot_mask).sum())
    n_zero_1_only = int((~nonzero1 & nonzero2).sum())
    n_zero_2_only = int((nonzero1 & ~nonzero2).sum())
    n_nonzero_both = int((nonzero1 & nonzero2).sum())
    print(
        f"\nNonzero in both: {n_nonzero_both}\nZero in {label1} only: "
        f"{n_zero_1_only} \nZero in {label2} only: {n_zero_2_only} "
        f"\nZero in both (not plotted): {n_zero_both}"
    )

    ids_plot = ids[plot_mask]
    total1_raw_p = total1_raw[plot_mask]
    total2_raw_p = total2_raw[plot_mask]
    data1_p = {k: v[plot_mask] for k, v in aligned1.items()}
    data2_p = {k: v[plot_mask] for k, v in aligned2.items()}

    # +1 pseudocount so species zero in one experiment still plot on log axes:
    total1_plot = total1_raw_p + 1
    total2_plot = total2_raw_p + 1

    # Shared title lines: map the short Sim 1/Sim 2 tags to the full names once,
    # then the per-experiment counts and the zero/nonzero breakdown:
    exp_legend = (
        f"<sub>Sim 1 = {exp1} ({data1['n_cells']} cells)</sub><br>"
        f"<sub>Sim 2 = {exp2} ({data2['n_cells']} cells)</sub><br>"
        f"<sub>Nonzero species in both: {n_nonzero_both}</sub><br>"
        f"<sub>Zero in both (not plotted): {n_zero_both}</sub><br>"
        f"<sub>{skip_n_gens} generations skipped in each simulation</sub><br>"
    )

    # Plot 1: molecules categorized by sequestration type
    title_cat = f"<b>Average Total Small Molecule Counts Comparison</b><br>{exp_legend}"
    fig_cat = build_categorized_figure(
        ids_plot,
        total1_plot,
        total2_plot,
        total1_raw_p,
        total2_raw_p,
        data1_p,
        data2_p,
        "Sim 1",
        "Sim 2",
        title_cat,
    )
    if fig_cat is not None:
        name = f"small_molecule_comparison_plotly_{exp1}_vs_{exp2}.html"
        fig_cat.write_html(os.path.join(outdir, name))
        print(f"\nSaved categorized plot: {name}")

    # Plot 2: highlighted small molecules of interest
    # Print messages about the compartment tag options:
    highlight_set, messages = resolve_highlights(highlight_list, list(ids_plot))
    print("\nCOMPARTMENT TAG MESSAGES:")
    for msg in messages:
        print(msg)
    if not messages:
        print("No messages.")

    # Print messages about the changes between the highlight_list and highlight_set:
    untagged_highlight_list = [h[:-3] if "]" in h else h for h in highlight_list]
    untagged_highlight_set = [h[:-3] if "]" in h else h for h in highlight_set]

    if untagged_highlight_set != set(untagged_highlight_list):
        all_ids_untagged = set([h[:-3] if "]" in h else h for h in ids])
        print("\nHIGHLIGHT LIST UPDATES:")
        print(f"  Original highlight list: {highlight_list}")
        print(f"  Resolved highlight set: {sorted(highlight_set)}")
        for sm_id in untagged_highlight_list:
            if sm_id not in untagged_highlight_set:
                if sm_id in all_ids_untagged:
                    print(
                        f"  - '{sm_id}' was removed from the highlight set "
                        f"because it either had zero counts across both "
                        f"simulations or the compartment tag passed through "
                        f"was not valid (see messages above for valid compartment tags)."
                    )
                else:
                    print(
                        f"  - '{sm_id}' was removed from the highlight set "
                        f"because it is likely not tracked tracked in either "
                        f"simulation."
                    )
            else:
                continue

    else:
        print(
            "No updates had to be made to the user specified the small molecule highlight list, all molecules passed through have nonzero counts in"
            " at least one simulation and have valid compartment tags."
        )

    highlight_mask = np.array([sid in highlight_set for sid in ids_plot])

    title_hi = f"<b>Total Small Molecule Counts Comparison</b><br>{exp_legend}"
    fig_hi = build_highlighted_figure(
        ids_plot,
        total1_plot,
        total2_plot,
        total1_raw_p,
        total2_raw_p,
        data1_p,
        data2_p,
        highlight_mask,
        "Sim 1",
        "Sim 2",
        title_hi,
    )
    if fig_hi is not None:
        name = f"small_molecule_comparison_plotly_highlighted_{exp1}_vs_{exp2}.html"
        fig_hi.write_html(os.path.join(outdir, name))
        print(f"Saved highlighted plot: {name}")
