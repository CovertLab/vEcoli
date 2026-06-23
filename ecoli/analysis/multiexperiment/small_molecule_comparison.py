"""
Multiexperiment comparison plot

Compares two simulation runs for a list of small molecules (each plotted
separately), using the SmallMoleculeCounts listener's
``totalSmallMoleculeCounts`` column.

Some molecules are listed to be plotted by default below, but the list can be
overridden by providing a "metabolites" key in params. Each listed molecule can
be a "bare" ID like 'ATP' (which matches every compartment variant present in
the metadata, e.g. 'ATP[c]', 'ATP[e]', 'ATP[p]') or a "tagged" ID, like 'ATP[c]',
which will result in the plot for only that one compartment to be plotted.

For each small-molecule species, the figure produces 8 plots:
  Left plots (4 stacked time-series panels):
    1. Experiment 1's total counts over time
    2. Experiment 2's total counts over time
    3. Experiment 1's rate-of-change of total counts
    4. Experiment 2's rate-of-change of total counts
    NOTE: the ROC excludes dilution steps at each cell division, so there are
    breaks at each generation boundary.
  Right plots (4 histograms corresponding to the time-series panel on its left):
    - per-cell average total count
    - per-cell average ROC
    In each sub-axis the per-cell values are histogrammed per seed plus a
    thin black "composite" outline over all seeds combined. The legend reports
    the number of generations per seed and the composite total. NOTE: since
    the first few generations of each seed may be dominated by startup effects,
    the histograms can skip the first N generations of each seed (default N=2,
    configurable via ``skip_n_gens`` in params). The time-series panels always
    show every generation.
"""

import os
from typing import Any

import numpy as np
from duckdb import DuckDBPyConnection
import matplotlib.pyplot as plt
from matplotlib import gridspec

from ecoli.library.parquet_emitter import field_metadata, read_stacked_columns

# DEFAULTS:

# Default small molecules to plot (can be overridden by params["molecules"]):
PLOT_MOLECULES = ["ATP", "Pi"]

# Seed colors (shared by the left panels and the histograms):
SEED_PALETTE = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:olive",
    "tab:cyan",
    "gold",
]


def resolve_species(sm_id: str, sm_ids: list[str]) -> list[str]:
    """Expand a bare or tagged ID to the species present in sm_ids."""
    if "[" in sm_id:
        return [sm_id] if sm_id in sm_ids else []
    prefix = sm_id + "["
    return [m for m in sm_ids if m.startswith(prefix)]


def cell_roc(time: np.ndarray, counts: np.ndarray):
    """Compute rate of change (ROC) in counts within one cell.

    NOTE: ROC is only calculated within the cell, so the dilution halving at
    division is never computed (i.e. the last step of the previous generation and
    the first step of the new generation are in different cells). Every
    consecutive within-cell step is kept: the first ROC is counts[1]-counts[0]
    (at time[1]) and the last is counts[-1]-counts[-2] (at time[-1]). Returns
    (roc_time, roc). Needs >=2 timepoints to yield any output.
    """
    if len(counts) < 2:
        return time[:0], counts[:0]
    dt = np.diff(time)
    # guard against zero dt within a cell:
    dt = np.where(dt == 0, np.nan, dt)
    # length T-1, aligned to time[1:]:
    roc = np.diff(counts) / dt
    return time[1:], roc


def load_species_cells(
    conn: DuckDBPyConnection, history_sql: str, idx: int
) -> dict[str, list[dict]]:
    """Loads cell time series for one species column, grouped by experiment.

    Returns a dict experiment_id -> list of per-cell dicts with keys
    ``lineage_seed``, ``generation``, ``time``, ``counts``, ``roc_time``,
    ``roc``. DuckDB lists are 1-indexed, so element ``idx + 1`` is read.
    """
    listener_col = "listeners__small_molecule_counts__totalSmallMoleculeCounts"

    subquery = read_stacked_columns(
        history_sql,
        [f"{listener_col}[{idx + 1}] AS total"],
        order_results=False,
    )
    data = conn.sql(
        f"""
        SELECT experiment_id, lineage_seed, generation, agent_id, time, total
        FROM ({subquery})
        ORDER BY experiment_id, lineage_seed, generation, agent_id, time
        """
    ).pl()

    out: dict[str, list[dict]] = {}
    group_cols = ["experiment_id", "lineage_seed", "generation", "agent_id"]
    for (exp_id, seed, gen, _agent), cell in data.group_by(
        group_cols, maintain_order=True
    ):
        time = cell["time"].to_numpy().astype(float)
        counts = cell["total"].to_numpy().astype(float)
        order = np.argsort(time)
        time, counts = time[order], counts[order]
        roc_time, roc = cell_roc(time, counts)
        out.setdefault(exp_id, []).append(
            dict(
                lineage_seed=int(seed),
                generation=int(gen),
                time=time,
                counts=counts,
                roc_time=roc_time,
                roc=roc,
            )
        )
    return out


def group_by_seed(cells: list[dict]) -> list[tuple[int, list[dict]]]:
    """Organizes a flat list of per-cell dicts into [(seed, [gens...]), ...]"""
    by_seed: dict[int, list[dict]] = {}
    for c in cells:
        by_seed.setdefault(c["lineage_seed"], []).append(c)
    out = []
    for seed in sorted(by_seed):
        gens = sorted(by_seed[seed], key=lambda g: g["generation"])
        out.append((seed, gens))
    return out


def _concat(gens, time_key, val_key):
    """Concatenates per-generation arrays into one continuous trace for plotting"""
    good = [g for g in gens if len(g[val_key]) > 0]
    if not good:
        return np.array([]), np.array([])
    return (
        np.concatenate([g[time_key] for g in good]),
        np.concatenate([g[val_key] for g in good]),
    )


def _concat_breaks(gens, time_key, val_key):
    """Concatenates per-generation arrays with a NaN gap between generations so
    the plotted line breaks at each division boundary."""
    good = [g for g in gens if len(g[val_key]) > 0]
    if not good:
        return np.array([]), np.array([])
    ts, vs = [], []
    for k, g in enumerate(good):
        if k > 0:
            ts.append(np.array([np.nan]))
            vs.append(np.array([np.nan]))
        ts.append(g[time_key])
        vs.append(g[val_key])
    return np.concatenate(ts), np.concatenate(vs)


def _seed_color(i):
    return SEED_PALETTE[i % len(SEED_PALETTE)]


# Plotting functions:
def _plot_timeseries_panel(ax, exp_data, value_key, label, ylabel):
    """Plot one panel: every seed overlaid in its own color. ROC is
    drawn with a break at each generation boundary."""
    tkey = "time" if value_key == "counts" else "roc_time"
    concat = _concat if value_key == "counts" else _concat_breaks
    for i, (_seed, gens) in enumerate(exp_data):
        t, v = concat(gens, tkey, value_key)
        if len(t):
            ax.plot(
                t,
                v,
                color=_seed_color(i),
                alpha=0.6,
                linewidth=0.8,
                label=f"seed {i + 1}",
            )
    ax.set_title(label, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)


def _per_seed_cell_values(exp_data, value_key, skip_gens):
    """Lists of per-cell time-averages (one value per generation,
    skipping early generations) for each seed. Returns [(seed_index, [values...]), ...]."""
    out = []
    for i, (_seed, gens) in enumerate(exp_data):
        vals = [
            float(np.nanmean(g[value_key]))
            for g in gens
            if g["generation"] >= skip_gens and len(g[value_key]) > 0
        ]
        vals = [v for v in vals if not np.isnan(v)]
        out.append((i, vals))
    return out


def _plot_hist_one(ax, exp_data, value_key, skip_gens, bins, exp_label):
    """Generates overlaid histogram for each seed in one experiment and a
    composite outline over all seeds combined."""
    per_seed = _per_seed_cell_values(exp_data, value_key, skip_gens)
    all_vals = [v for _i, vals in per_seed for v in vals]
    if not all_vals:
        ax.set_title(f"{exp_label}: no data (skip={skip_gens})", fontsize=8)
        ax.tick_params(labelsize=7)
        return
    for i, vals in per_seed:
        if vals:
            ax.hist(
                vals,
                bins=bins,
                color=_seed_color(i),
                alpha=0.5,
                label=f"seed {i + 1} (n={len(vals)} gens)",
            )
    # Composite outline over all seeds combined:
    ax.hist(
        all_vals,
        bins=bins,
        histtype="step",
        color="black",
        linewidth=1.0,
        label=f"composite (n={len(all_vals)} gens total)",
    )
    ax.set_title(
        f"{exp_label} (mean={np.mean(all_vals):.3g}, std={np.std(all_vals):.3g})",
        fontsize=8,
    )
    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.tick_params(labelsize=7)


def _plot_hist_region(
    ax_top,
    ax_bot,
    exp1_data,
    exp2_data,
    value_key,
    skip_gens,
    label1,
    label2,
    xlabel,
):
    """Plot histograms (Exp1 top, Exp2 bottom)"""
    vals1 = [
        v
        for _i, vals in _per_seed_cell_values(exp1_data, value_key, skip_gens)
        for v in vals
    ]
    vals2 = [
        v
        for _i, vals in _per_seed_cell_values(exp2_data, value_key, skip_gens)
        for v in vals
    ]
    all_vals = np.array(vals1 + vals2)
    if len(all_vals) == 0:
        ax_top.set_title(f"{xlabel}: no data (skip={skip_gens})", fontsize=8)
        return
    if len(all_vals) == 1 or np.ptp(all_vals) == 0:
        span = abs(all_vals[0]) * 0.1 or 1.0
        bins = np.linspace(all_vals.min() - span, all_vals.max() + span, 10)
    else:
        bins = np.linspace(all_vals.min(), all_vals.max(), 20)

    _plot_hist_one(ax_top, exp1_data, value_key, skip_gens, bins, label1)
    _plot_hist_one(ax_bot, exp2_data, value_key, skip_gens, bins, label2)
    ax_bot.set_xlabel(xlabel, fontsize=8)


def make_figure(
    species, exp1_data, exp2_data, label1, label2, skip_gens, outdir, file_prefix
):
    """Builds and saves one 4x2 figure for a single small-molecule species"""
    fig = plt.figure(figsize=(17, 11))

    gs = gridspec.GridSpec(
        4, 2, figure=fig, width_ratios=[2, 1], hspace=0.55, wspace=0.55
    )

    # left plots: 4 stacked time-series panels
    ax_c1 = fig.add_subplot(gs[0, 0])
    ax_c2 = fig.add_subplot(gs[1, 0], sharex=ax_c1)
    ax_r1 = fig.add_subplot(gs[2, 0])
    ax_r2 = fig.add_subplot(gs[3, 0], sharex=ax_r1)
    _plot_timeseries_panel(
        ax_c1,
        exp1_data,
        "counts",
        f"{label1}: {species} total counts",
        "counts",
    )
    _plot_timeseries_panel(
        ax_c2,
        exp2_data,
        "counts",
        f"{label2}: {species} total counts",
        "counts",
    )
    _plot_timeseries_panel(
        ax_r1,
        exp1_data,
        "roc",
        f"{label1}: {species} change in counts (dilution excluded)",
        "counts/s",
    )
    _plot_timeseries_panel(
        ax_r2,
        exp2_data,
        "roc",
        f"{label2}: {species} change in counts (dilution excluded)",
        "counts/s",
    )
    ax_r2.set_xlabel("time (s)", fontsize=8)
    for ax in (ax_c1, ax_c2, ax_r1, ax_r2):
        ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.01, 0.5))

    # Right plots: histograms for each time-series panel
    gs_top = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0:2, 1], hspace=0.4)
    gs_bot = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2:4, 1], hspace=0.4)
    _plot_hist_region(
        fig.add_subplot(gs_top[0]),
        fig.add_subplot(gs_top[1]),
        exp1_data,
        exp2_data,
        "counts",
        skip_gens,
        label1,
        label2,
        "Avg. total count per cell",
    )
    _plot_hist_region(
        fig.add_subplot(gs_bot[0]),
        fig.add_subplot(gs_bot[1]),
        exp1_data,
        exp2_data,
        "roc",
        skip_gens,
        label1,
        label2,
        "Avg. ROC (counts/s) per cell",
    )

    fig.suptitle(
        f"{species}: {label1} vs {label2} "
        f"\n(skips first {skip_gens} generations in histograms)",
        fontsize=12,
    )

    safe_species = species.replace("[", "_").replace("]", "")
    output_filename = os.path.join(outdir, f"{file_prefix}_{safe_species}.png")
    plt.savefig(output_filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


# Generate plots:
def plot(
    params: dict[str, Any],
    conn: DuckDBPyConnection,
    history_sql: str,
    config_sql: str,
    success_sql: str,
    sim_data_paths: dict[str, dict[int, Any]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_names: dict[str, str],
):
    """
    Compare total small molecule counts and rate-of-change (ROC) between two
    experiments.

    Args:
        params: Dictionary containing parameters of the format::

            {
                # Initial generations to skip in the histograms only
                "skip_n_gens": int (default: 2),
                # Optional override of the module-level PLOT_METABOLITES list
                "metabolites": list[str] (default: PLOT_METABOLITES)
            }
    """
    skip_gens = params.get("skip_n_gens", 2)
    plot_molecules = params.get("molecules", PLOT_MOLECULES)
    listener_col = "listeners__small_molecule_counts__totalSmallMoleculeCounts"

    sm_ids = field_metadata(conn, config_sql, listener_col)

    # Determine the two experiments to compare:
    present = set(
        conn.sql(f"SELECT DISTINCT experiment_id FROM ({history_sql})")
        .pl()["experiment_id"]
        .to_list()
    )
    unique_exp_ids = [e for e in sim_data_paths.keys() if e in present]
    if len(unique_exp_ids) < 2:
        raise ValueError(
            f"Expected 2 experiments but found {len(unique_exp_ids)}: "
            f"{unique_exp_ids}. Make sure both experiment_ids are in the config."
        )
    exp_id_1, exp_id_2 = unique_exp_ids[0], unique_exp_ids[1]
    print(f"Comparing {exp_id_1} (Exp 1) vs {exp_id_2} (Exp 2)")

    # Check that all listed metabolites are present in the metadata and can be plotted:
    species_to_plot: list[str] = []
    for sm_id in plot_molecules:
        species_list = resolve_species(sm_id, sm_ids)
        if not species_list:
            print(
                f"WARNING: {sm_id!r} not present in SmallMoleculeCounts "
                f"metadata; skipping."
            )
            continue
        species_to_plot.extend(species_list)

    for species in species_to_plot:
        idx = sm_ids.index(species)
        print(f"Plotting {species} (idx={idx}) ...")
        cells_by_exp = load_species_cells(conn, history_sql, idx)
        exp1_data = group_by_seed(cells_by_exp.get(exp_id_1, []))
        exp2_data = group_by_seed(cells_by_exp.get(exp_id_2, []))
        if not exp1_data and not exp2_data:
            print(f"WARNING: no cell data for {species}; skipping.")
            continue
        make_figure(
            species,
            exp1_data,
            exp2_data,
            exp_id_1,
            exp_id_2,
            skip_gens,
            outdir,
            "small_molecule_comparison",
        )
