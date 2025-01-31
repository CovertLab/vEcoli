"""
Script to make majority of figures in paper.
Before running, extract colony_data.zip in the "data" folder.
Mapping between filenames in "colony_data/sim_dfs" folder and
simulation conditions in ecoli/analysis/antibiotics_colony/__init__.py.
The __init__ file also constructs a dictionary of all columns in the saved
CSV files (and gives their paths in the simulation store hierarchy).

Refer to these other scripts for the code to create remaining figures:
    - ecoli/analysis/antibiotics_colony/subgen_gene_plots/
        - count_subgen.py: counts for Fig. 1B
            - Run with "--data data/colony_data/glc_10000_expressome.csv"
        - make_fig_1b.py: create Fig. 1B
    - ecoli/analysis/antibiotics_colony/snapshot_and_hist_plot.py: Fig. 2E-F
        - Run with "--local data/colony_data/2022-12-08_00-35-28_562633+0000.csv"
    - ecoli/analysis/antibiotics_colony/tet_dry_mass.py: Fig. 3E
    - ecoli/analysis/antibiotics_colony/amp_plots.py: Fig. 4D-J, L
        - Run with "--glc_data data/colony_data/2022-12-08_00-33-56_581605+0000.csv"
          and "--amp_data data/colony_data/2022-12-08_17-03-56_357734+0000.csv"
    - ecoli/analysis/antibiotics_colony/spatial_autocorrelation.py: Fig. S4
        - Run with "data/colony_data/sim_dfs/2022-12-08_00-35-28_562633+0000.csv"
        - Repeat with CSV files for other two baseline glucose simulations
          to get all Moran's I and p-values in Table S1
    - ecoli/analysis/proteinCountsValidation.py: Fig. S2A
        - Run with "--avg_data data/colony_data/glc_10000_proteome_avgs.csv"
    - ecoli/analysis/centralCarbonMetabolismScatter.py: Fig. S2B
        - Run with "--numpy_data data/colony_data/glc_10000_fluxome.csv" and
          "--sim_df data/colony_data/2022-12-08_00-35-28_562633+0000.csv"
"""

import ast
import argparse
import os
from itertools import combinations
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.constants import N_A
from scipy.stats import spearmanr
import seaborn as sns
from typing import Any
from vivarium.library.dict_utils import deep_merge

from ecoli.analysis.antibiotics_colony import (
    COUNTS_PER_FL_TO_NANOMOLAR,
    DE_GENES,
    EXPERIMENT_ID_MAPPING,
    MAX_TIME,
    SPLIT_TIME,
    restrict_data,
)
from ecoli.analysis.antibiotics_colony.exploration import (
    plot_exp_growth_rate,
    plot_ampc_phylo,
)
from ecoli.analysis.antibiotics_colony.timeseries import (
    make_tag_video,
    plot_field_snapshots,
    plot_tag_snapshots,
    plot_timeseries,
)
from ecoli.analysis.antibiotics_colony.validation import (
    plot_colony_growth,
    plot_mrna_fc,
    plot_protein_synth_inhib,
    plot_death_timescale_analysis,
)


def make_figure_1a(data, metadata):
    # Generational (ompF) vs sub-generational (marR) expression
    columns_to_plot = {
        "ompF mRNA": "0.4",
        "marR mRNA": (0, 0.4, 1),
        "OmpF monomer": "0.4",
        "MarR monomer": (0, 0.4, 1),
    }
    fig, axes = plt.subplots(2, 2, sharex="col", figsize=(6, 6))
    axes = np.ravel(axes)
    # Arbitrarily pick a surviving agent to plot trace of
    highlight_agent = "011001001"
    print(f"Highlighted agent: {highlight_agent}")
    plot_timeseries(
        data=data,
        axes=axes,
        columns_to_plot=columns_to_plot,
        highlight_lineage=highlight_agent,
        background_lineages=False,
    )
    axes[0].set_xlabel(None)
    axes[1].set_xlabel(None)
    # Put gene name on top and remove superfluous axes labels
    gene_1 = axes[0].get_ylabel().split(" ")[0]
    gene_2 = axes[1].get_ylabel().split(" ")[0]
    axes[0].set_ylabel("mRNA\n(counts)")
    axes[2].set_ylabel("Monomer\n(counts)")
    axes[0].set_title(f"Exponential: {gene_1}", fontsize=10)
    axes[1].set_title(f"Sub-generational: {gene_2}", fontsize=10)
    axes[0].yaxis.set_label_coords(-0.3, 0.5)
    axes[2].yaxis.set_label_coords(-0.3, 0.5)
    axes[1].yaxis.label.set_visible(False)
    axes[3].yaxis.label.set_visible(False)
    axes[0].xaxis.set_visible(False)
    axes[0].spines.bottom.set_visible(False)
    axes[1].xaxis.set_visible(False)
    axes[1].spines.bottom.set_visible(False)
    for ax in axes:
        [item.set_fontsize(8) for item in ax.get_xticklabels()]
        [item.set_fontsize(8) for item in ax.get_yticklabels()]
        ax.xaxis.label.set_fontsize(9)
        ax.yaxis.label.set_fontsize(9)
        ax.xaxis.set_label_coords(0.5, -0.2)
    ax.tick_params(axis="both", which="major")
    fig.set_size_inches(4, 3)
    plt.draw()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    os.makedirs("out/analysis/paper_figures/", exist_ok=True)
    plt.savefig(
        f"out/analysis/paper_figures/fig_1a_{highlight_agent}.svg", bbox_inches="tight"
    )
    plt.close()
    print("Done with Figure 1A.")


def make_figure_s3a(data, metadata):
    # Doubling time histogram vs literature
    doubling_times = []
    grouped_data = data.groupby(["Seed"])
    for condition, condition_data in grouped_data:
        agent_ids = condition_data.loc[:, "Agent ID"].unique()
        grouped_agents = condition_data.groupby(["Agent ID"])
        for agent, agent_data in grouped_agents:
            # Exclude cells alive at final time point because they
            # have not completed their cell cycle. Also exclude cells
            # that died prematurely from cell wall defects
            if agent + "0" in agent_ids:
                doubling_time = (
                    agent_data.loc[:, "Time"].max() - agent_data.loc[:, "Time"].min()
                )
                doubling_times.append(doubling_time)
    fig, ax = plt.subplots(figsize=(2, 2))
    # Literature doubling time = 44 minutes (10.1128/jb.119.1.270-281.1974)
    ax.vlines(44, 0, 150, colors=["tab:orange"])
    ax.text(
        45,
        150,
        "experiment\n\u03c4 = 44 min.",
        verticalalignment="top",
        horizontalalignment="left",
        c="tab:orange",
        fontsize=8,
    )
    doubling_times = np.array(doubling_times) / 60
    ax.hist(doubling_times)
    ax.set_xlabel("Doubling time (min)")
    ax.set_ylabel("# of simulated cells")
    ax.set_ylim()
    sim_avg = doubling_times.mean()
    ax.vlines(sim_avg, ax.get_ylim()[0], 100, linestyles=["dashed"], colors=["k"])
    ax.text(
        sim_avg + 2,
        100,
        f"simulation\n\u03c4 = {np.round(sim_avg, 1)} min.",
        verticalalignment="top",
        horizontalalignment="left",
        c="tab:blue",
        fontsize=8,
    )
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.tight_layout()
    os.makedirs("out/analysis/paper_figures/", exist_ok=True)
    plt.savefig(
        "out/analysis/paper_figures/fig_s3a_doubling_time.svg", bbox_inches="tight"
    )
    plt.close()
    print("Done with Figure S3A.")


def make_figure_2a(data, metadata):
    # Snapshots of colony at 1.8 hour increments with environmental
    # glucose concentration colored and a single lineage highlighted
    # in blue
    final_timestep = data.loc[data.loc[:, "Time"] == MAX_TIME, :]
    agent_ids = final_timestep.loc[:, "Agent ID"]
    # Arbitrarily select a lineage to highlight
    highlight_agent = agent_ids[100]
    print(f"Highlighted agent: {highlight_agent}")
    plot_field_snapshots(
        data=data,
        metadata=metadata,
        highlight_lineage=highlight_agent,
        highlight_color=(0, 0.4, 1),
        min_pct=0.8,
        colorbar_decimals=2,
    )
    print("Done with Figure 2A.")


def make_figure_2b_d(data, metadata):
    # Timeseries of dry mass and mRNA/protein concentrations for
    # OmpF, TolC, AmpC, and MarR with highlighted lineage in blue
    # Use same highlighted agent as in Figure 2A
    final_timestep = data.loc[data.loc[:, "Time"] == MAX_TIME, :]
    agent_ids = final_timestep.loc[:, "Agent ID"]
    highlight_agent = agent_ids[100]
    print(f"Highlighted agent: {highlight_agent}")
    # Set up subplot layout for timeseries plots
    fig = plt.figure()
    gs = fig.add_gridspec(3, 4)
    axes = [fig.add_subplot(gs[0, :])]
    for i in range(4):
        axes.append(fig.add_subplot(gs[2, i]))
    for i in range(4):
        axes.append(fig.add_subplot(gs[1, i], sharex=axes[i + 1]))
    columns_to_plot = {
        "Dry mass": (0, 0.4, 1),
    }
    plot_timeseries(
        data=data,
        axes=axes,
        columns_to_plot=columns_to_plot,
        highlight_lineage=highlight_agent,
    )
    columns_to_plot = {
        "OmpF monomer": (0, 0.4, 1),
        "TolC monomer": (0, 0.4, 1),
        "AmpC monomer": (0, 0.4, 1),
        "MarR monomer": (0, 0.4, 1),
        "ompF mRNA": (0, 0.4, 1),
        "tolC mRNA": (0, 0.4, 1),
        "ampC mRNA": (0, 0.4, 1),
        "marR mRNA": (0, 0.4, 1),
    }
    # Use periplasmic volume for proteins that localize to periplasm
    # or outer membrane
    periplasmic = ["OmpF monomer", "AmpC monomer", "TolC monomer"]
    for column in columns_to_plot:
        if column in periplasmic:
            data.loc[:, column] /= data.loc[:, "Volume"] * 0.2
        else:
            data.loc[:, column] /= data.loc[:, "Volume"] * 0.8
        data.loc[:, column] *= COUNTS_PER_FL_TO_NANOMOLAR
    monomer_cols = list(columns_to_plot)[:4]
    # Convert monomer concentrations to uM
    data.loc[:, monomer_cols] /= 1000
    plot_timeseries(
        data=data,
        axes=axes[1:],
        columns_to_plot=columns_to_plot,
        highlight_lineage=highlight_agent,
    )
    # Add more regularly spaced tick marks to dry mass timeseries
    time_ticks = axes[0].get_xticks()
    new_ticks = np.arange(1, np.ceil(time_ticks[1]), 1).astype(int)
    # No need for tick at 7 since final tick is 7.2
    new_ticks = new_ticks[new_ticks != 7].tolist()
    time_ticks = [0] + new_ticks + [time_ticks[1]]
    axes[0].set_xticks(ticks=time_ticks, labels=time_ticks)
    # Put gene name on top and remove superfluous axes labels
    gene = axes[1].get_ylabel().split(" ")[0]
    axes[0].set_ylabel("Dry mass (fg)")
    axes[5].set_title(gene, fontsize=12, fontweight="bold")
    axes[5].set_ylabel("mRNA (nM)")
    axes[1].set_ylabel("Protein (\u03bcM)")
    for i in range(2, 5):
        gene = axes[i].get_ylabel().split(" ")[0]
        axes[i].yaxis.label.set_visible(False)
        axes[4 + i].set_title(gene, fontsize=12, fontweight="bold")
        axes[4 + i].yaxis.label.set_visible(False)
    for ax in axes[5:]:
        ax.xaxis.set_visible(False)
        ax.spines.bottom.set_visible(False)
    for ax in axes:
        [item.set_fontsize(8) for item in ax.get_xticklabels()]
        [item.set_fontsize(8) for item in ax.get_yticklabels()]
        ax.xaxis.label.set_fontsize(10)
        ax.yaxis.label.set_fontsize(10)
        ax.tick_params(axis="both", which="major")
    fig.set_size_inches(7, 4)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.3)
    for ax in axes[1:]:
        ax.xaxis.set_label_coords(0.5, -0.3)
        left, bottom, width, height = ax.get_position().bounds
        ax.set_position((left, bottom - 0.15, width, height))
    left, bottom, width, height = axes[0].get_position().bounds
    axes[0].set_position((left, bottom + 0.03, width, height))
    axes[0].xaxis.set_label_coords(0.5, -0.3)
    axes[0].yaxis.set_label_coords(-0.09, 0.5)
    axes[5].yaxis.set_label_coords(-0.5, 0.5)
    axes[1].yaxis.set_label_coords(-0.5, 0.5)
    # Prettify axes (moving axis titles in to save space)
    for ax in axes[1:5]:
        xmin, xmax = ax.get_xlim()
        ax.set_xticks([(xmin + xmax) / 2], labels=[ax.get_xlabel()], minor=True)
        ax.set_xlabel(None)
        ax.tick_params(
            which="minor",
            width=0,
            length=ax.xaxis.get_major_ticks()[0].get_tick_padding(),
            labelsize=10,
        )
    # Calculate average monomer concentrations per cell then
    # average across all cell
    average_agent_data = data.groupby("Agent ID").mean().mean()
    # Use average simulated volume to calculate literature concentrations
    volume = average_agent_data["Volume"]
    # 10.1016/j.cell.2014.02.033: absolute monomer counts
    # for MG1655 grown in MOPS minimal media + glucose
    lit_protein_concs = {
        "OmpF monomer": 71798,
        "TolC monomer": 3141,
        "AmpC monomer": 132,
        "MarR monomer": 20,
    }
    orange = (1, 100 / 255, 0)
    for ax, monomer in zip(axes[1:5], lit_protein_concs):
        monomer_vol = volume * 0.2 if monomer in periplasmic else volume * 0.8
        monomer_conc = (
            lit_protein_concs[monomer] / monomer_vol * COUNTS_PER_FL_TO_NANOMOLAR / 1000
        )
        print(monomer, monomer_conc, monomer_vol)
        ax.scatter(
            MAX_TIME / 3600 + 0.3, monomer_conc, c=orange, marker="_", clip_on=False
        )
        ax.scatter(
            MAX_TIME / 3600 + 0.3,
            average_agent_data[monomer],
            c=(0, 0.4, 1),
            marker="_",
            clip_on=False,
        )
        ax.scatter(
            MAX_TIME / 3600 + 0.5,
            monomer_conc,
            c=orange,
            marker="o",
            clip_on=False,
            s=20,
        )
        ax.scatter(
            MAX_TIME / 3600 + 0.5,
            average_agent_data[monomer],
            c=(0, 0.4, 1),
            marker="o",
            clip_on=False,
            s=20,
        )
    # 10.1126/science.abk2066: mRNA fractions (concentration over
    # total mRNA concentration) from K-12 strain NCM3722 in glucose
    # minimal medium (2 biological replicates)
    lit_mrna_concs = {
        "ompF mRNA": (8.57e-03, 8.08e-03),
        "tolC mRNA": (3.21e-04, 3.06e-04),
        "ampC mRNA": (6.55e-06, 7.09e-06),
        "marR mRNA": (7.88e-06, 7.13e-06),
    }
    mrna_counts = json.load(open("data/colony_data/glc_10000_total_mrna.json", "r"))
    # Exclude cells from end of simulation and cells that died because they
    # did not complete their normal cell cycle (inaccurate average mRNA count)
    exclude_cells = data.loc[data["Time"] == MAX_TIME, "Agent ID"].tolist()
    cracked_cells = data.loc[data["Wall cracked"], "Agent ID"].unique()
    unique_agents = data["Agent ID"].unique().tolist()
    for cracked_agent in cracked_cells:
        if cracked_agent + "0" not in unique_agents:
            exclude_cells.append(cracked_agent)
    complete_mrna_counts = [
        mrna_count
        for agent, mrna_count in mrna_counts.items()
        if agent not in exclude_cells
    ]
    average_mrna_count = np.mean(complete_mrna_counts)
    for ax, mrna in zip(axes[5:], lit_mrna_concs):
        mrna_vol = volume * 0.8
        mrna_conc = (
            np.mean(lit_mrna_concs[mrna])
            * average_mrna_count
            / mrna_vol
            * COUNTS_PER_FL_TO_NANOMOLAR
        )
        print(mrna, mrna_conc, mrna_vol)
        ax.scatter(
            MAX_TIME / 3600 + 0.3, mrna_conc, c=orange, marker="_", clip_on=False
        )
        ax.scatter(
            MAX_TIME / 3600 + 0.3,
            average_agent_data[mrna],
            c=(0, 0.4, 1),
            marker="_",
            clip_on=False,
        )
        ax.scatter(
            MAX_TIME / 3600 + 0.5, mrna_conc, c=orange, marker="o", clip_on=False, s=20
        )
        ax.scatter(
            MAX_TIME / 3600 + 0.5,
            average_agent_data[mrna],
            c=(0, 0.4, 1),
            marker="o",
            clip_on=False,
            s=20,
        )
    # Bring back disappearing y ticks
    for ax in axes:
        yticks = ax.get_yticks()
        ax.set_yticks(yticks, yticks.astype(int))
    # Get fractional upper limit for MarR monomer expression
    marR_max = np.round(data.loc[:, "MarR monomer"].max(), 1)
    axes[4].set_ylim([0, marR_max])
    axes[4].set_yticks([0, marR_max], [0, marR_max], fontsize=8)
    axes[4].spines.left.set_bounds([0, marR_max])
    os.makedirs("out/analysis/paper_figures/", exist_ok=True)
    plt.savefig(
        "out/analysis/paper_figures/fig_2b_d_timeseries.svg", bbox_inches="tight"
    )
    plt.close()
    print("Done with Figure 2B-D.")


def make_figure_s1(data, metadata):
    # Make plots describing glucose depletion from environment
    # Convert glucose uptake to units of mmol / g DCW / hr
    exchange_data = data.loc[:, ["Dry mass", "Exchanges"]]
    glc_flux = exchange_data.apply(
        lambda x: x["Exchanges"]["GLC[p]"]
        / N_A
        * 1000
        / (x["Dry mass"] * 1e-15)
        / 2
        * 3600,
        axis=1,
    )
    data["Glucose intake"] = -glc_flux
    grouped_data = data.groupby("Agent ID").mean()
    print(
        "Mean glucose intake (mmol/g DCW/hr): "
        + str(grouped_data.loc[:, "Glucose intake"].mean())
    )
    print(
        "Std. dev. glucose intake (mmol/g DCW/hr): "
        + str(grouped_data.loc[:, "Glucose intake"].std())
    )
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.histplot(
        grouped_data.loc[:, "Glucose intake"], color=(0, 0.4, 1), ax=ax, linewidth=1
    )
    ax.set_xlabel("Glucose intake\n(mmol/g DCW/hr)", fontsize=10)
    ax.set_ylabel("Simulated cells", fontsize=10)
    ax.set_xticks(ax.get_xticks(), ax.get_xticks().astype(int), fontsize=10)
    ax.set_yticks(ax.get_yticks(), ax.get_yticks().astype(int), fontsize=10)
    plt.tight_layout()
    os.makedirs("out/analysis/paper_figures/", exist_ok=True)
    plt.savefig(
        "out/analysis/paper_figures/fig_s1a_glc_intake.svg", bbox_inches="tight"
    )
    plt.close()
    print("Done with Figure S1A.")

    # Cross-section of environmental glucose concentration
    field_data = metadata["Glucose"][10000]["fields"]
    xticks = np.arange(0, 50, 5)
    xcoords = xticks + 2.5
    sample_times = [10400, 15600, 20800, 26000]
    cmap = matplotlib.colormaps["Greys"]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=26000)
    fig, ax = plt.subplots(figsize=(3, 3))
    for sample_time in sample_times:
        cross_sec = np.array((field_data[sample_time]["GLC[p]"])).T[4]
        color = cmap(norm(sample_time))
        ax.plot(xcoords, cross_sec, c=color)
        ax.scatter(xcoords, cross_sec, c=color)
        ax.text(
            51,
            cross_sec.mean(),
            f"{np.round(sample_time / 3600, 1)} hr",
            horizontalalignment="left",
            verticalalignment="center",
            c=color,
        )
    ax.set_xlabel("Distance from left edge\nof environment (\u03bcm)")
    ax.set_ylabel("Cross-sectional glucose (mM)")
    xticks = np.append(xticks, 50).astype(int)
    ax.set_xticks(xticks)
    plt.savefig("out/analysis/paper_figures/fig_s1b_env_cross.svg", bbox_inches="tight")
    plt.close()
    print("Done with Figure S1B.")


def make_figure_s5(data, metadata):
    # Perform linear regression on average per-cell protein expression for
    # all possible pairs of the four antibiotic resistance genes
    data = restrict_data(data)
    monomers = ["OmpF monomer", "AmpC monomer", "TolC monomer", "MarR monomer"]
    data.loc[:, monomers] = (
        data.loc[:, monomers].divide(data.loc[:, "Volume"], axis=0)
        * COUNTS_PER_FL_TO_NANOMOLAR
        / 1000
    )
    avg_concs = data.loc[:, monomers + ["Agent ID"]].groupby("Agent ID").mean()
    new_colnames = {col: col + " (\u03bcM)" for col in monomers}
    avg_concs.rename(columns=new_colnames, inplace=True)
    g = sns.pairplot(avg_concs, kind="reg", corner=True)

    combos = list(combinations(new_colnames.values(), 2))
    plot_xvars = np.array(g.x_vars)
    plot_yvars = np.array(g.y_vars)
    for monomer_1, monomer_2 in combos:
        # Add Spearman R and p-value to corner of each plot
        r, p = spearmanr(avg_concs[monomer_1], avg_concs[monomer_2])
        adj_p = p * len(combos)
        adj_p = min(adj_p, 1)
        x_idx = np.where(plot_xvars == monomer_1)[0][0]
        y_idx = np.where(plot_yvars == monomer_2)[0][0]
        if x_idx > y_idx:
            x_idx = np.where(plot_xvars == monomer_2)[0][0]
            y_idx = np.where(plot_yvars == monomer_1)[0][0]
        ax = g.axes[y_idx, x_idx]
        ax.text(
            s=f"r = {np.round(r, 2)}",
            y=1,
            x=1,
            ha="right",
            va="top",
            transform=ax.transAxes,
        )
        if adj_p < 0.05:
            ax.text(
                s=f"p = {np.format_float_scientific(adj_p, 1)}*",
                y=0.9,
                x=1,
                ha="right",
                va="top",
                transform=ax.transAxes,
                weight="bold",
            )
        else:
            ax.text(
                s=f"p = {np.format_float_scientific(adj_p, 1)}",
                y=0.9,
                x=1,
                ha="right",
                va="top",
                transform=ax.transAxes,
            )
        print(f"{monomer_1} vs. {monomer_2}: r = {r}, Bonferroni corrected p = {adj_p}")
    os.makedirs("out/analysis/paper_figures/", exist_ok=True)
    plt.savefig("out/analysis/paper_figures/fig_s5_supp_protein_pairplot.svg")
    plt.close()
    print("Done with Figure S5.")


def make_figure_s3b(data, metadata):
    # Plot std. dev. in average birth time of cells for each generation
    agent_data = data.groupby("Agent ID")
    start_times = []
    for agent_id, agent in agent_data:
        # Exclude final generation of cells because not all cells in the prior
        # generation had finished dividing
        if len(agent_id) > 9:
            continue
        start_times.append((len(agent_id), agent["Time"].min()))
    start_times = pd.DataFrame(start_times)
    start_times.rename(columns={0: "Generation", 1: "Start time (s)"}, inplace=True)
    std_start_times = start_times.groupby("Generation").std()
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.scatter(std_start_times.index, std_start_times["Start time (s)"], c="k")
    ax.set_xlabel("Generation", fontsize=9)
    ax.set_ylabel("Birth time std. dev. (s)", fontsize=9)
    ax.set_xticks(range(2, 10), range(2, 10), fontsize=8)
    ax.set_yticks(range(0, 600, 100), range(0, 600, 100), fontsize=8)
    sns.despine(ax=ax, trim=True, offset=3)
    ax.set_xticks(range(2, 10), range(2, 10), fontsize=8)
    ax.set_yticks(range(0, 600, 100), range(0, 600, 100), fontsize=8)
    os.makedirs("out/analysis/paper_figures/", exist_ok=True)
    plt.savefig(
        "out/analysis/paper_figures/fig_s3b_birth_time_std_dev.svg", bbox_inches="tight"
    )
    plt.close()
    print("Done with Figure S3B.")


def make_figure_3l(data, metadata):
    # Plot total colony mass traces for all tested tetracycline concs.
    fig, ax = plt.subplots(figsize=(2.1, 2.35))
    plot_colony_growth(data, ax)
    offset_time = SPLIT_TIME / 3600
    min_time = np.round(-offset_time, 1)
    max_time = np.round(MAX_TIME / 3600 - offset_time, 1)
    ticks = [min_time, 0, int(max_time)]
    ax.set_xticks(np.array(ticks) + offset_time, ticks, size=8)
    ax.spines["bottom"].set_bounds(0, MAX_TIME / 3600)
    ax.spines["left"].set_bounds(ax.get_ylim())
    ax.set_xlabel("Hours after tet. addition", size=9)
    ax.set_ylabel("Colony mass (fg)", size=9)
    yticklabels = [f"$10^{int(exp)}$" for exp in np.log10(ax.get_yticks())]
    ax.set_yticks(ax.get_yticks(), yticklabels, size=9)
    plt.tight_layout()
    os.makedirs("out/analysis/paper_figures/", exist_ok=True)
    fig.savefig(
        "out/analysis/paper_figures/fig_3l_tet_colony_mass.svg", bbox_inches="tight"
    )
    plt.close()
    print("Done with Figure 3L.")


def make_figure_3d_and_3m(data, metadata):
    # Plot snapshots with intervals of 1.3 hours of decrease in
    # instantaneous growth rate after tetracycline addition (3D)
    # Also, create two scatterplots of average doubling rate
    # against active ribosome concentration for cells in the first
    # and fourth hours of tetracycline exposure (3M)
    data = data.loc[data.loc[:, "Time"] <= MAX_TIME, :]
    data = data.sort_values(["Condition", "Agent ID", "Time"])
    # Draw blue border around highlighted agent lineage
    highlight_agent_id = "0011111"
    plot_exp_growth_rate(data, metadata, highlight_agent_id)
    plt.close()
    print("Done with Figure 3D.")


def make_figure_3f_h(data, metadata):
    # Timeseries traces of tetracycline and active ribosome
    # concentrations in minutes surrounding tet. addition
    # Filter data to only include 150 seconds before and after
    glucose_mask = (
        (data.loc[:, "Time"] >= 11400)
        & (data.loc[:, "Time"] <= SPLIT_TIME)
        & (data.loc[:, "Condition"] == "Glucose")
    )
    tet_mask = (
        (data.loc[:, "Time"] >= SPLIT_TIME)
        & (data.loc[:, "Time"] <= 11700)
        & (data.loc[:, "Condition"] == "Tetracycline (1.5 mg/L)")
    )
    transition_data = data.loc[glucose_mask | tet_mask, :]
    # Convert tetracycline concentrations to uM
    transition_data.loc[:, "Periplasmic tetracycline"] *= 1000
    transition_data.loc[:, "Cytoplasmic tetracycline"] *= 1000
    # Convert to concentration using cytoplasmic volume
    transition_data.loc[:, "Active ribosomes"] /= transition_data.loc[:, "Volume"] * 0.8
    # Convert all concentrations to uM
    transition_data.loc[:, "Active ribosomes"] *= COUNTS_PER_FL_TO_NANOMOLAR / 1000
    transition_data.rename(
        columns={
            "Periplasmic tetracycline": "Tetracycline\n(periplasm)",
            "Cytoplasmic tetracycline": "Tetracycline\n(cytoplasm)",
            "Active ribosomes": "Active\nribosomes",
        },
        inplace=True,
    )
    fig, axes = plt.subplots(1, 3, figsize=(5, 1.5))
    short_term_columns = {
        "Tetracycline\n(periplasm)": 0,
        "Tetracycline\n(cytoplasm)": 1,
        "Active\nribosomes": 2,
    }
    for column, ax_idx in short_term_columns.items():
        plot_timeseries(
            data=transition_data,
            axes=[axes.flat[ax_idx]],
            columns_to_plot={column: (0, 0.4, 1)},
            highlight_lineage="0011111",
            filter_time=False,
            background_alpha=0.5,
            background_linewidth=0.3,
        )
    for ax in axes.flat:
        ylim = ax.get_ylim()
        yticks = np.round(ylim, 0).astype(int)
        ax.set_yticks(yticks, yticks, size=9)
        ax.set_xlabel(None)
        # Mark minutes since tetracycline addition
        ax.set_xticks(
            ticks=[
                11430 / 3600,
                11490 / 3600,
                11550 / 3600,
                11610 / 3600,
                11670 / 3600,
            ],
            labels=[-2, -1, 0, 1, 2],
            size=9,
        )
        ax.spines.bottom.set(
            bounds=(11400 / 3600, 11700 / 3600),
            linewidth=1,
            visible=True,
            color=(0, 0, 0),
            alpha=1,
        )
        ylabel = ax.get_ylabel()
        ax.set_ylabel(None)
        ax.set_title(ylabel, size=9)
    axes.flat[0].set_ylabel("\u03bcM", size=9, labelpad=0)
    axes.flat[1].set_xlabel("Minutes after tetracycline addition", size=9)
    # Ensure that active ribosomes plot starts at y = 0
    axes.flat[-1].set_ylim(0, axes.flat[-1].get_ylim()[1])
    new_yticks = [0, axes.flat[-1].get_yticks()[-1]]
    axes.flat[-1].set_yticks(new_yticks, new_yticks)
    axes.flat[-1].spines["left"].set_bounds(new_yticks)
    plt.tight_layout()
    fig.subplots_adjust(left=0.08, right=0.99, top=0.8, bottom=0.3, wspace=0.35)
    os.makedirs("out/analysis/paper_figures/", exist_ok=True)
    fig.savefig("out/analysis/paper_figures/fig_3f_h_tet_short_term.svg")
    plt.close()
    print("Done with Figure 3F-H.")


def make_figure_3i_k(data, metadata):
    # Timeseries traces of micF-ompF duplex, ompF mRNA, and OmpF monomer
    # concentrations in hours surrounding tetracycline addition
    # Filter data to include glucose for first 11550 seconds and
    # tetracycline data for remainder of simulation
    long_transition_data = restrict_data(data)
    long_term_columns = {
        "micF-ompF duplex": 0,
        "ompF mRNA": 1,
        "OmpF monomer": 2,
    }
    # Use periplasmic volume for OmpF monomer because it localizes
    # to outer membrane
    periplasmic = ["OmpF monomer"]
    for column in long_term_columns:
        if column in periplasmic:
            long_transition_data.loc[:, column] /= (
                long_transition_data.loc[:, "Volume"] * 0.2
            )
        else:
            long_transition_data.loc[:, column] /= (
                long_transition_data.loc[:, "Volume"] * 0.8
            )
        # Convert all concentrations to uM
        long_transition_data.loc[:, column] *= COUNTS_PER_FL_TO_NANOMOLAR / 1000
    fig, axes = plt.subplots(1, 3, figsize=(5, 1.5))
    for column, ax_idx in long_term_columns.items():
        plot_timeseries(
            data=long_transition_data,
            axes=[axes.flat[ax_idx]],
            columns_to_plot={column: (0, 0.4, 1)},
            highlight_lineage="0011111",
            filter_time=False,
            background_alpha=0.5,
            background_linewidth=0.3,
        )
    split_hours = SPLIT_TIME / 3600
    rounded_split_hours = np.round(split_hours, 1)
    for ax in axes.flat:
        ylim = ax.get_ylim()
        yticks = np.round(ylim, 0).astype(int)
        ax.set_yticks(yticks, yticks, size=9)
        # Mark hours since tetracycline addition
        xlim = np.array(ax.get_xlim())
        xticks = np.append(xlim, split_hours)
        xtick_labels = np.trunc(xticks - split_hours).astype(int).tolist()
        xtick_labels = [
            label if label != int(-split_hours) else -rounded_split_hours
            for label in xtick_labels
        ]
        ax.set_xticks(ticks=xticks, labels=xtick_labels, size=9)
        ax.set_xlabel(None)
        ax.spines.bottom.set(
            bounds=(0, MAX_TIME / 3600),
            linewidth=1,
            visible=True,
            color=(0, 0, 0),
            alpha=1,
        )
        ylabel = ax.get_ylabel()
        ax.set_ylabel(None)
        ax.set_title(ylabel, size=9, pad=12)
    axes.flat[0].set_ylabel("\u03bcM", size=9, labelpad=-6)
    fig.supxlabel("Hours after tetracycline addition", size=9)
    # Ensure that OmpF monomer plot starts at y = 0
    for i in range(2):
        y_max = np.round(axes.flat[i].get_ylim()[-1], 2)
        new_yticks = [0, y_max]
        axes.flat[i].set_yticks(new_yticks, new_yticks)
        axes.flat[i].spines["left"].set_bounds(new_yticks)
    axes.flat[-1].set_ylim(0, axes.flat[-1].get_ylim()[1])
    new_yticks = [0, axes.flat[-1].get_yticks()[-1]]
    axes.flat[-1].set_yticks(new_yticks, new_yticks)
    axes.flat[-1].spines["left"].set_bounds(new_yticks)
    plt.tight_layout()
    fig.subplots_adjust(left=0.08, right=0.99, top=0.8, bottom=0.3, wspace=0.35)
    os.makedirs("out/analysis/paper_figures/", exist_ok=True)
    fig.savefig("out/analysis/paper_figures/fig_3i_k_tet_long_term.svg")
    plt.close()
    print("Done with Figure 3I-K.")


def make_figure_s6b(data, metadata):
    # Scatter plot of simulated and measured mRNA fold changes
    # after exposure to 1.5 mg/L tetracycline. acrA, acrB, tolC
    # genes highlighted in blue
    genes_to_plot = DE_GENES.loc[:, "Gene name"]
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    genes = ["acrA", "acrB", "tolC"]
    highlight_genes = {gene: (0, 0.4, 1) for gene in genes}
    plot_mrna_fc(data, ax, genes_to_plot, highlight_genes=highlight_genes)
    ax.spines["left"].set_bounds((-1, 1.5))
    ax.set_yticks([-1, -0.5, 0, 0.5, 1, 1.5])
    os.makedirs("out/analysis/paper_figures/", exist_ok=True)
    fig.savefig(
        "out/analysis/paper_figures/fig_s6b_tet_gene_exp.svg", bbox_inches="tight"
    )
    plt.close()
    print("Done with Figure S6B.")


def make_figure_s6a(data, metadata):
    # Dose-response curve for protein synthesis inhibition in model
    # and real cells
    # Jenner et al. 2013: 10.1073/pnas.1216691110
    jenner = pd.read_csv("data/colony_data/jenner_2013.csv", header=None).rename(
        columns={0: "Tetracycline", 1: "Percent inhibition"}
    )
    jenner["Source"] = ["Jenner et al. 2013"] * len(jenner)
    jenner.loc[:, "Percent inhibition"] = 100 - (jenner.loc[:, "Percent inhibition"])
    # Olson et al. 2006: 10.1128/AAC.01499-05
    olson = pd.read_csv("data/colony_data/olson_2006.csv", header=None).rename(
        columns={0: "Tetracycline", 1: "Percent inhibition"}
    )
    olson["Source"] = ["Olson et al. 2006"] * len(olson)
    literature = pd.concat([jenner, olson])
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    plot_protein_synth_inhib(data, ax, literature)
    plt.tight_layout()
    plt.subplots_adjust(top=1, bottom=0.25, left=0.2, right=1)
    ax.set_xlabel("Tetracycline (\u03bcM)", size=10)
    ax.set_ylabel("Protein synthesis inhibition (%)", size=10)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.spines["bottom"].set_bounds((0.08, 300))
    children = ax.get_children()
    for i in [4, 5, 6]:
        children[i].set_fontsize(10)
    os.makedirs("out/analysis/paper_figures/", exist_ok=True)
    plt.savefig(
        "out/analysis/paper_figures/fig_s6a_protein_synth_inhib.svg",
        bbox_inches="tight",
    )
    plt.close()
    print("Done with Figure S6A.")


def make_figure_s7(data, metadata):
    # Decrease in outer membrane permeability but increase in
    # cytoplasmic tetracycline and decreased AcrAB-TolC conc.
    data = restrict_data(data)
    data["Time"] -= SPLIT_TIME
    column = "Outer tet. permeability (cm/s)"
    # Convert permeability to nm / s for readability
    data[column] *= 1e7
    fig, ax = plt.subplots(figsize=(3, 3))
    plot_timeseries(
        data=data,
        axes=[ax],
        columns_to_plot={column: (0, 0.4, 1)},
        highlight_lineage="0011111",
        filter_time=False,
        background_alpha=0.5,
        background_linewidth=0.3,
    )
    data[column].max() * 1e7
    ax.set_ylabel("OM tet. perm. (nm/s)")
    ax.set_xlabel("Hours after tetracycline addition")
    ax.set_xticks(np.append(ax.get_xticks(), 0), np.append(ax.get_xticks(), 0))
    os.makedirs("out/analysis/paper_figures/", exist_ok=True)
    plt.savefig(
        "out/analysis/paper_figures/fig_s7a_tet_om_perm.svg", bbox_inches="tight"
    )
    plt.close()
    # Get AcrAB-TolC periplasmic concentration in uM
    data["AcrAB-TolC (\u03bcM)"] = (
        data["AcrAB-TolC"] / (data["Volume"] * 0.2) * COUNTS_PER_FL_TO_NANOMOLAR / 1000
    )
    data["Time"] /= 60
    # Convert tetracycline concentrations to uM
    data["Initial external tet."] *= 1000
    data["Cytoplasmic tetracycline"] *= 1000
    data.rename(
        columns={
            "Cytoplasmic tetracycline": "Cytoplasmic tetracycline (\u03bcM)",
            "Time": "Minutes after tetracycline addition",
            "Initial external tet.": "External tet. (\u03bcM)",
        },
        inplace=True,
    )
    # Start 2 min after tet. addition to skip initial spike in tet conc.
    tet_data = data.loc[
        data["Minutes after tetracycline addition"] >= 2,
        [
            "External tet. (\u03bcM)",
            "Cytoplasmic tetracycline (\u03bcM)",
            "Periplasmic tetracycline (\u03bcM)",
            "Minutes after tetracycline addition",
            "AcrAB-TolC (\u03bcM)",
        ],
    ]
    # Highlight MIC in blue and use grayscale for other concentrations
    cmap = matplotlib.colormaps["Greys"]
    antibiotic_min = data.loc[:, "External tet. (\u03bcM)"].min()
    antibiotic_max = data.loc[:, "External tet. (\u03bcM)"].max()
    norm = matplotlib.colors.Normalize(
        vmin=1.5 * antibiotic_min - 0.5 * antibiotic_max, vmax=antibiotic_max
    )
    antibiotic_concs = data.loc[:, "External tet. (\u03bcM)"].unique()
    palette = {
        antibiotic_conc: cmap(norm(antibiotic_conc))
        for antibiotic_conc in antibiotic_concs
    }
    palette[3.375] = (0, 0.4, 1)
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.lineplot(
        tet_data,
        x="Minutes after tetracycline addition",
        y="AcrAB-TolC (\u03bcM)",
        hue="External tet. (\u03bcM)",
        ax=ax,
        palette=palette,
        errorbar=None,
    )
    max_time = np.round((MAX_TIME - SPLIT_TIME) / 60, 0)
    ax.set_xlim(2, int(max_time))
    xticks = ax.get_xticks()[:-1]
    xticks = np.insert(xticks[xticks != 0], 0, 2)
    xticks = np.append(xticks, int(max_time))
    ax.set_xticks(xticks, xticks.astype(int))
    plt.tight_layout()
    ax.legend(bbox_to_anchor=(1.1, 1), title="Ext. tet. (\u03bcM)")
    sns.despine(ax=ax, trim=True, offset=3)
    plt.savefig(
        "out/analysis/paper_figures/fig_s7c_acrab_tet_concs.svg", bbox_inches="tight"
    )
    plt.close()
    # Subtract avg. cytoplasmic tetracycline concentration at 2 min
    # post-tetracycline addition
    initial_steady_tet = (
        tet_data.loc[tet_data["Minutes after tetracycline addition"] == 2, :]
        .groupby("External tet. (\u03bcM)")
        .mean()
    )
    tet_data = (
        tet_data.set_index("External tet. (\u03bcM)") - initial_steady_tet
    ).reset_index()
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.lineplot(
        tet_data,
        x="Minutes after tetracycline addition",
        y="Cytoplasmic tetracycline (\u03bcM)",
        ax=ax,
        hue="External tet. (\u03bcM)",
        palette=palette,
        errorbar=None,
        legend=None,
    )
    max_time = np.round((MAX_TIME - SPLIT_TIME) / 60, 0)
    ax.set_xlim(2, int(max_time))
    xticks = ax.get_xticks()[:-1]
    xticks = np.insert(xticks[xticks != 0], 0, 2)
    xticks = np.append(xticks, int(max_time))
    ax.set_xticks(xticks, xticks.astype(int))
    ax.set_ylabel("Increase in cyto. tet. (\u03bcM)")
    plt.tight_layout()
    sns.despine(ax=ax, trim=True, offset=3)
    plt.savefig(
        "out/analysis/paper_figures/fig_s7b_tet_conc_cyto.svg", bbox_inches="tight"
    )
    plt.close()


def make_figure_s8(data, metadata):
    # Long-term positive feedback in growth inhibition
    # Filter data to include glucose for first 11550 seconds and
    # tetracycline data for remainder of simulation
    long_transition_data = restrict_data(data)
    long_term_columns = {"Active RNAP": (0, 0.4, 1), "mRNA mass": (0, 0.4, 1)}
    # Convert RNAP count to uM using cytoplasmic volume
    long_transition_data.loc[:, "Active RNAP"] /= (
        long_transition_data.loc[:, "Volume"] * 0.8
    )
    long_transition_data.loc[:, "Active RNAP"] *= COUNTS_PER_FL_TO_NANOMOLAR / 1000
    fig, axes = plt.subplots(1, 2, figsize=(5, 1.5))
    plot_timeseries(
        data=long_transition_data,
        axes=axes.flat,
        columns_to_plot=long_term_columns,
        highlight_lineage="0011111",
        filter_time=False,
        background_alpha=0.5,
        background_linewidth=0.3,
    )
    split_hours = SPLIT_TIME / 3600
    rounded_split_hours = np.round(split_hours, 1)
    for ax in axes.flat:
        ylim = ax.get_ylim()
        yticks = np.round(ylim, 0).astype(int)
        ax.set_yticks(yticks, yticks, size=9)
        # Mark hours since tetracycline addition
        xlim = np.array(ax.get_xlim())
        xticks = np.append(xlim, split_hours)
        xtick_labels = np.trunc(xticks - split_hours).astype(int).tolist()
        xtick_labels = [
            label if label != int(-split_hours) else -rounded_split_hours
            for label in xtick_labels
        ]
        ax.set_xticks(ticks=xticks, labels=xtick_labels, size=9)
        ax.set_xlabel(None)
        ax.spines.bottom.set(
            bounds=(0, MAX_TIME / 3600),
            linewidth=1,
            visible=True,
            color=(0, 0, 0),
            alpha=1,
        )
        ylabel = ax.get_ylabel()
        ax.set_ylabel(None)
        ax.set_title(ylabel, size=9, pad=12)
    axes.flat[0].set_ylabel("\u03bcM", size=9, labelpad=-6)
    fig.supxlabel("Hours after tetracycline addition", size=9)
    # Ensure that plots start at y = 0
    for i in range(2):
        y_max = np.round(axes.flat[i].get_ylim()[-1], 2)
        new_yticks = [0, y_max]
        axes.flat[i].set_yticks(new_yticks, new_yticks)
        axes.flat[i].spines["left"].set_bounds(new_yticks)
    axes.flat[-1].set_ylim(0, axes.flat[-1].get_ylim()[1])
    new_yticks = [0, axes.flat[-1].get_yticks()[-1]]
    axes.flat[-1].set_yticks(new_yticks, new_yticks)
    axes.flat[-1].spines["left"].set_bounds(new_yticks)
    # mRNA mass has units fg
    axes[-1].set_ylabel("fg", size=9, labelpad=-6)
    plt.tight_layout()
    fig.subplots_adjust(left=0.08, right=0.99, top=0.8, bottom=0.3, wspace=0.35)
    os.makedirs("out/analysis/paper_figures/", exist_ok=True)
    fig.savefig("out/analysis/paper_figures/fig_s8_tet_supp_feedback.svg")
    plt.close()
    print("Done with Figure S8.")


def make_figure_4l(data, metadata):
    # Timeseries traces of total colony mass for all tested ampicillin concs.
    fig, ax = plt.subplots(figsize=(2.1, 2.1))
    plot_colony_growth(
        data, ax, antibiotic_col="Initial external amp.", mic=5.724, antibiotic="Amp."
    )
    offset_time = SPLIT_TIME / 3600
    min_time = np.round(-offset_time, 1)
    max_time = np.round(MAX_TIME / 3600 - offset_time, 1)
    ticks = [min_time, 0, int(max_time)]
    ax.set_xticks(np.array(ticks) + offset_time, ticks, size=8)
    ax.spines["bottom"].set_bounds(0, MAX_TIME / 3600)
    ax.spines["left"].set_bounds(ax.get_ylim())
    ax.set_xlabel("Hours after amp. addition", size=9)
    ax.set_ylabel("Colony mass (fg)", size=9)
    yticklabels = [f"$10^{int(exp)}$" for exp in np.log10(ax.get_yticks())]
    ax.set_yticks(ax.get_yticks(), yticklabels, size=9)
    plt.tight_layout()
    os.makedirs("out/analysis/paper_figures/", exist_ok=True)
    fig.savefig("out/analysis/paper_figures/fig_4l_amp_colony_mass.svg")
    plt.close()
    print("Done with Figure 4L.")


def make_figure_4c(data, metadata):
    # Snapshots of colony in 1.3 hour intervals after ampicillin addition
    # with ratio of current gap area over average untreated gap area colored
    # as log 2 fold change
    # Get fold change over average glucose porosity
    data["Relative porosity"] = (
        data.loc[:, "Porosity"] * data.loc[:, "Extension factor"]
    )
    mean_glc_porosity = data.loc[
        data.loc[:, "Condition"] == "Glucose", "Relative porosity"
    ].mean()
    fc_col = "Total defect area\n($\\mathregular{log_2}$ fold change)"
    data.loc[:, fc_col] = np.log2(data.loc[:, "Relative porosity"] / mean_glc_porosity)
    # For 10 seconds after division, cells are assigned a porosity of 0
    # until the next update from the cell wall process. Clean up these values
    # so colormap works properly.
    data.loc[data.loc[:, fc_col] == -np.inf, fc_col] = 0
    # Set up custom divergent colormap
    cmp = matplotlib.colors.LinearSegmentedColormap.from_list(
        "divergent", [(0, 0, 0), (1, 1, 1), (0.678, 0, 0.125)]
    )
    magnitude = data.loc[:, fc_col].abs().max()
    norm = matplotlib.colors.Normalize(vmin=-magnitude, vmax=magnitude)
    snapshot_times = np.array([1.9, 3.2, 4.5, 5.8, 7.1]) * 3600
    snapshot_times = np.array([3.2, 4.5, 5.8, 7.1]) * 3600
    # Draw blue border around highlighted agent lineage
    highlight_agent_id = "001111111"
    highlight_agent_ids = [
        highlight_agent_id[: i + 1] for i in range(len(highlight_agent_id))
    ]
    highlight_agent = {
        agent_id: {"membrane_width": 0.5, "membrane_color": (0, 0.4, 1)}
        for agent_id in highlight_agent_ids
    }
    fig = plot_tag_snapshots(
        data=data,
        metadata=metadata,
        tag_colors={fc_col: {"cmp": cmp, "norm": norm}},
        snapshot_times=snapshot_times,
        return_fig=True,
        figsize=(6, 1.5),
        highlight_agent=highlight_agent,
        show_membrane=True,
    )
    fig.axes[0].set_xticklabels(
        np.abs(np.round(fig.axes[0].get_xticks() / 3600 - SPLIT_TIME / 3600, 1))
    )
    fig.axes[0].set_xlabel("Hours after ampicillin addition")
    os.makedirs("out/analysis/paper_figures/", exist_ok=True)
    fig.savefig(
        "out/analysis/paper_figures/fig_4c_amp_snapshots.svg", bbox_inches="tight"
    )
    plt.close()
    print("Done with Figure 4C.")


def make_figure_4m(data, metadata):
    # Plot phylogenetic tree of cells in a representative ampicillin sim,
    # where each node is a cell colored by average AmpC concentration
    os.makedirs("out/analysis/paper_figures/", exist_ok=True)
    plot_ampc_phylo(data)
    print("Done with Figure 4M.")


def make_figure_4n(data, metadata):
    # Compare average generations to lysis in our simulations
    # against literature data
    fig, ax = plt.subplots(1, 1, figsize=(2.2, 2.25))
    axs = [ax]
    plot_death_timescale_analysis(data, axs)
    os.makedirs("out/analysis/paper_figures/", exist_ok=True)
    plt.savefig("out/analysis/paper_figures/fig_4n_misc.svg")
    plt.close()
    print("Done with Figure 4N.")


def make_figure_s10(data, metadata):
    # Pairplot of avg. AmpC, OmpF, PBP1B, and PBP1B concentration against
    # avg. periplasmic ampicillin concentration (with Spearman R and p val.)
    data = restrict_data(data)
    data = data.loc[data.loc[:, "Time"] >= SPLIT_TIME, :]
    protein_names = [
        "AmpC monomer",
        "OmpF monomer",
        "PBP1a complex",
        "PBP1b gamma complex",
        "AcrAB-TolC",
    ]
    # Convert everything into uM (use periplasmic concentration for all
    # because of outer membrane/periplasmic localization)
    data[protein_names] = (
        data[protein_names].divide(data["Volume"] * 0.2, axis=0)
        * COUNTS_PER_FL_TO_NANOMOLAR
        / 1000
    )
    data["Periplasmic ampicillin"] *= 1000
    agent_ids = data.loc[:, "Agent ID"].unique()
    final_agents = data.loc[data.loc[:, "Time"] == MAX_TIME, "Agent ID"].unique()
    # Return True if cell died
    dead_dict = {}

    def check_death(agent_id):
        status = dead_dict.get(agent_id)
        if status:
            return status
        if agent_id + "0" not in agent_ids:
            if agent_id not in final_agents:
                dead_dict[agent_id] = True
                return True
        dead_dict[agent_id] = False
        return False

    data["Dead"] = data["Agent ID"].apply(check_death)
    add_vars = ["Agent ID", "Dead", "Periplasmic ampicillin"]
    avg_data = data[protein_names + add_vars].groupby("Agent ID").mean()
    avg_data["Dead"] = avg_data["Dead"].astype(bool)
    rename_cols = {curr_name: curr_name + " (\u03bcM)" for curr_name in protein_names}
    rename_cols["Periplasmic ampicillin"] = "Periplasmic ampicillin (\u03bcM)"
    avg_data = avg_data.rename(columns=rename_cols)
    fig, axs = plt.subplots(2, 3, sharey=True)
    for i, column in enumerate(protein_names):
        column += " (\u03bcM)"
        plot_legend = False
        if i == len(protein_names) - 1:
            plot_legend = True
        sns.scatterplot(
            avg_data,
            x=column,
            y="Periplasmic ampicillin (\u03bcM)",
            hue="Dead",
            ax=axs.flat[i],
            legend=plot_legend,
        )
        r, p = spearmanr(avg_data[column], avg_data["Periplasmic ampicillin (\u03bcM)"])
        # Bonferroni correction
        p = p * len(protein_names)
        if p > 1:
            p = 1
        print(f"{column}: r = {r}, p = {p}")
        sns.despine(ax=axs.flat[i])
        axs.flat[i].text(
            s=f"r = {np.round(r, 2)}",
            y=1,
            x=1,
            ha="right",
            va="top",
            transform=axs.flat[i].transAxes,
        )
        if p < 0.05:
            axs.flat[i].text(
                s=f"p = {np.format_float_scientific(p, 1)}*",
                y=0.9,
                x=1,
                ha="right",
                va="top",
                transform=axs.flat[i].transAxes,
                weight="bold",
            )
        else:
            axs.flat[i].text(
                s=f"p = {np.format_float_scientific(p, 1)}",
                y=0.9,
                x=1,
                ha="right",
                va="top",
                transform=axs.flat[i].transAxes,
            )
    plt.tight_layout()
    sns.move_legend(axs.flat[-2], "center right", bbox_to_anchor=(2.0, 0.5))
    axs.flat[-1].remove()
    plt.savefig("out/analysis/paper_figures/fig_s10_protein_amp_corr.svg")
    plt.close()
    print("Done with Figure S10.")


def make_figure_videos(data: pd.DataFrame, metadata: dict[str, Any]):
    """
    One-off video of sub-generational ampC expression and cell death at MIC.
    """
    data["AmpC monomer (\u03bcM)"] = (
        data["AmpC monomer"]
        / (data.loc[:, "Volume"] * 0.2)
        * COUNTS_PER_FL_TO_NANOMOLAR
        / 1000
    )
    make_tag_video(
        data=data,
        metadata=metadata,
        tag_colors={"AmpC monomer (\u03bcM)": (0.6, 1, 1)},
        out_prefix="ampicillin_2mg_L",
    )


def load_exp_data(experiment_ids):
    data = []
    metadata = {}
    for exp_id in experiment_ids:
        exp_data = pd.read_csv(
            f"data/colony_data/sim_dfs/{exp_id}.csv",
            dtype={"Agent ID": str, "Seed": int},
            index_col=0,
        )
        if exp_data.loc[:, "Dry mass"].iloc[-1] == 0:
            exp_data = exp_data.iloc[:-1, :]
        data.append(exp_data)
        with open(f"data/colony_data/sim_dfs/{exp_id}_metadata.json", "r") as f:
            metadata = deep_merge(metadata, json.load(f))
    data = pd.concat(data)
    # Convert strings to dictionaries
    data["Boundary"] = data["Boundary"].apply(ast.literal_eval)
    initial_external_tet = []
    initial_external_amp = []
    for condition in data["Condition"].unique():
        cond_data = data.loc[data.loc[:, "Condition"] == condition, :]
        if condition == "Glucose":
            initial_external_tet += [0] * len(cond_data)
            initial_external_amp += [0] * len(cond_data)
            continue
        curr_len = len(initial_external_tet)
        for boundary_data in cond_data.loc[:, "Boundary"]:
            # Assumes only one antibiotic is used at a time
            tet_conc = boundary_data["external"]["tetracycline"]
            if tet_conc != 0:
                initial_external_tet += [tet_conc] * len(cond_data)
                initial_external_amp += [0] * len(cond_data)
                break
            amp_conc = boundary_data["external"]["ampicillin[p]"]
            if amp_conc != 0:
                initial_external_amp += [amp_conc] * len(cond_data)
                initial_external_tet += [0] * len(cond_data)
                break
        if len(initial_external_tet) == curr_len:
            initial_external_tet += [0] * len(cond_data)
            initial_external_amp += [0] * len(cond_data)
    data["Initial external tet."] = initial_external_tet
    data["Initial external amp."] = initial_external_amp
    return data, metadata


def main():
    tet_local = [
        "2022-12-08_01-13-41_036971+0000",
        "2022-12-08_01-37-02_043920+0000",
        "2022-12-08_01-37-17_383563+0000",
        "2022-12-08_01-37-25_382616+0000",
        "2022-12-08_01-37-31_999399+0000",
        "2022-12-08_01-37-38_566402+0000",
        "2022-12-08_01-37-44_216110+0000",
        "2022-12-08_01-37-52_725211+0000",
        "2022-12-08_01-37-57_809101+0000",
        "2022-12-08_01-38-03_635076+0000",
        "2022-12-08_01-38-09_020029+0000",
    ]
    conditions = {
        "1a": ["Glucose"],
        "s3a": ["Glucose"],
        "2a": ["Glucose"],
        "2b_d": ["Glucose"],
        "s1": ["Glucose"],
        "s5": ["Glucose"],
        "s3b": ["Glucose"],
        "3l": [
            "Glucose",
            "Tetracycline (0.5 mg/L)",
            "Tetracycline (1 mg/L)",
            "Tetracycline (1.5 mg/L)",
            "Tetracycline (2 mg/L)",
            "Tetracycline (4 mg/L)",
        ],
        "3d_and_3m": [
            "Glucose",
            "Tetracycline (0.5 mg/L)",
            "Tetracycline (1 mg/L)",
            "Tetracycline (1.5 mg/L)",
            "Tetracycline (2 mg/L)",
            "Tetracycline (4 mg/L)",
        ],
        "3f_h": ["Glucose", "Tetracycline (1.5 mg/L)"],
        "3i_k": ["Glucose", "Tetracycline (1.5 mg/L)"],
        "s6b": ["Glucose", "Tetracycline (1.5 mg/L)"],
        "s6a": [str(i) for i in range(11)],
        "s7": [
            "Glucose",
            "Tetracycline (0.5 mg/L)",
            "Tetracycline (1 mg/L)",
            "Tetracycline (1.5 mg/L)",
            "Tetracycline (2 mg/L)",
            "Tetracycline (4 mg/L)",
        ],
        "s8": ["Glucose", "Tetracycline (1.5 mg/L)"],
        "4l": [
            "Glucose",
            "Ampicillin (0.5 mg/L)",
            "Ampicillin (1 mg/L)",
            "Ampicillin (1.5 mg/L)",
            "Ampicillin (2 mg/L)",
            "Ampicillin (4 mg/L)",
        ],
        "4c": ["Glucose", "Ampicillin (2 mg/L)"],
        "4m": ["Glucose", "Ampicillin (2 mg/L)"],
        "4n": [
            "Ampicillin (0.5 mg/L)",
            "Ampicillin (1 mg/L)",
            "Ampicillin (1.5 mg/L)",
            "Ampicillin (2 mg/L)",
            "Ampicillin (4 mg/L)",
        ],
        "videos": ["Ampicillin (2 mg/L)"],
    }
    seeds = {
        "1a": [10000],
        "s3a": [10000],
        "2a": [10000],
        "2b_d": [10000],
        "s1": [10000],
        "s5": [10000],
        "s3b": [10000],
        "3l": [0],
        "3d_and_3m": [0],
        "3f_h": [0],
        "3i_k": [0],
        "s6b": [0, 100, 10000],
        "s6a": [0],
        "s7": [0],
        "s8": [0],
        "4l": [0],
        "4c": [0],
        "4m": [0],
        "4n": [0],
        "videos": [0],
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fig_ids",
        "-f",
        help="List of lowercase figure IDs to create (e.g. 1a). \
            Default is all.",
        nargs="+",
        choices=list(seeds.keys()),
    )
    args = parser.parse_args()

    if args.fig_ids is None:
        args.fig_ids = conditions.keys() - {"3f"}

    ids_to_load = []
    for fig_id in args.fig_ids:
        if fig_id == "3f":
            ids_to_load.extend(tet_local)
            continue
        for condition in conditions[fig_id]:
            for seed in seeds[fig_id]:
                ids_to_load.append(EXPERIMENT_ID_MAPPING[condition][seed])
    # De-duplicate IDs while preserving order
    ids_to_load = list(dict.fromkeys(ids_to_load))

    data, metadata = load_exp_data(ids_to_load)

    for fig_id in args.fig_ids:
        filter = np.isin(data.loc[:, "Condition"], conditions[fig_id]) & np.isin(
            data.loc[:, "Seed"], seeds[fig_id]
        )
        fig_data = data.loc[filter, :].copy()
        globals()[f"make_figure_{fig_id}"](fig_data, metadata)


if __name__ == "__main__":
    main()
