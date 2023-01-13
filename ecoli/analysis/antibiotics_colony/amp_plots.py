import argparse
import os
import pickle

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ecoli.analysis.antibiotics_colony import COUNTS_PER_FL_TO_NANOMOLAR
from ecoli.analysis.antibiotics_colony.plot_utils import (
    BG_GRAY,
    HIGHLIGHT_BLUE,
    prettify_axis,
)
from ecoli.analysis.antibiotics_colony.timeseries import (
    plot_field_snapshots,
    plot_timeseries,
)
from ecoli.library.parameters import param_store


PERIPLASMIC_VOLUME_FRACTION = 0.2


def load_data(
    glc_data,
    glc_metadata,
    amp_data,
    amp_metadata,
    additional_amp_concs=None,
    verbose=False,
):
    # Load data from additional ampicillin concentrations
    # (Do this first to try and avoid memory usage issues)
    add_amp_data = None
    KEEP_COLS = ["Time", "Seed", "Condition", "Agent ID", "Dry mass"]
    if additional_amp_concs:
        add_amp_data = []
        for file in tqdm(
            additional_amp_concs, desc="Loading additional Ampicillin data..."
        ):
            with open(file, "rb") as f:
                add_amp_data.append(pickle.load(f)[KEEP_COLS])

        add_amp_data = pd.concat(add_amp_data)

    # Load glc data
    if verbose:
        print("Loading Glucose data...")
    with open(glc_data, "rb") as f:
        glc_data = pickle.load(f)

    # Load glc metadata
    if verbose:
        print("Loading Glucose metadata...")
    with open(glc_metadata, "rb") as f:
        glc_metadata = pickle.load(f)

    # Load amp data
    if verbose:
        print("Loading Ampicillin data...")
    with open(amp_data, "rb") as f:
        amp_data = pickle.load(f)

    # Load amp metadata
    if verbose:
        print("Loading Ampicillin metadata...")
    with open(amp_metadata, "rb") as f:
        amp_metadata = pickle.load(f)

    # Validate data:
    # - glc_data, glc_metadata must be in Glucose condition,
    #   amp_data, amp_metadata, additional amp data must be in (one of) Ampicillin condition(s)
    # - Condition must match exactly between amp_data and amp_metadata (including dosage)
    # - Conditions in additional amp data must not match condition of amp_data, amp_metadata
    # - seed must match across glc_data, glc_metadata, amp_data, amp_metadata, additional amp data
    if verbose:
        print("Validating and cleaning data...")

    assert isinstance(glc_data, pd.DataFrame)
    assert isinstance(amp_data, pd.DataFrame)
    assert isinstance(glc_metadata, dict)
    assert isinstance(amp_metadata, dict)
    assert (
        glc_data["Condition"].unique().size
        == amp_data["Condition"].unique().size
        == len(glc_metadata.keys())
        == len(amp_metadata.keys())
        == 1
    ), "One of glc_data, amp_data, glc_metadata, amp_data has data for more than one condition!"

    # - glc_data, glc_metadata must be in Glucose condition,
    #   amp_data, amp_metadata additional amp data must be in (one of) Ampicillin condition(s)

    glc_data_condition = glc_data["Condition"].unique()[0]
    glc_metadata_condition = list(glc_metadata.keys())[0]
    amp_data_condition = amp_data["Condition"].unique()[0]
    amp_metadata_condition = list(amp_metadata.keys())[0]
    add_amp_conditions = (
        add_amp_data["Condition"].unique() if add_amp_data is not None else []
    )

    assert "Glucose" in glc_data_condition, "glc_data was not from Glucose condition!"
    assert (
        "Glucose" in glc_metadata_condition
    ), "glc_metadata was not from Glucose condition!"
    assert (
        "Ampicillin" in amp_data_condition
    ), "amp_data was not from Ampicillin condition!"
    assert (
        "Ampicillin" in amp_metadata_condition
    ), "amp_metadata was not from Ampicillin condition!"
    for condition in add_amp_conditions:
        assert "Ampicillin" in condition, (
            f"Condition {condition} in additional ampicillin"
            "data was not from Ampicillin condition!"
        )

    # - Condition must match exactly between amp_data and amp_metadata (including dosage)

    assert (
        amp_data_condition == amp_metadata_condition
    ), "Condition does not match between amp_data and amp_metadata!"

    # - Conditions in additional amp data must not match condition of amp_data, amp_metadata
    assert (
        amp_data_condition not in add_amp_conditions
    ), "Additional ampicillin data cannot contain the original condition!"

    # - seed must match across glc_data, glc_metadata, amp_data, amp_metadata, additional amp data
    assert (
        glc_data["Seed"].unique().size
        == amp_data["Seed"].unique().size
        == (add_amp_data["Seed"].unique().size if add_amp_data is not None else 1)
        == len(glc_metadata[glc_data_condition].keys())
        == len(amp_metadata[amp_data_condition].keys())
        == 1
    ), (
        "One of glc_data, amp_data, glc_metadata, amp_data, additional amp data "
        "has data for more than one seed!"
    )

    glc_data_seed = glc_data["Seed"].unique()[0]
    amp_data_seed = amp_data["Seed"].unique()[0]
    add_amp_seed = (
        add_amp_data["Seed"].unique()[0] if add_amp_data is not None else amp_data_seed
    )
    glc_metadata_seed = list(glc_metadata[glc_metadata_condition].keys())[0]
    amp_metadata_seed = list(amp_metadata[amp_metadata_condition].keys())[0]

    assert (
        glc_data_seed
        == amp_data_seed
        == add_amp_seed
        == glc_metadata_seed
        == amp_metadata_seed
    ), (
        "Seeds do not match across glc_data, glc_metadata, "
        "amp_data, amp_metadata, additional amp data!"
    )

    # Merge dataframes from before/after addition of ampicillin
    amp_data = pd.concat([glc_data[glc_data.Time < amp_data.Time.min()], amp_data])

    # Figure out which cells died and when
    def died(lineage, agent_ids):
        d1, d2 = lineage + "0", lineage + "1"
        return (d1 not in agent_ids) and (d2 not in agent_ids)

    # Get all cell ids, set of cells that died, and time of death for those cells
    for data in [glc_data, amp_data]:
        unique_ids = data["Agent ID"].unique()
        dead_ids = list(filter(lambda id: died(id, unique_ids), unique_ids))
        time_of_death = {
            id: data["Time"][data["Agent ID"] == id].max() for id in dead_ids
        }

        # Remove cells still present at sim end from dead
        time_of_death = {
            id: time for id, time in time_of_death.items() if time != data.Time.max()
        }
        dead_ids = list(time_of_death.keys())

        # Create columns for whether a cell died, and the time of death where applicable
        data["Died"] = data["Agent ID"].isin(dead_ids)
        data["Time of Death"] = data["Agent ID"].map(
            lambda id: time_of_death.get(id, None)
        )

    # Add additional columns for periplasmic volume,
    # concentration of AmpC in the periplasm
    for data in [glc_data, amp_data]:
        data["Periplasmic Volume"] = PERIPLASMIC_VOLUME_FRACTION * data["Volume"]
        data["AmpC conc"] = (
            data["AmpC monomer"] / data["Periplasmic Volume"]
        ) * COUNTS_PER_FL_TO_NANOMOLAR

    return glc_data, amp_data, {**glc_metadata, **amp_metadata}, add_amp_data


def agent_summary(data, var_summarizers):
    cols = []

    for var, summarization in var_summarizers.items():
        if summarization == "mean":
            summarizer = lambda c: c.mean()
        elif summarization == "min":
            summarizer = lambda c: c.min()
        elif summarization == "max":
            summarizer = lambda c: c.max()
        elif callable(summarization):
            summarizer = summarization
        else:
            raise ValueError(f"{summarization} is not a recognized summarization.")

        cols.append(summarizer(data.groupby("Agent ID")[var]))

    return pd.concat(cols, axis=1)


def make_figures(
    glc_data,
    amp_data,
    metadata,
    add_amp_data,
    verbose,
    as_svg=False,
    outdir="out/figure_4",
):

    # Prepare to output figures
    os.makedirs(outdir, exist_ok=True)
    ext = "svg" if as_svg else "png"

    # Get time when ampicillin added
    amp_time = amp_data["Time"][
        amp_data["Condition"].map(lambda s: "Ampicillin" in s)
    ].min()
    max_time = amp_data["Time"].max()

    # Plot snapshots ==============================================================================
    if verbose:
        print("Making fields plot...")
    plot_field_snapshots(amp_data, metadata, highlight_color=HIGHLIGHT_BLUE)

    # Plot colony mass vs time with deaths ========================================================
    if verbose:
        print("Making colony mass vs. time (with marked deaths) plot...")

    fig, ax = plt.subplots()
    timeseries_death_plot(
        amp_data,
        "Dry mass",
        highlight_lineage=str(amp_data["Agent ID"].iloc[-1]),
        ax=ax,
    )

    # Relabel time axis to use time after addition of ampicillin
    amp_time_hrs, max_time_hrs = amp_time / 3600, max_time / 3600
    new_ticks = np.concatenate(
        (
            [0],
            np.arange(amp_time_hrs, 1, -1)[::-1],
            np.arange(amp_time_hrs, max_time_hrs, 1)[1:],
            # [max_time_hrs],
        )
    )
    new_tick_labels = [f"{t - amp_time_hrs:.0f}" for t in new_ticks]
    new_tick_labels[0] = f"{-amp_time_hrs:.1f}"
    ax.set_xticks(new_ticks, labels=new_tick_labels)

    fig.set_size_inches(5, 1.25)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"colony_mass_timeseries.{ext}"))

    # Plot minute-scale timeseries ================================================================
    minute_vars = {
        "Periplasmic ampicillin": ("Periplasmic\n ampicillin", False),
        "AmpC conc": ("AmpC (nM)", False),
        "Active fraction PBP1a": ("Frac active\n PBP1a", False),
    }
    fig, axs = plt.subplots(1, len(minute_vars))
    for ax, (minute_var, (label, to_conc)) in zip(axs, minute_vars.items()):
        if verbose:
            print(f"Plotting {minute_var} timeseries...")

        minute_scale_plot(amp_data, minute_var, dt=2, ax=ax, to_conc=to_conc)
        ticklabs = [lab.get_text() for lab in ax.get_xticklabels()]

        # Remove x label, make sure axis visible
        xaxis = ax.axes.get_xaxis()
        xaxis.set_visible(True)
        xaxis.get_label().set_visible(False)

        ax.set_ylabel(label)
        prettify_axis(
            ax,
            xticks=ax.get_xticks(),
            xlabel_as_tick=False,
            ylabel_as_tick=False,
            tick_format_y="{:.1f}",
        )
        ax.set_xticklabels(ticklabs)
        ax.spines.bottom.set(bounds=ax.get_xlim(), visible=True, color="0", alpha=1)
        ax.spines.left.set(bounds=ax.get_ylim(), visible=True, color="0", alpha=1)

    fig.set_size_inches(5, 1.5)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"minute_scale_timeseries.{ext}"))

    # Plot hour-scale timeseries ==================================================================

    # Clean data
    amp_data["Max hole area (nm^2)"] = [len(x) for x in amp_data["Max hole size"]]
    basal_area = (
        param_store.get(("cell_wall", "inter_strand_distance"))
        + param_store.get(("cell_wall", "disaccharide_width"))
    ) * param_store.get(("cell_wall", "disaccharide_height"))
    amp_data["Max hole area (nm^2)"] = (
        amp_data["Max hole area (nm^2)"] * amp_data["Extension factor"] * basal_area
    )

    hour_variables = ["Porosity", "Extension factor", "Max hole area (nm^2)"]
    fig, axs = plt.subplots(1, len(hour_variables))
    for ax, hour_variable in zip(axs, hour_variables):
        if verbose:
            print(f"Plotting {hour_variable} timeseries...")

        hour_scale_plot(amp_data, hour_variable, ax=ax, to_conc=False)

        xaxis = ax.axes.get_xaxis()
        xaxis.get_label().set_visible(False)

        prettify_axis(ax, xticks=ax.get_xticks(), xlabel_as_tick=False)

        new_ticks = np.array([0, amp_time_hrs, max_time_hrs])
        new_tick_labels = [f"{t - amp_time_hrs:.0f}" for t in new_ticks]
        new_tick_labels[0] = f"{-amp_time_hrs:.1f}"
        ax.set_xticks(new_ticks, labels=new_tick_labels)

        ax.spines.bottom.set(bounds=ax.get_xlim(), visible=True, color="0", alpha=1)
        ax.spines.left.set(bounds=ax.get_ylim(), visible=True, color="0", alpha=1)

    fig.set_size_inches(5, 1.5)
    fig.savefig(os.path.join(outdir, f"hour_scale_timeseries.{ext}"))

    # Plot beta-lactamase, AcrAB-TolC... vs. death ================================================

    death_factor_formatting = {
        "AmpC conc": ("{:.1f}", False),
        "Periplasmic ampicillin": ("{:.2e}", False),
        "PBP1a complex": ("{:.1f}", True),
        "PBP1b gamma complex": ("{:.1f}", True),
    }
    for factor, (fmt, to_conc) in death_factor_formatting.items():
        if verbose:
            print(f"Making histogram plot for {factor}...")

        fig, ax = plt.subplots()
        _, _, bins = hist_by_death_plot(amp_data, factor, counts_to_conc=to_conc, ax=ax)

        # prettify_axis(
        #     ax,
        #     xlim=[bins[0], bins[-1]],
        #     ylim=np.round(ax.get_ylim()),
        #     tick_format_x=fmt,
        #     tick_format_y="{:.0f}",
        # )
        leg = ax.get_legend()
        if leg:
            leg.remove()

        fig.set_size_inches(1.25, 1.75)
        fig.savefig(
            os.path.join(outdir, f"{factor}.{ext}"),
            bbox_inches="tight",
        )

    # Pairplot ====================================================================================
    # Define transformation to get mean concentration of a variable
    # if verbose:
    #     print("Plotting pairplot...")

    # def mean_concentration(amp_var, data):
    #     data = data.copy()
    #     data[amp_var] = (
    #         data[amp_var] / data["Boundary"].map(lambda d: d["volume"])
    #     ) * COUNTS_PER_FL_TO_NANOMOLAR

    #     result = data.groupby("Agent ID")[amp_var].mean()
    #     del data

    #     return result

    # def mean_doubling_time(amp_var, data):
    #     data = data.copy()

    #     result = {"Agent ID": [], amp_var: []}
    #     for name, group in data.groupby("Agent ID"):
    #         dt = np.diff(group["Time"])
    #         doubling_time = (
    #             np.log2(group[amp_var].to_numpy()[1:] / group[amp_var].to_numpy()[:-1])
    #             / dt
    #         )

    #         result["Agent ID"].append(name)
    #         result[amp_var].append(np.mean(doubling_time))

    #     del data
    #     return pd.DataFrame(result).set_index("Agent ID")

    # # TODO: divide AcrAB-TolC by *periplasmic volume*, not total cell volume
    # pairplot_vars = {
    #     "AcrAB-TolC": mean_concentration,
    #     "Periplasmic ampicillin": lambda var, data: data.groupby("Agent ID")[
    #         var
    #     ].mean(),
    #     "AmpC monomer": mean_concentration,
    #     "PBP1a complex": mean_concentration,
    #     "PBP1b gamma complex": mean_concentration,
    #     # "Growth rate": lambda var, data: data.groupby("Agent ID")[var].mean(),
    #     "Dry mass": mean_doubling_time,
    #     "Active ribosomes": mean_concentration,
    # }

    # # Get data only after ampicillin added, for cells that appear for more than one timestep
    # plot_data = amp_data[amp_data["Time"] >= amp_time]
    # include = plot_data["Agent ID"].value_counts()[plot_data["Agent ID"]] > 1
    # plot_data = plot_data[include.to_numpy()]

    # g = pairplot(plot_data, pairplot_vars)
    # g.figure.set_size_inches(10, 10)
    # g.figure.savefig(
    #     os.path.join(outdir, f"pairplot.{ext}"),
    #     bbox_inches="tight",
    # )
    # plt.close(g.figure)

    # Plot colony mass vs concentration => MIC ====================================================

    if verbose:
        print("Plotting colony mass vs. time by ampicillin concentration...")

    fig, ax = plt.subplots()
    amp_conc_sweep(glc_data, amp_data, add_amp_data, ax=ax)
    ax.set_xticks(new_ticks, labels=new_tick_labels)
    prettify_axis(ax, ylabel_as_tick=False)
    ax.set_yscale("log")
    leg = ax.get_legend()
    if leg:
        leg.remove()
    fig.set_size_inches(2, 2)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"amp_conc_sweep.{ext}"))

    # Run logistic regression to explain death ====================================================

    if verbose:
        print("Running logistic regression...")

    log_vars = {
        "AcrAB-TolC": "mean",
        # "Periplasmic ampicillin",
        # "Active ribosomes",
        # "Active RNAP",
        # "Murein tetramer",
        "PBP1a complex": "mean",
        # "PBP1a mRNA",
        # "PBP1b alpha complex",
        # "PBP1b mRNA",
        "PBP1b gamma complex": "mean",
        "AmpC conc": "mean",
        # "ampC mRNA",
        # "Unincorporated murein",
        # "Incorporated murein",
        # "Shadow murein",
        # "Porosity",
        # "Active fraction PBP1a",
        # "Active fraction PBP1b",
        "Died": "max",
    }

    # Create summary data for each agent,
    # Excluding agents present at the end of the simulation
    # (since these may not have had enough time to die)
    present_at_end = (
        amp_data.groupby("Agent ID")["Time"].max() == amp_data["Time"].max()
    )
    final_cells = present_at_end.index[present_at_end]

    log_reg_data = agent_summary(
        amp_data[~np.isin(amp_data["Agent ID"], final_cells)], log_vars
    )
    log_reg_data = sm.add_constant(log_reg_data)

    log_reg, confusion = logistic_explain_death(log_reg_data, random_state=1342534)

    # Output text log
    with open(os.path.join(outdir, "logistic_regression_full.txt"), "w") as f:
        f.write(str(log_reg.summary()))
        f.write("\n")

        # Confusion matrix
        f.write("Confusion matrix:\n")
        f.write(str(confusion))

    # Run logistic regression again, with AmpC as only predictor

    # Create training data, test data
    log_reg_data_2 = log_reg_data[["const", "AmpC conc", "Died"]]
    log_reg_2, confusion = logistic_explain_death(log_reg_data_2, random_state=1342534)

    # Output text log
    with open(os.path.join(outdir, "logistic_regression_reduced.txt"), "w") as f:
        f.write(str(log_reg_2.summary()))
        f.write("\n")

        # Confusion matrix
        f.write("Confusion matrix:\n")
        f.write(str(confusion))

    # Plot separation for AmpC
    fig, ax = plt.subplots()
    x = log_reg_data[["const", "AmpC conc"]]
    min_ampC, max_ampC = x[["AmpC conc"]].min(), x[["AmpC conc"]].max()
    ax.scatter(
        x[["AmpC conc"]],
        log_reg_data_2[["Died"]],
        color=HIGHLIGHT_BLUE,
        alpha=0.5,
    )
    ampC = np.linspace(min_ampC, max_ampC, 100)
    ax.plot(
        ampC,
        log_reg_2.predict(sm.add_constant(ampC.reshape(-1, 1))),
        color=HIGHLIGHT_BLUE,
    )

    ax.set_xlabel("AmpC (nM)")
    ax.set_ylabel("Died")
    prettify_axis(ax)
    ax.set_yticks([0, 1], labels=["0", "1"])
    fig.set_size_inches(1.5, 1.75)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"logistic_regression.{ext}"))

    # Violin plots showing death/survival =========================================================

    if verbose:
        print("Making violin plot...")

    violin_vars = {
        "AcrAB-TolC": "mean",
        "PBP1a complex": "mean",
        "PBP1b gamma complex": "mean",
        "AmpC conc": "mean",
        "OmpF complex": "mean",
    }
    violin_cols = list(violin_vars.keys())

    violin_data = amp_data[violin_cols].div(amp_data["Volume"], axis=0)
    violin_data = pd.concat(
        [amp_data["Agent ID"], violin_data, amp_data["Died"]], axis=1
    )
    violin_data = agent_summary(violin_data, {**violin_vars, **{"Died": "max"}})

    # convert to concentrations
    for var in set(violin_cols) - {"AmpC conc"}:
        violin_data[var] *= COUNTS_PER_FL_TO_NANOMOLAR

    # center data for each molecule about its mean (use z-score??)

    violin_data[violin_cols] = violin_data[violin_cols].apply(
        zscore
    )  # .apply(lambda s: s - s.mean())

    # pivot to long format, in preparation for plotting
    violin_data_long = violin_data.reset_index().melt(
        id_vars=["Agent ID", "Died"], var_name="Molecule", value_name="Concentration"
    )

    fig, ax = plt.subplots()
    sns.violinplot(
        violin_data_long,
        x="Molecule",
        y="Concentration",
        hue="Died",
        palette={True: BG_GRAY, False: HIGHLIGHT_BLUE},
        inner=None,
        split=True,
        legend=False,
        linewidth=0,
        ax=ax,
    )

    # Remove legend
    ax.legend([], [], frameon=False)

    # prettify_axis(
    #     ax, xticks="all", tick_format_x="{}", xlabel_as_tick=False, ylabel_as_tick=False
    # )
    fig.set_size_inches(3, 2)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"violin_plot.{ext}"))

    # Lysis ratio vs ampicillin ===================================================================

    if verbose:
        print("Doing lysis ratio analysis...")

    # In the following, we define "time of lysis" as the time when >= 50% of the mass
    # in a phylogenetic tree is lost due to lysis

    # Collate data from all ampicillin conditions, get roots (cells present at time
    # when ampcillin is introduced
    all_amp_data = pd.concat([amp_data, add_amp_data], axis=0)
    roots = all_amp_data.groupby("Agent ID")["Time"].agg(
        lambda s: s.min() <= amp_time <= s.max()
    )
    roots = roots.index[roots]

    plot_x = []
    plot_y = []
    for condition in all_amp_data.Condition.unique():
        if condition.startswith("Gl"):
            continue

        print(condition)
        print("========")

        for root in roots:
            cond_data = all_amp_data[all_amp_data["Condition"] == condition]

            amp_lineage = cond_data[
                cond_data["Agent ID"].map(lambda s: s.startswith(root))
            ]
            glc_lineage = glc_data[
                glc_data["Agent ID"].map(lambda s: s.startswith(root))
            ]
            amp_mass_timeseries = (
                amp_lineage.groupby("Time")["Dry mass"].sum().sort_index()
            )
            glc_mass_timeseries = (
                glc_lineage.groupby("Time")["Dry mass"].sum().sort_index()
            )

            mass_timeseries = pd.concat(
                [amp_mass_timeseries, glc_mass_timeseries], axis=1, keys=["amp", "glc"]
            )

            candidate_times = np.where(
                (mass_timeseries["amp"] / mass_timeseries["glc"] <= 0.5)
            )[0]
            lysis_time = candidate_times.min() if candidate_times.size > 0 else max_time

            plot_x.append(condition)
            plot_y.append(lysis_time)

            print(f"\t{root}: {lysis_time}")

    # Plot lysis ratio vs. ampcillin condition

    plot_x = [float(x[len("Ampicillin (") : -len(" mg/L)")]) for x in plot_x]

    generation_time = (
        glc_data.groupby("Agent ID")["Time"]
        .agg(["min", "max"])
        .query(f"max != {max_time}")
        .diff(axis=1)["max"]
        .mean()
    )
    lysis_ratios = [generation_time / lysis_time for lysis_time in plot_y]

    plot_data = pd.DataFrame({"Ampicillin (mg/L)": plot_x, "Lysis Ratio": lysis_ratios})
    means = plot_data.groupby("Ampicillin (mg/L)").mean().reset_index()

    fig, ax = plt.subplots()
    ax.scatter(plot_x, lysis_ratios, color=HIGHLIGHT_BLUE, alpha=0.5)
    ax.plot(means["Ampicillin (mg/L)"], means["Lysis Ratio"], "--", color="0.4")
    ax.set_ylabel("Lysis ratio")
    # ax.set_ylabel(r"Lysis Ratio $=\frac{\text{mean generation time}}{\text{time to loss of >50% of lineage mass}}$")
    ax.set_xlabel("Ampicillin (mg/L)")
    prettify_axis(ax)
    fig.set_size_inches(1.5, 1.75)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"lysis_ratio.{ext}"))


def timeseries_death_plot(data, var, highlight_lineage=None, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    data = data.copy()

    plot_timeseries(
        data,
        axes=[ax],
        columns_to_plot={var: HIGHLIGHT_BLUE},
        highlight_lineage=highlight_lineage,
        conc=False,
        mark_death=True,
    )

    return ax


def minute_scale_plot(data, var, dt=5, highlight_lineage=None, ax=None, to_conc=False):
    if not ax:
        _, ax = plt.subplots()

    amp_time = data["Time"][data["Condition"].map(lambda s: "Ampicillin" in s)].min()

    plot_data = data[
        (amp_time - 60 * dt <= data["Time"]) & (data["Time"] <= amp_time + 60 * dt)
    ]
    plot_timeseries(
        data=plot_data,
        axes=[ax],
        columns_to_plot={var: HIGHLIGHT_BLUE},
        highlight_lineage=str(plot_data["Agent ID"].iloc[-1]),
        filter_time=False,
        background_alpha=0.5,
        background_linewidth=0.3,
        conc=to_conc,
    )

    ax.set_xticks([(amp_time + d) / 3600 for d in np.arange(-60 * dt, 60 * dt + 1, 60)])
    ax.set_xticklabels([int(d) for d in range(-dt, dt + 1)])

    return ax


def hour_scale_plot(data, var, highlight_lineage=None, ax=None, to_conc=False):
    if not ax:
        _, ax = plt.subplots()

    amp_time = data["Time"][data["Condition"].map(lambda s: "Ampicillin" in s)].min()

    plot_timeseries(
        data=data,
        axes=[ax],
        columns_to_plot={var: HIGHLIGHT_BLUE},
        highlight_lineage=(
            highlight_lineage if highlight_lineage else str(data["Agent ID"].iloc[-1])
        ),
        filter_time=False,
        background_alpha=0.5,
        background_linewidth=0.3,
        conc=to_conc,
    )

    return ax


def hist_by_death_plot(
    data,
    var,
    var_summarize=lambda var: var.mean(),
    counts_to_conc=True,
    ax=None,
    xlabel=None,
):
    data = data.copy()

    if ax is None:
        _, ax = plt.subplots()

    # Convert from counts to concentration if requested
    if counts_to_conc:
        data[var] = (data[var] / data["Volume"]) * COUNTS_PER_FL_TO_NANOMOLAR

    # Prepare data for plotting
    hist_data = pd.concat(
        [
            var_summarize(data.groupby("Agent ID")[var]),
            data.groupby("Agent ID")[["Died"]].max(),
        ],
        axis=1,
    )

    # Replace Died=True with "Lysed", Died=False with "Survived"
    hist_data["Died"] = hist_data["Died"].replace({False: "Survived", True: "Lysed"})

    # Plot histogram
    sns.histplot(
        x=var,
        hue="Died",
        data=hist_data,
        palette={"Lysed": BG_GRAY, "Survived": HIGHLIGHT_BLUE},
        ax=ax,
        edgecolor=None,
    )

    if xlabel is None:
        xlabel = f"{var}{' (nM)' if counts_to_conc else ''}"

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cells")

    hist, bins = np.histogram(hist_data[var])

    return ax, hist, bins


def ampicillin_pca(
    data,
    vars,
    var_summarizers={},
    outcome="Died",
    counts_to_conc=True,
    ax=None,
):
    data = data.copy()

    if ax is None:
        _, ax = plt.subplots()

    # Convert from counts to concentration if requested
    if counts_to_conc:
        data[var] = (data[var] / data["Volume"]) * COUNTS_PER_FL_TO_NANOMOLAR

    # Summarize timeseries data for each agent using the specified R^T => R summarizer,
    # using maximum for the "outcome" variable unless specified
    var_summarizers[outcome] = var_summarizers.get(outcome, lambda x: x.max())

    agent_summary_data = pd.concat(
        [
            var_summarizers.get(var, lambda x: x.mean())(data.groupby("Agent ID")[var])
            for var in vars + [outcome]
        ],
        axis=1,
    )

    scaled_summary_data = StandardScaler().fit_transform(
        agent_summary_data.drop("Died", axis=1).values
    )

    pca = PCA(n_components=2)
    X_r = pca.fit_transform(scaled_summary_data)

    sns.scatterplot(x=X_r[:, 0], y=X_r[:, 1], hue=agent_summary_data["Died"], ax=ax)

    return pca, X_r, ax


def logistic_explain_death(data, outcome="Died", test_size=0.3, random_state=None):
    # Create training data, test data
    x_data = data[[x for x in data.columns if x != outcome]]
    y_data = data[outcome]
    x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(
        x_data, y_data, test_size=0.3, random_state=1342534
    )

    model = sm.Logit(y_training_data, x_training_data).fit()
    yhat = model.predict(x_test_data)
    predictions = list(map(round, yhat))

    return model, confusion_matrix(y_test_data, predictions)


def pairplot(
    data,
    variable_transform_map={},
    hue="Died",
):
    # Summarize timeseries data for each agent using the specified R^T => R summarizer,
    # using maximum for the hue variable unless specified
    variable_transform_map[hue] = variable_transform_map.get(
        hue, lambda x, data: data.groupby("Agent ID")[x].max()
    )

    agent_summary_data = pd.concat(
        [summarizer(var, data) for var, summarizer in variable_transform_map.items()],
        axis=1,
    )

    import ipdb

    ipdb.set_trace()

    grid = sns.pairplot(
        agent_summary_data,
        vars=[v for v in variable_transform_map.keys() if v != hue],
        hue=hue,
        palette={False: HIGHLIGHT_BLUE, True: BG_GRAY},
        corner=True,
    )

    return grid


def amp_conc_sweep(glc_data, amp_data, add_amp_data, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    amp_conc_vs_mass_data = pd.concat(
        [
            glc_data.groupby(["Condition", "Time"]).sum("Dry mass"),
            amp_data.groupby(["Condition", "Time"]).sum("Dry mass"),
            add_amp_data.groupby(["Condition", "Time"]).sum("Dry mass"),
        ],
        join="inner",
    )

    # Time to hours
    amp_conc_vs_mass_data = amp_conc_vs_mass_data.reset_index()
    amp_conc_vs_mass_data.Time /= 60 * 60

    # Create color palette
    condition_palette = {
        cond: float(cond[len("Ampicillin (") : -len(" mg/L)")])
        if "Ampicillin" in cond
        else 0
        for cond in amp_conc_vs_mass_data["Condition"].unique()
    }
    max_color = max(condition_palette.values())
    condition_palette = {
        cond: tuple((color / max_color) * np.array(HIGHLIGHT_BLUE))
        for cond, color in condition_palette.items()
    }

    sns.lineplot(
        amp_conc_vs_mass_data,
        x="Time",
        y="Dry mass",
        hue="Condition",
        hue_order=list(
            dict(sorted(condition_palette.items(), key=lambda kv: kv[1])).keys()
        ),
        palette=condition_palette,
        ax=ax,
    )

    ax.set_xlim(0, amp_conc_vs_mass_data.Time.max())
    ax.set_ylim(
        amp_conc_vs_mass_data["Dry mass"].min(), amp_conc_vs_mass_data["Dry mass"].max()
    )
    ax.set_xlabel("Time (hr)")
    ax.set_ylabel("Colony Dry Mass")

    return ax


def cli():
    parser = argparse.ArgumentParser(
        "Generate analysis plots for ampicillin colony sims."
    )

    parser.add_argument(
        "glc_data",
        type=str,
        help="Locally saved dataframe file for glucose (before addition of ampicillin.)",
    )

    parser.add_argument(
        "glc_metadata",
        type=str,
        help="Locally saved metadata file for glucose (before addition of ampicillin.)",
    )

    parser.add_argument(
        "amp_data",
        type=str,
        help="Locally saved dataframe file for data following addition of ampicillin.",
    )

    parser.add_argument(
        "amp_metadata",
        type=str,
        help="Locally saved metadata file for data following addition of ampicillin.",
    )

    parser.add_argument(
        "--additional_amp_concs",
        "-a",
        nargs="+",
        default=None,
        type=str,
        help="Data following addition of ampicillin at alternative concentrations.",
    )

    parser.add_argument(
        "--outdir",
        "-d",
        default="out/figure_4",
        help="Directory in which to output the generated figures.",
    )
    parser.add_argument("--svg", "-s", action="store_true", help="Save as svg.")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    return args


def main():
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["svg.fonttype"] = "none"

    options = cli()

    glc_data, amp_data, metadata, add_amp_data = load_data(
        options.glc_data,
        options.glc_metadata,
        options.amp_data,
        options.amp_metadata,
        options.additional_amp_concs,
        options.verbose,
    )

    make_figures(
        glc_data,
        amp_data,
        metadata,
        add_amp_data,
        options.verbose,
        as_svg=options.svg,
        outdir=options.outdir,
    )


if __name__ == "__main__":
    main()
