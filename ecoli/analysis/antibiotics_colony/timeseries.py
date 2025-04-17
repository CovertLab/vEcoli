import itertools
import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from matplotlib.colors import rgb_to_hsv
from mpl_toolkits.axes_grid1 import anchored_artists

from ecoli.analysis.antibiotics_colony import COUNTS_PER_FL_TO_NANOMOLAR, restrict_data
from ecoli.analysis.colony.snapshots import plot_snapshots, plot_tags, make_video


def plot_timeseries(
    data: pd.DataFrame,
    axes: list[plt.Axes],
    columns_to_plot: dict[str, tuple],
    highlight_lineage: str,
    conc: bool = False,
    mark_death: bool = False,
    background_lineages: bool = True,
    filter_time: bool = True,
    background_color: tuple = (0.5, 0.5, 0.5),
    background_alpha: float = 0.5,
    background_linewidth: float = 0.1,
) -> None:
    """Plot selected traces with specific lineage highlighted and others in gray.

    Args:
        data: DataFrame where each row is an agent and each column is a variable
            of interest. Must have these columns: 'Time', 'Death', 'Agent ID',
            'Boundary', 'Condition', and 'Seed'. The first experimental condition
            in the 'Condition' column is treated as a control and plotted in gray.
            Include at most 2 conditions and 1 seed per condition. If more than 1
            condition is supplied, either ensure that they do not share any time
            points or run with the ``restrict_data`` option set to true.
        axes: Columns are plotted sequentially on these Axes.
        columns_to_plot: Dictionary of columns in data to plot sequentially on
            axes. Each column name corresponds to a RGB tuple to color the trace
            of the highlighted lineage on that plot.
        highlight_lineage: Agent ID to plot lineage trace for. Alternatively,
            one of 'mean' or 'median'.
        conc: Whether to normalize data by volume and convert to nM
        mark_death: Mark cells that die with red X on time step of death
        background_lineages: Whether to plot traces for other lineages (gray).
        filter_time: Apply default time filter for ``data`` (take first 11550
            seconds from assumed control condition and 11550-26000 seconds from
            all other conditions)
        background_color: Color used to plot traces for non-highlighted agents
        background_alpha: Alpha used to plot traces for non-highlighted agents
        background_linewidth: Linewidth for non-highlighted agent traces
    """
    columns_to_include = list(
        columns_to_plot.keys() | {"Agent ID", "Condition", "Time"}
    )
    if conc:
        columns_to_include.append("Volume")
    data = data.loc[:, columns_to_include]
    if conc:
        # Convert to concentrations
        data = data.set_index(["Condition", "Time", "Agent ID"])
        data = data.divide(data["Volume"], axis=0).drop(["Volume"], axis=1)
        data = data * COUNTS_PER_FL_TO_NANOMOLAR
        data = data.reset_index()
    # Sort values by time for ease of plotting later
    data = data.sort_values("Time")
    if highlight_lineage == "mean":
        highlight_data = data.groupby(["Condition", "Time"]).mean().reset_index()
        highlight_data["Agent ID"] = highlight_lineage
        background_data = data.copy()
    elif highlight_lineage == "median":
        highlight_data = data.groupby(["Condition", "Time"]).median().reset_index()
        highlight_data["Agent ID"] = highlight_lineage
        background_data = data.copy()
    else:
        # For '010010', return ['0', '01', '010', '0100', '010010']
        lineage_ids = list(itertools.accumulate(highlight_lineage))
        lineage_mask = np.isin(data.loc[:, "Agent ID"], lineage_ids)
        highlight_data = data.loc[lineage_mask, :]
        background_data = data.loc[~lineage_mask, :]
    # Plot up to SPLIT_TIME with first condition and between SPLIT_TIME
    # and MAX_TIME with second condition
    if filter_time:
        background_data = restrict_data(background_data)
        highlight_data = restrict_data(highlight_data)

    # Convert time to hours
    data.loc[:, "Time"] /= 3600
    highlight_data.loc[:, "Time"] /= 3600
    background_data.loc[:, "Time"] /= 3600
    # Collect data for timesteps before agent death
    if mark_death:
        death_data = []
        grouped_data = data.groupby(["Condition", "Agent ID"])
        data = data.set_index("Condition")
        for group in grouped_data:
            condition = group[0][0]
            agent_id = group[0][1]
            agent_data = group[1].reset_index()
            condition_data = data.loc[condition, :]
            # Cell did not die if at least one daughter exists,
            # or if the cell was present at the final timepoint
            final_agents = condition_data["Agent ID"][
                condition_data.Time == condition_data.Time.max()
            ].values
            if (
                (agent_id + "0" in condition_data.loc[:, "Agent ID"].values)
                or (agent_id + "1" in condition_data.loc[:, "Agent ID"].values)
            ) or agent_id in final_agents:
                continue
            max_group_time = agent_data.loc[:, "Time"].max()
            death_data.append(
                agent_data.loc[
                    agent_data.loc[:, "Time"] == max_group_time, :
                ].reset_index()
            )
        death_data = pd.concat(death_data)
        data = data.reset_index()

    # Iterate over agents
    background_data = background_data.groupby("Agent ID")
    highlight_data = highlight_data.groupby("Agent ID")
    for ax_idx, (column, color) in enumerate(columns_to_plot.items()):
        curr_ax = axes[ax_idx]
        if background_lineages:
            for _, background_agent in background_data:
                curr_ax.plot(
                    background_agent.loc[:, "Time"],
                    background_agent.loc[:, column],
                    c=background_color,
                    linewidth=background_linewidth,
                    alpha=background_alpha,
                )
        for _, highlight_agent in highlight_data:
            curr_ax.plot(
                highlight_agent.loc[:, "Time"],
                highlight_agent.loc[:, column],
                c=color,
                linewidth=1,
            )
        curr_ax.set_ylabel(column)
        if mark_death:
            # Mark values where cell died with a red "X"
            curr_ax.scatter(
                x=death_data.loc[:, "Time"],
                y=death_data.loc[:, column],
                c="maroon",
                alpha=0.5,
                marker="x",
            )
        curr_ax.autoscale(enable=True, axis="both", tight=True)
        curr_ax.set_yticks(ticks=np.around(curr_ax.get_ylim(), decimals=0))
        xticks = np.around(curr_ax.get_xlim(), decimals=1)
        xticklabels = []
        for time in xticks:
            if time % 1 == 0:
                xticklabels.append(str(int(time)))
            else:
                xticklabels.append(str(time))
        curr_ax.set_xticks(ticks=xticks, labels=xticklabels)
        curr_ax.set_xlabel("Time (hr)")
        sns.despine(ax=curr_ax, offset=3, trim=True)


def plot_field_snapshots(
    data: pd.DataFrame,
    metadata: dict[str, dict[str, dict[str, Any]]],
    highlight_lineage: Optional[str] = None,
    highlight_color: tuple = (1, 0, 0),
    min_pct=1,
    max_pct=1,
    colorbar_decimals=1,
    return_fig=False,
    n_snapshots=5,
    figsize=(9, 1.75),
) -> None:
    """Plot a row of snapshot images that span a replicate for each condition.
    In each of these images, the cell corresponding to a highlighted lineage
    is colored while the others are white.

    Args:
        data: DataFrame where each row is an agent and each column is a variable
            of interest. Must have these columns: 'Time', 'Death', 'Agent ID',
            'Boundary', 'Condition', and 'Seed'. 1 condition/seed at a time.
        metadata: Nested dictionary where each condition is an outer key and each
            initial seed is an inner key. Each seed point to a dictionary with the
            following keys: 'bounds' is of the form [x, y] and gives the dimensions
            of the spatial environment and 'fields' is a dictionary timeseries of
            the 'fields' Store for that condition and initial seed
        highlight_lineage: Agent ID to plot lineage trace for
        highlight_color: Color to plot highlight lineage with (default red)
        min_pct: Percent of minimum field concentration to use as minimum value
            in colorbar (1 = 100%)
        max_pct: Percent of maximum field concentration to use as maximum value
            in colorbar (1 = 100%)
        colorbar_decimals: Number of decimals to include in colorbar labels.
        return_fig: Whether to return the Figure
        n_snapshots: Number of equally-spaced (temporally) snapshots
        figsize: Desired size of entire figure
    """
    # Last snapshot at last tenth of an hour
    max_time_hrs = np.around(data.loc[:, "Time"].max() / 3600, decimals=1)
    snapshot_times_hrs = np.around(
        np.linspace(0, max_time_hrs, n_snapshots), decimals=1
    )
    snapshot_times = snapshot_times_hrs * 3600
    data = pd.concat([data.loc[data["Time"] == time, :] for time in snapshot_times])
    agent_ids = data.loc[:, "Agent ID"].unique()
    # For '010010', return ['0', '01', '010', '0100', '010010']
    lineage_ids = (
        list(itertools.accumulate(highlight_lineage)) if highlight_lineage else []
    )
    # Color all agents white except for highlighted lineage
    agent_colors = {
        agent_id: (0, 0, 1) for agent_id in agent_ids if agent_id not in lineage_ids
    }
    for agent_id in lineage_ids:
        agent_colors[agent_id] = tuple(list(rgb_to_hsv(highlight_color)))
    condition = data.loc[:, "Condition"].unique()[0]
    seed = data.loc[:, "Seed"].unique()[0]
    data = data.sort_values("Time")
    # Get field data at five equidistant time points
    condition_fields = metadata[condition][str(seed)]["fields"]
    condition_fields = {time: condition_fields[time] for time in data["Time"]}
    condition_bounds = metadata[condition][str(seed)]["bounds"]
    # Convert data back to dictionary form for snapshot plot
    snapshot_data: dict[float, dict] = {}
    for time, agent_id, boundary in zip(
        data["Time"], data["Agent ID"], data["Boundary"]
    ):
        data_at_time = snapshot_data.setdefault(time, {})
        agent_at_time = data_at_time.setdefault(agent_id, {})
        agent_at_time["boundary"] = boundary
    snapshots_fig = plot_snapshots(
        agents=snapshot_data,
        agent_colors=agent_colors,
        fields=condition_fields,
        bounds=condition_bounds,
        include_fields=["GLC[p]"],
        scale_bar_length=10,
        membrane_color=(0, 0, 0),
        membrane_width=0.01,
        colorbar_decimals=colorbar_decimals,
        default_font_size=10,
        field_label_size=0,
        min_pct=min_pct,
        max_pct=max_pct,
        n_snapshots=n_snapshots,
        figsize=figsize,
    )
    # New scale bar with reduced space between bar and label
    snapshots_fig.axes[1].artists[0].remove()
    scale_bar = anchored_artists.AnchoredSizeBar(
        snapshots_fig.axes[1].transData,
        10,
        "10 μm",
        "lower left",
        frameon=False,
        size_vertical=0.5,
        fontproperties={"size": 9},
    )
    snapshots_fig.axes[1].add_artist(scale_bar)
    # Move time axis up to save space
    bounds = list(snapshots_fig.axes[0].get_position().bounds)
    bounds[1] += 0.05
    snapshots_fig.axes[0].set_position(bounds)
    # Resize colorbar to save space and for looks
    upper_lim = snapshots_fig.axes[-3].get_position().y1
    bounds = list(snapshots_fig.axes[-2].get_position().bounds)
    bounds[1] += 0.05
    bounds[-1] = upper_lim - bounds[1]
    bounds[-2] += 0.1
    snapshots_fig.axes[-2].set_position(bounds)
    snapshots_fig.axes[1].set_ylabel(None)
    snapshots_fig.axes[-2].set_title(None)
    snapshots_fig.axes[-1].set_yticks([0.7, 0.8, 0.9, 1], size=9)
    snapshots_fig.axes[-1].yaxis.set_major_formatter(lambda x, pos: str(np.round(x, 1)))
    snapshots_fig.axes[-1].set_title("Glucose (mM)", y=1.05, fontsize=10, loc="left")
    snapshots_fig.axes[0].set_xticklabels(snapshot_times_hrs)
    snapshots_fig.axes[0].set_xlabel("Time (hr)", labelpad=4, size=10)
    snapshots_fig.axes[0].xaxis.set_tick_params(width=1, length=4)
    snapshots_fig.axes[0].spines["bottom"].set_linewidth(1)
    for ax in snapshots_fig.axes[1:-2]:
        ax.set_title(ax.get_title(), y=1.05, size=10)
    if return_fig:
        return snapshots_fig
    os.makedirs("out/analysis/paper_figures/", exist_ok=True)
    snapshots_fig.savefig(
        "out/analysis/paper_figures/"
        + f"{condition.replace('/', '_')}_seed_{seed}_fields.svg",
        bbox_inches="tight",
    )
    plt.close(snapshots_fig)


def plot_tag_snapshots(
    data: pd.DataFrame,
    metadata: dict[str, dict[int, dict[str, Any]]],
    tag_colors: dict[str, Any],
    snapshot_times: npt.NDArray[np.float64],
    conc: bool = False,
    min_color: Any = (1, 1, 1),
    out_prefix: Optional[str] = None,
    show_membrane: bool = False,
    return_fig: bool = False,
    figsize=(9, 1.75),
    highlight_agent=None,
) -> None:
    """Plot a row of snapshot images that span a replicate for each condition.
    In each of these images, cells will be will be colored with highlight_color
    and intensity corresponding to their value of highlight_column.

    Args:
        data: DataFrame where each row is an agent and each column is a variable
            of interest. Must have these columns: 'Time', 'Death', 'Agent ID',
            'Boundary', 'Condition', and 'Seed'. 1 condition/seed at a time.
        metadata: Nested dictionary where each condition is an outer key and each
            initial seed is an inner key. Each seed point to a dictionary with the
            following keys: 'bounds' is of the form [x, y] and gives the dimensions
            of the spatial environment and 'fields' is a dictionary timeseries of
            the 'fields' Store for that condition and initial seed
        tag_colors: Mapping column names in ``data`` to either RGB tuples or
            dictionaries containing the ``cmp`` and ``norm`` keys for the
            :py:class:`matplotlib.colors.Colormap` and
            :py:class:`matplotlib.colors.Normalize` instances to use for that tag
            If dictionaries are used, the ``min_color`` key is overrriden
        conc: Whether to normalize by volume before plotting
        snapshot_times: Times (in seconds) to make snapshots for
        min_color: Color for cells with lowest highlight_column value (default white)
        out_prefix: Prefix for output filename
        show_membrane: Whether to draw outline for agents
        return_fig: Whether to return figure. Only use with one tag.
        figsize: Desired size of entire figure
        highlight_agent: Mapping of agent IDs to `membrane_color` and `membrane_width`.
            Useful for highlighting specific agents, with rest using defaults
    """
    for highlight_column, tag_color in tag_colors.items():
        if snapshot_times is None:
            # Last snapshot at last tenth of an hour
            max_time_hrs = np.around(data.loc[:, "Time"].max() / 3600, decimals=1)
            snapshot_times_hrs = np.around(np.linspace(0, max_time_hrs, 5), decimals=1)
            snapshot_times = snapshot_times_hrs * 3600
            # Use 10 seconds for first snapshot to include cell wall update
            snapshot_times[0] = 10
        else:
            snapshot_times_hrs = snapshot_times / 3600
        data = pd.concat([data.loc[data["Time"] == time, :] for time in snapshot_times])
        # Get first SPLIT_TIME seconds from condition #1 and rest from condition #2
        data = restrict_data(data)
        # Sort values by time for ease of plotting later
        data = data.sort_values("Time")
        condition = "_".join(data.loc[:, "Condition"].unique())
        seed = data.loc[:, "Seed"].unique()[0]
        if conc:
            # Convert to concentrations
            data = data.set_index(["Condition", "Time", "Agent ID"])
            data = data.divide(data["Volume"], axis=0).drop(["Volume"], axis=1)
            data = data * COUNTS_PER_FL_TO_NANOMOLAR
            data = data.reset_index()
        condition_bounds = metadata[min(metadata)][seed]["bounds"]
        # Convert data back to dictionary form for snapshot plot
        snapshot_data: dict[float, dict] = {}
        data_max = data[highlight_column].max()
        data_min = data[highlight_column].min()
        tag_ranges = {(highlight_column,): [data_min, data_max]}
        for time, agent_id, boundary, column in zip(
            data["Time"], data["Agent ID"], data["Boundary"], data[highlight_column]
        ):
            data_at_time = snapshot_data.setdefault(time, {})
            agents_at_time = data_at_time.setdefault("agents", {})
            agent_at_time = agents_at_time.setdefault(agent_id, {})
            agent_at_time["boundary"] = boundary
            agent_at_time[highlight_column] = column
        if show_membrane:
            membrane_width = 0.1
        else:
            membrane_width = 0
        snapshots_fig = plot_tags(
            data=snapshot_data,
            bounds=condition_bounds,
            scale_bar_length=5,
            membrane_width=membrane_width,
            membrane_color=(0, 0, 0),
            colorbar_decimals=1,
            background_color="white",
            min_color=min_color,
            tag_colors={(highlight_column,): tag_color},
            tagged_molecules=[(highlight_column,)],
            default_font_size=9,
            convert_to_concs=False,
            tag_path_name_map={(highlight_column,): highlight_column},
            xlim=[15, 35],
            ylim=[15, 35],
            n_snapshots=len(snapshot_times),
            tag_ranges=tag_ranges,
            figsize=figsize,
            highlight_agent=highlight_agent,
        )
        # New scale bar with reduced space between bar and label
        snapshots_fig.axes[1].artists[0].remove()
        scale_bar = anchored_artists.AnchoredSizeBar(
            snapshots_fig.axes[1].transData,
            5,
            "5 μm",
            "lower left",
            frameon=False,
            size_vertical=0.5,
            fontproperties={"size": 9},
            sep=3,
        )
        snapshots_fig.axes[1].add_artist(scale_bar)
        # Move time axis up to save space
        bounds = list(snapshots_fig.axes[0].get_position().bounds)
        bounds[1] += 0.05
        snapshots_fig.axes[0].set_position(bounds)
        # Resize colorbar to save space and for looks
        upper_lim = snapshots_fig.axes[-3].get_position().y1
        bounds = list(snapshots_fig.axes[-2].get_position().bounds)
        bounds[1] += 0.05
        bounds[-1] = upper_lim - bounds[1]
        bounds[-2] += 0.1
        snapshots_fig.axes[-2].set_position(bounds)
        snapshots_fig.axes[1].set_ylabel(None)
        snapshots_fig.axes[-2].set_title(None)
        snapshots_fig.axes[-1].set_title(None)
        snapshots_fig.axes[-1].set_title(
            highlight_column, y=1.05, fontsize=8, loc="left"
        )
        snapshots_fig.axes[0].set_xticklabels(snapshot_times_hrs)
        snapshots_fig.axes[0].set_xlabel("Time (hr)", labelpad=4, size=9)
        snapshots_fig.axes[0].xaxis.set_tick_params(width=1, length=4)
        snapshots_fig.axes[0].spines["bottom"].set_linewidth(1)
        for ax in snapshots_fig.axes[1:-2]:
            ax.set_title(ax.get_title(), y=1.05, size=9)
        if return_fig and len(tag_colors) == 1:
            return snapshots_fig
        out_name = f"{condition.replace('/', '_')}_seed_{seed}_tags.svg"
        out_name = highlight_column.replace("/", "_") + "_" + out_name
        if out_prefix:
            out_name = "_".join([out_prefix, out_name])
        out_dir = "out/analysis/paper_figures/"
        snapshots_fig.savefig(os.path.join(out_dir, out_name), bbox_inches="tight")
        plt.close(snapshots_fig)


def make_tag_video(
    data: pd.DataFrame,
    metadata: dict[str, dict[str, dict[str, Any]]],
    tag_colors: dict[str, Any],
    out_prefix: str,
    conc: bool = False,
    min_color: Any = (0, 0, 0),
    show_membrane: bool = False,
    highlight_agent=None,
) -> None:
    """Make a video of snapshot images that span a replicate for a condition.
    In each of these videos, cells will be will be colored with highlight_color
    and intensity corresponding to their value of highlight_column.

    Args:
        data: DataFrame where each row is an agent and each column is a variable
            of interest. Must have these columns: 'Time', 'Death', 'Agent ID',
            'Boundary', 'Condition', and 'Seed'. 1 condition/seed at a time.
        metadata: Nested dictionary where each condition is an outer key and each
            initial seed is an inner key. Each seed point to a dictionary with the
            following keys: 'bounds' is of the form [x, y] and gives the dimensions
            of the spatial environment and 'fields' is a dictionary timeseries of
            the 'fields' Store for that condition and initial seed
        tag_colors: Mapping column names in ``data`` to either RGB tuples or
            dictionaries containing the ``cmp`` and ``norm`` keys for the
            :py:class:`matplotlib.colors.Colormap` and
            :py:class:`matplotlib.colors.Normalize` instances to use for that tag
            If dictionaries are used, the ``min_color`` key is overrriden
        conc: Whether to normalize by volume before plotting
        min_color: Color for cells with lowest highlight_column value (default black)
        out_prefix: Prefix for output filename
        show_membrane: Whether to draw outline for agents
        figsize: Desired size of entire figure
        highlight_agent: Mapping of agent IDs to `membrane_color` and `membrane_width`.
            Useful for highlighting specific agents, with rest using defaults
    """
    for highlight_column, tag_color in tag_colors.items():
        # Get first SPLIT_TIME seconds from condition #1 and rest from condition #2
        data = restrict_data(data)
        # Sort values by time for ease of plotting later
        data = data.sort_values("Time")
        seed = data.loc[:, "Seed"].unique()[0]
        if conc:
            # Convert to concentrations
            data = data.set_index(["Condition", "Time", "Agent ID"])
            data = data.divide(data["Volume"], axis=0).drop(["Volume"], axis=1)
            data = data * COUNTS_PER_FL_TO_NANOMOLAR
            data = data.reset_index()
        condition_bounds = metadata[min(metadata)][str(seed)]["bounds"]
        # Convert data back to dictionary form for snapshot plot
        snapshot_data: dict[float, dict] = {}
        for time, agent_id, boundary, column in zip(
            data["Time"], data["Agent ID"], data["Boundary"], data[highlight_column]
        ):
            data_at_time = snapshot_data.setdefault(time, {})
            agents_at_time = data_at_time.setdefault("agents", {})
            agent_at_time = agents_at_time.setdefault(agent_id, {})
            agent_at_time["boundary"] = boundary
            agent_at_time[highlight_column] = column
        if show_membrane:
            membrane_width = 0.1
        else:
            membrane_width = 0
        make_video(
            data=snapshot_data,
            bounds=condition_bounds,
            plot_type="tags",
            filename=out_prefix + "_snapshot_vid",
            scale_bar_length=5,
            membrane_width=membrane_width,
            membrane_color=(1, 1, 1),
            colorbar_decimals=1,
            background_color="white",
            min_color=min_color,
            tag_colors={(highlight_column,): tag_color},
            tagged_molecules=[(highlight_column,)],
            default_font_size=18,
            tag_label_size=18,
            convert_to_concs=False,
            tag_path_name_map={(highlight_column,): highlight_column},
            xlim=[15, 35],
            ylim=[15, 35],
            highlight_agent=highlight_agent,
            figsize=(12, 6),
        )
