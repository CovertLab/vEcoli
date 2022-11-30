from typing import List, Dict, Any
import itertools

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv

from ecoli.plots.snapshots import plot_snapshots, plot_tags

from ecoli.analysis.antibiotics_colony import (
    COUNTS_PER_FL_TO_NANOMOLAR, restrict_data)


def plot_timeseries(
    data: pd.DataFrame,
    axes: List[plt.Axes] = None,
    columns_to_plot: Dict[str, tuple] = None,
    highlight_lineage: str = None,
    conc: bool = False,
    mark_death: bool = False,
    background_lineages: bool = True,
    filter_time: bool = True,
    background_color: tuple = (0.5, 0.5, 0.5),
    background_alpha: float = 0.5,
    background_linewidth: float = 0.1
) -> None:
    '''Plot selected traces with specific lineage highlighted and others in gray.

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
    '''
    columns_to_include = list(columns_to_plot.keys() |
        {'Agent ID', 'Condition', 'Time'})
    if conc:
        columns_to_include.append('Volume')
    data = data.loc[:, columns_to_include]
    if conc:
        # Convert to concentrations
        data = data.set_index(['Condition', 'Time', 'Agent ID'])
        data = data.divide(data['Volume'], axis=0).drop(['Volume'], axis=1)
        data = data * COUNTS_PER_FL_TO_NANOMOLAR
        data = data.reset_index()
    # Sort values by time for ease of plotting later
    data = data.sort_values('Time')
    if highlight_lineage == 'mean':
        highlight_data = data.groupby(['Condition', 'Time']
            ).mean().reset_index()
        highlight_data['Agent ID'] = highlight_lineage
        background_data = data.copy()
    elif highlight_lineage == 'median':
        highlight_data = data.groupby(['Condition', 'Time']
            ).median().reset_index()
        highlight_data['Agent ID'] = highlight_lineage
        background_data = data.copy()
    else:
        # For '010010', return ['0', '01', '010', '0100', '010010']
        lineage_ids = list(itertools.accumulate(highlight_lineage))
        lineage_mask = np.isin(data.loc[:, 'Agent ID'], lineage_ids)
        highlight_data = data.loc[lineage_mask, :]
        background_data = data.loc[~lineage_mask, :]
    # Plot up to SPLIT_TIME with first condition and between SPLIT_TIME
    # and MAX_TIME with second condition
    if filter_time:
        background_data = restrict_data(background_data)
        highlight_data = restrict_data(highlight_data)
    
    # Convert time to hours
    data.loc[:, 'Time'] /= 3600
    highlight_data.loc[:, 'Time'] /= 3600
    background_data.loc[:, 'Time'] /= 3600
    # Collect data for timesteps before agent death
    if mark_death:
        death_data = []
        grouped_data = data.groupby(['Condition', 'Agent ID'])
        data = data.set_index('Condition')
        for group in grouped_data:
            condition = group[0][0]
            agent_id = group[0][1]
            agent_data = group[1].reset_index()
            condition_data = data.loc[condition, :]
            # Cell did not die if at least one daughter exists
            import ipdb; ipdb.set_trace()
            if ((agent_id + '0' in condition_data.loc[:, 'Agent ID'].values) or
                (agent_id + '1' in condition_data.loc[:, 'Agent ID'].values)
            ):
                continue
            max_group_time = agent_data.loc[:, 'Time'].max()
            death_data.append(agent_data.loc[agent_data.loc[:, 'Time']
                ==max_group_time, :].reset_index())
        death_data = pd.concat(death_data)
        data = data.reset_index()
    
    # Iterate over agents
    background_data = background_data.groupby('Agent ID')
    highlight_data = highlight_data.groupby('Agent ID')
    for ax_idx, (column, color) in enumerate(columns_to_plot.items()):
        curr_ax = axes[ax_idx]
        if background_lineages:
            for _, background_agent in background_data:
                curr_ax.plot(background_agent.loc[:, 'Time'],
                    background_agent.loc[:, column], c=background_color,
                    linewidth=background_linewidth, alpha=background_alpha)
        for _, highlight_agent in highlight_data:
            curr_ax.plot(highlight_agent.loc[:, 'Time'],
                highlight_agent.loc[:, column], c=color,
                linewidth=1)
        curr_ax.set_ylabel(column)
        if mark_death:
            # Mark values where cell died with a red "X"
            curr_ax.scatter(
                x=death_data.loc[:, 'Time'], y=death_data.loc[:, column],
                c='maroon', alpha=0.5, marker='x')
        curr_ax.autoscale(enable=True, axis='both', tight='both')
        curr_ax.set_yticks(ticks=np.around(
            curr_ax.get_ylim(), decimals=0))
        xticks = np.around(curr_ax.get_xlim(), decimals=1)
        xticklabels = []
        for time in xticks:
            if time % 1 == 0:
                xticklabels.append(int(time))
            else:
                xticklabels.append(time)
        curr_ax.set_xticks(ticks=xticks, labels=xticklabels)
        curr_ax.set_xlabel('Time (hr)')
        sns.despine(ax=curr_ax, offset=3, trim=True)


def plot_field_snapshots(
    data: pd.DataFrame,
    metadata: Dict[str, Dict[int, Dict[str, Any]]],
    highlight_lineage: str = None,
    highlight_color: tuple = (1, 0, 0),
    min_pct=1,
    max_pct=1,
    colorbar_decimals=1,
) -> None:
    '''Plot a row of snapshot images that span a replicate for each condition.
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
    '''
    # Last snapshot at last tenth of an hour
    max_time_hrs = np.around(data.loc[:, 'Time'].max()/3600, decimals=1)
    snapshot_times_hrs = np.around(np.linspace(0, max_time_hrs, 5),
        decimals=1)
    snapshot_times = snapshot_times_hrs * 3600
    data = pd.concat([
        data.loc[data['Time']==time, :] for time in snapshot_times
    ])
    agent_ids = data.loc[:, 'Agent ID'].unique()
    # For '010010', return ['0', '01', '010', '0100', '010010']
    lineage_ids = list(itertools.accumulate(highlight_lineage))
    # Color all agents white except for highlighted lineage
    agent_colors = {
        agent_id: (0, 0, 1) for agent_id in agent_ids
        if agent_id not in lineage_ids
    }
    for agent_id in lineage_ids:
        agent_colors[agent_id] = rgb_to_hsv(highlight_color)
    condition = data.loc[:, 'Condition'].unique()[0]
    seed = data.loc[:, 'Seed'].unique()[0]
    data = data.sort_values('Time')
    # Get field data at five equidistant time points
    condition_fields = metadata[condition][seed]['fields']
    condition_fields = {
        time: condition_fields[time]
        for time in data['Time']
    }
    condition_bounds = metadata[condition][seed]['bounds']
    # Convert data back to dictionary form for snapshot plot
    snapshot_data = {}         
    for time, agent_id, boundary in zip(
        data['Time'], data['Agent ID'], data['Boundary']
    ):
        data_at_time = snapshot_data.setdefault(time, {})
        agent_at_time = data_at_time.setdefault(agent_id, {})
        agent_at_time['boundary'] = boundary
    snapshots_fig = plot_snapshots(
        agents=snapshot_data,
        agent_colors=agent_colors,
        fields=condition_fields,
        bounds=condition_bounds,
        include_fields=['GLC[p]'],
        scale_bar_length=10,
        membrane_color=(0, 0, 0),
        membrane_width=0.01,
        colorbar_decimals=colorbar_decimals,
        default_font_size=48,
        min_pct=min_pct,
        max_pct=max_pct,
    )
    snapshots_fig.subplots_adjust(wspace=0.7, hspace=0.1)
    snapshots_fig.axes[1].set_ylabel(None)
    snapshots_fig.axes[-2].set_title(None)
    snapshots_fig.axes[-1].set_title(
        'External Glucose\n(mM)', y=1.1, fontsize=48)
    snapshots_fig.axes[0].set_xticklabels(snapshot_times_hrs)
    snapshots_fig.axes[0].set_xlabel('Time (hr)')
    snapshots_fig.savefig('out/analysis/paper_figures/' + 
        f'{condition.replace("/", "_")}_seed_{seed}_fields.svg',
        bbox_inches='tight')
    plt.close(snapshots_fig)


def plot_tag_snapshots(
    data: pd.DataFrame,
    metadata: Dict[str, Dict[int, Dict[str, Any]]],
    tag_colors: Dict[str, Any] = None,
    conc: bool = False,
    snapshot_times: List = None,
    min_color: Any = None,
    out_prefix: str = None,
    show_membrane: bool = False,
) -> None:
    '''Plot a row of snapshot images that span a replicate for each condition.
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
    '''
    for highlight_column, tag_color in tag_colors.items():
        if snapshot_times is None:
            # Last snapshot at last tenth of an hour
            max_time_hrs = np.around(data.loc[:, 'Time'].max()/3600, decimals=1)
            snapshot_times_hrs = np.around(np.linspace(0, max_time_hrs, 5),
                decimals=1)
            snapshot_times = snapshot_times_hrs * 3600
            # Use 10 seconds for first snapshot to include cell wall update
            snapshot_times[0] = 10
        else:
            snapshot_times_hrs = snapshot_times / 3600
        data = pd.concat([
            data.loc[data['Time']==time, :] for time in snapshot_times
        ])
        # Get first SPLIT_TIME seconds from condition #1 and rest from condition #2
        data = restrict_data(data)
        # Sort values by time for ease of plotting later
        data = data.sort_values('Time')
        condition = '_'.join(data.loc[:, 'Condition'].unique())
        seed = data.loc[:, 'Seed'].unique()[0]
        if conc:
            # Convert to concentrations
            data = data.set_index(['Condition', 'Time', 'Agent ID'])
            data = data.divide(data['Volume'], axis=0).drop(['Volume'], axis=1)
            data = data * COUNTS_PER_FL_TO_NANOMOLAR
            data = data.reset_index()
        condition_bounds = metadata[min(metadata)][seed]['bounds']
        # Convert data back to dictionary form for snapshot plot
        snapshot_data = {}
        magnitude = data[highlight_column].abs().max()
        tag_ranges = {(highlight_column,): [-magnitude, magnitude]}
        for time, agent_id, boundary, column in zip(
            data['Time'], data['Agent ID'], data['Boundary'],
            data[highlight_column]
        ):
            data_at_time = snapshot_data.setdefault(time, {})
            agents_at_time = data_at_time.setdefault('agents', {})
            agent_at_time = agents_at_time.setdefault(agent_id, {})
            agent_at_time['boundary'] = boundary
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
            background_color='white',
            min_color=min_color,
            tag_colors={(highlight_column,): tag_color},
            tagged_molecules=[(highlight_column,)],
            default_font_size=48,
            convert_to_concs=False,
            tag_path_name_map={(highlight_column,): highlight_column},
            xlim=[15, 35],
            ylim=[15, 35],
            n_snapshots=len(snapshot_times),
            tag_ranges=tag_ranges
        )
        snapshots_fig.subplots_adjust(wspace=0.7, hspace=0.1)
        snapshots_fig.axes[1].set_ylabel(None)
        snapshots_fig.axes[-1].set_title(None)
        snapshots_fig.axes[-1].set_title(highlight_column, fontsize=48, y=1.05)
        snapshots_fig.axes[0].set_xticklabels(snapshot_times_hrs, fontsize=48)
        snapshots_fig.axes[0].set_xlabel('Time (hr)', fontsize=48)
        out_name = f'{condition.replace("/", "_")}_seed_{seed}_tags.svg'
        out_name = highlight_column.replace("/", "_") + '_' + out_name
        if out_prefix:
            out_name  = '_'.join([out_prefix, out_name])
        out_dir = 'out/analysis/paper_figures/'
        snapshots_fig.savefig(os.path.join(out_dir, out_name),
            bbox_inches='tight')
        plt.close(snapshots_fig)
