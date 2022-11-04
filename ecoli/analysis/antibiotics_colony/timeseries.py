from typing import List, Dict, Any
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv

from ecoli.plots.snapshots import plot_snapshots, plot_tags

from ecoli.analysis.antibiotics_colony import (
    MAX_TIME, SPLIT_TIME, COUNTS_PER_FL_TO_NANOMOLAR)


def restrict_data(data: pd.DataFrame):
    """If there is more than one condition in data, keep up
    to SPLIT_TIME from the first condition and between SPLIT_TIME
    and MAX_TIME from the second."""
    conditions = data.loc[:, 'Condition'].unique()
    if len(conditions) > 1:
        data = data.set_index(['Condition'])
        condition_1_mask = ((data.loc[conditions[0]]['Time'] 
            <= SPLIT_TIME))
        condition_1_data = data.loc[conditions[0]].loc[
            condition_1_mask, :]
        condition_2_mask = ((data.loc[conditions[1]]['Time'] 
            >= SPLIT_TIME) & (data.loc[conditions[1]]['Time'] 
            <= MAX_TIME))
        condition_2_data = data.loc[conditions[1]].loc[
            condition_2_mask, :]
        data = pd.concat([condition_1_data, condition_2_data])
        data = data.reset_index()
    else:
        data = data.loc[data.loc[:, 'Time'] <= MAX_TIME, :]
    return data


def plot_timeseries(
    data: pd.DataFrame,
    axes: List[plt.Axes] = None,
    columns_to_plot: Dict[str, tuple] = None,
    highlight_lineage: str = None,
    conc: bool = False,
    mark_death: bool = False,
    background_lineages: bool = True,
) -> None:
    '''Plot selected traces with specific lineage highlighted and others in gray.

    Args:
        data: DataFrame where each row is an agent and each column is a variable
            of interest. Must have these columns: 'Time', 'Death', 'Agent ID',
            'Boundary', 'Condition', and 'Seed'. The first experimental condition
            in the 'Condition' column is treated as a control and plotted in gray.
            Include at most 2 conditions and 1 seed per condition.
        axes: Columns are plotted sequentially on these Axes.
        columns_to_plot: Dictionary of columns in data to plot sequentially on
            axes. Each column name corresponds to a RGB tuple to color the trace
            of the highlighted lineage on that plot.
        highlight_lineage: Agent ID to plot lineage trace for.
        conc: Whether to normalize data by volume and convert to nM
        mark_death: Mark cells that die with red X on time step of death
        background_lineages: Whether to plot traces for other lineages (gray).
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
    # For '010010', return ['0', '01', '010', '0100', '010010']
    lineage_ids = list(itertools.accumulate(highlight_lineage))
    lineage_mask = np.isin(data.loc[:, 'Agent ID'], lineage_ids)
    highlight_data = data.loc[lineage_mask, :]
    background_data = data.loc[~lineage_mask, :]
    # Plot up to SPLIT_TIME with first condition and between SPLIT_TIME
    # and MAX_TIME with second condition
    background_data = restrict_data(background_data)
    highlight_data = restrict_data(highlight_data)
    
    # Convert time to hours
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
            if ((agent_id + '0' in condition_data.index) or
                (agent_id + '1' in condition_data.index)
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
                    background_agent.loc[:, column], c=(0.5, 0.5, 0.5),
                    linewidth=0.1)
        for _, highlight_agent in highlight_data:
            curr_ax.plot(highlight_agent.loc[:, 'Time'],
                highlight_agent.loc[:, column], c=color,
                linewidth=1)
        curr_ax.set_ylabel(column)
        if mark_death:
            # Mark values where cell died with a red "X"
            curr_ax.scatter(
                x=death_data.loc[:, 'Time'], y=death_data.loc[:, column],
                c='maroon', markers='X')
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
    highlight_color: tuple = (1, 0, 0)
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
        colorbar_decimals=1
    )
    snapshots_fig.subplots_adjust(wspace=0.7, hspace=0.1)
    snapshots_fig.axes[1].set_ylabel(None)
    snapshots_fig.axes[-2].set_title(None)
    snapshots_fig.axes[-1].set_title(
        'External Glucose\n(mM)', y=1.1, fontsize=36)
    snapshots_fig.axes[0].set_xticklabels(snapshot_times_hrs)
    snapshots_fig.axes[0].set_xlabel('Time (hr)')
    snapshots_fig.savefig('out/analysis/paper_figures/' + 
        f'{condition.replace("/", "_")}_seed_{seed}_fields.svg',
        bbox_inches='tight')
    plt.close(snapshots_fig)


def plot_tag_snapshots(
    data: pd.DataFrame,
    metadata: Dict[str, Dict[int, Dict[str, Any]]],
    highlight_column: str = None,
    highlight_color: tuple = (1, 0, 0),
    conc: bool = False,
    snapshot_times: List = None,
    min_color: Any = None,
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
        highlight_column: Variable to use to color cells
        highlight_color: Color for cells with highest highlight_column value
        conc: Whether to normalize by volume before plotting
        snapshot_times: Times (in seconds) to make snapshots for
        min_color: Color for cells with lowest highlight_column value (default white)
    '''
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
    for time, agent_id, boundary, column in zip(
        data['Time'], data['Agent ID'], data['Boundary'],
        data[highlight_column]
    ):
        data_at_time = snapshot_data.setdefault(time, {})
        agents_at_time = data_at_time.setdefault('agents', {})
        agent_at_time = agents_at_time.setdefault(agent_id, {})
        agent_at_time['boundary'] = boundary
        agent_at_time[highlight_column] = column
    snapshots_fig = plot_tags(
        data=snapshot_data,
        bounds=condition_bounds,
        scale_bar_length=5,
        membrane_width=0,
        colorbar_decimals=1,
        background_color='white',
        min_color=min_color,
        tag_colors={(highlight_column,): rgb_to_hsv(highlight_color)},
        tagged_molecules=[(highlight_column,)],
        default_font_size=36,
        convert_to_concs=False,
        tag_path_name_map={(highlight_column,): highlight_column},
        xlim=[15, 35],
        ylim=[15, 35],
    )
    snapshots_fig.subplots_adjust(wspace=0.7, hspace=0.1)
    snapshots_fig.axes[1].set_ylabel(None)
    snapshots_fig.axes[-1].set_title(None)
    snapshots_fig.axes[-1].set_title(highlight_column, fontsize=36, y=1.05)
    snapshots_fig.axes[0].set_xticklabels(snapshot_times_hrs, fontsize=36)
    snapshots_fig.axes[0].set_xlabel('Time (hr)', fontsize=36)
    snapshots_fig.savefig('out/analysis/paper_figures/' + 
        f'{condition.replace("/", "_")}_seed_{seed}_tags.svg',
        bbox_inches='tight')
    plt.close(snapshots_fig)
