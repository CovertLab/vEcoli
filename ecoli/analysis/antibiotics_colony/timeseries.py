from typing import List, Dict, Any
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv

from ecoli.plots.snapshots import plot_snapshots

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
            of the highlighted lineage on that plot (other traces are gray).
        highlight_lineage: Agent ID to plot lineage trace for.
        conc: Whether to normalize data by volume and convert to nM
        mark_death: Mark cells that die with red X on time step of death
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
    # Need to iterate over agent IDs when plotting
    background_data = background_data.groupby('Agent ID')
    highlight_data = highlight_data.groupby('Agent ID')
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
    
    for ax_idx, (column, color) in enumerate(columns_to_plot.items()):
        curr_ax = axes[ax_idx]
        # Plot agents individually so traces are discontinuous
        for _, background_agent in background_data:
            sns.lineplot(
                data=background_agent, x='Time', y=column, c=(0.5, 0.5, 0.5),
                ax=curr_ax, legend=False, linewidth=0.1)
        for _, highlight_agent in highlight_data:
            sns.lineplot(
                data=highlight_agent, x='Time', y=column, c=color,
                ax=curr_ax, legend=False, linewidth=1)
        if mark_death:
            # Mark values where cell died with a red "X"
            sns.scatterplot(
                data=death_data, x='Time', y=column, ax=curr_ax, c='maroon',
                markers=['X'], style='Death', legend=False)
        curr_ax.autoscale(enable=True, axis='both', tight='both')
        curr_ax.set_yticks(ticks=np.around(
            curr_ax.get_ylim(), decimals=0))
        curr_ax.set_xticks(ticks=np.around(
            curr_ax.get_xlim(), decimals=1))
        curr_ax.set_xlabel('Time (hr)')
        sns.despine(ax=curr_ax, offset=3, trim=True)


def plot_snapshot_timeseries(
    data: pd.DataFrame,
    metadata: Dict[str, Dict[int, Dict[str, Any]]],
    highlight_lineage: str = None,
    highlight_color: tuple = (1, 0, 0),
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
    # Get data at five equidistant time points
    time_chunks = np.array_split(data['Time'].unique(), 4)
    split_times = [times[0] for times in time_chunks]
    split_times.append(time_chunks[-1][-1])
    data = pd.concat([
        data.loc[data['Time']==time, :] for time in split_times
    ])
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
        colorbar_decimals=1
    )
    snapshots_fig.subplots_adjust(wspace=0.7, hspace=0.1)
    snapshots_fig.axes[1].set_ylabel(None)
    snapshots_fig.axes[-2].set_title(None)
    snapshots_fig.axes[-1].set_title(
        'External Glucose\n(mM)', y=1.1, fontsize=36)
    ylimits = np.round(snapshots_fig.axes[-1].get_ylim())
    snapshots_fig.axes[-1].set_yticks([])
    snapshots_fig.axes[-1].set_title(int(ylimits[1]), fontsize=36)
    snapshots_fig.axes[-1].text(0.5, -0.01, int(ylimits[0]),
        horizontalalignment='center', verticalalignment='top',
        transform=snapshots_fig.axes[-1].transAxes, fontsize=36)
    snapshots_fig.savefig('out/analysis/paper_figures/' + 
        f'{condition.replace("/", "_")}_seed_{seed}_snapshots.svg',
        bbox_inches='tight')
    plt.close(snapshots_fig)
