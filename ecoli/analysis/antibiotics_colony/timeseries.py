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

def plot_generational_timeseries(
    data: pd.DataFrame,
    axes: List[plt.Axes] = None,
    columns_to_plot: Dict[str, tuple] = None,
    highlight_lineage: str = None,
    highlight_color: tuple = (1, 0, 0),
    colony_scale: bool = True,
    align:bool = True,
) -> None:
    '''For each generation of cells (identified by length of their agent ID),
    all generation start times are aligned and cells with long generation times
    are truncated at the median generation time. This aligned data is averaged
    across each generation to produce a 95% percentile interval on top of which
    the trace for a single highlighted lineage is plotted.

    Args:
        data: DataFrame where each row is an agent and each column is a variable
            of interest. Must have these columns: 'Time', 'Death', 'Agent ID',
            'Boundary', 'Condition', and 'Seed'. The first experimental condition
            in the 'Condition' column is treated as a control and plotted in gray.
            Include at most 2 conditions and 1 seed per condition.
        axes: Columns are plotted sequentially on these Axes.
        columns_to_plot: Dictionary of columns in data to plot sequentially on
            axes. Each column name corresponds to a RGB tuple to color that plot.
        highlight_lineage: Agent ID to plot lineage trace for
        highlight_color: Color to plot highlighted lineage with (default red)
        colony_scale: Whether to plot 95% percentile interval
        align: Whether to attempt alignment by generation
    '''
    columns_to_include = list(columns_to_plot.keys() |
        {'Agent ID', 'Condition', 'Time', 'Death'})
    data = data.loc[:, columns_to_include]
    # For '010010', return ['0', '01', '010', '0100', '010010']
    lineage_ids = list(itertools.accumulate(highlight_lineage))
    highlight_data = data.loc[
        np.isin(data.loc[:, 'Agent ID'], lineage_ids), :]
    agent_ids = data.loc[:, 'Agent ID'].unique()
    background_agents = agent_ids[~np.isin(agent_ids, lineage_ids)]
    agent_id_lengths = np.array([len(agent_id) for agent_id in data['Agent ID']])
    
    if align:
        n_generations = max(agent_id_lengths)
        previous_end_time = 0
        aligned_data = []
        # For each generation, assume start time = previous generation's end time
        # (0 for first gen.). Get median generation length and filter out all data
        # that exceeds that. Start time + median gen. length = new gen. start time
        for generation in range(1, n_generations +  1):
            generation_data = data.loc[agent_id_lengths == generation, :]
            grouped_data = generation_data.groupby(['Condition', 'Agent ID'])
            real_start_time = grouped_data.aggregate({'Time': 'min'})
            target_start_time = previous_end_time
            real_gen_length = grouped_data.aggregate({'Time': 'max'}) - real_start_time
            target_end_time = int(target_start_time + real_gen_length.median())
            for agent_id, agent_data in grouped_data:
                agent_start_time =  int(real_start_time.loc[agent_id])
                agent_data.loc[:, 'Time'] -= agent_start_time - target_start_time
                agent_data = agent_data.loc[agent_data['Time'] <= target_end_time, :]
                aligned_data.append(agent_data)
            previous_end_time = target_end_time
        aligned_data = pd.concat(aligned_data)
    else:
        aligned_data = data

    ax_idx = 0
    # Only plot data between SPLIT_TIME and MAX_TIME if multiple
    # conditions in supplied data
    conditions = aligned_data.loc[:, 'Condition'].unique()
    if len(conditions) > 1:
        aligned_data = aligned_data.set_index(['Condition'])
        condition_1_mask = ((aligned_data.loc[conditions[0]]['Time'] 
            >= SPLIT_TIME) & (aligned_data.loc[conditions[0]]['Time'] 
            <= MAX_TIME))
        condition_1_data = aligned_data.loc[conditions[0]].loc[
            condition_1_mask, :]
        condition_2_mask = ((aligned_data.loc[conditions[1]]['Time'] 
            >= SPLIT_TIME) & (aligned_data.loc[conditions[1]]['Time'] 
            <= MAX_TIME))
        condition_2_data = aligned_data.loc[conditions[1]].loc[
            condition_2_mask, :]
        aligned_data = pd.concat([condition_1_data, condition_2_data])
        aligned_data = aligned_data.reset_index()
        highlight_data = highlight_data.set_index(['Condition'])
        highlight_mask = ((highlight_data.loc[conditions[1]]['Time'] 
            >= SPLIT_TIME) & (highlight_data.loc[conditions[1]]['Time'] 
            <= MAX_TIME))
        highlight_data = highlight_data.loc[highlight_mask, :]
        highlight_data = highlight_data.reset_index()
    
    # Set agent ID as index for easy indexing when plotting
    aligned_data = aligned_data.set_index('Agent ID')
    highlight_data = highlight_data.set_index('Agent ID')
    # Convert time to minutes
    highlight_data.loc[:, 'Time'] /= 60
    aligned_data.loc[:, 'Time'] /= 60
    dead_data = aligned_data.loc[aligned_data['Death'], :]
    # Plot first condition in gray
    for column, color in columns_to_plot.items():
        curr_ax = axes[ax_idx]
        ax_idx += 1
        if colony_scale:
            if len(conditions) > 1:
                palette = {conditions[0]: (0, 0, 0), conditions[1]: color}
            else:
                palette = {conditions[0]: color}
            # # Plot colony-scale error bars
            # g = sns.lineplot(
            #     data=aligned_data, x='Time', y=column, hue='Condition',
            #     ax=curr_ax, errorbar=('pi', 95), linewidth=0, legend=False,
            #     palette=palette, err_kws={'alpha': 0.2})
            for background_agent in background_agents:
                sns.lineplot(
                    data=aligned_data.loc[background_agent, :],
                    x='Time', y=column, hue='Condition',
                    ax=curr_ax, legend=False, palette=palette,
                    linewidth=0.5)
            # Mark values where cell died
            g = sns.scatterplot(
                data=dead_data, x='Time', y=column, hue='Condition', ax=g,
                markers=['X'], style='Death', legend=False)
        # If no highlight color is specified, use the color for that column
        if len(conditions) > 1:
            palette = {conditions[0]: (0, 0, 0)}
            if highlight_color:
                palette[conditions[1]] = highlight_color
            else:
                palette[conditions[1]] = color
        else:
            palette = {}
            if highlight_color:
                palette[conditions[0]] = highlight_color
            else:
                palette[conditions[0]] = color
        # Plot each agent in highlighted lineage individually so
        # subsequent generations are discontinuous
        for highlighted_agent in lineage_ids:
            sns.lineplot(
                data=highlight_data.loc[highlighted_agent, :],
                x='Time', y=column, hue='Condition', linewidth=1,
                ax=curr_ax, legend=False, palette=palette)
        curr_ax.autoscale(enable=True, axis='both', tight='both')
        curr_ax.set_yticks(ticks=np.ceil(curr_ax.get_ylim()))
        curr_ax.set_xticks(ticks=np.ceil(curr_ax.get_xlim()))
        curr_ax.set_xlabel('Time (min)')
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


def plot_concentration_timeseries(
    data: pd.DataFrame,
    axes: List[plt.Axes],
    columns_to_plot: Dict[str, tuple] = None,
    highlight_lineage: str = None,
    highlight_color: tuple = (1, 0, 0),
    colony_scale: bool = True,
) -> None:
    '''Normalize variables by volume to create bulk timeseries plot of
    concentrations.

    Args:
        data: DataFrame where each row is an agent and each column is a variable
            of interest. Must have these columns: 'Time', 'Death', 'Agent ID',
            'Boundary', 'Condition', and 'Seed'. The first experimental condition
            in the 'Condition' column is treated as a control and plotted in gray.
            Include at most 2 conditions and 1 seed per condition.
        columns_to_plot: Dictionary of columns in data to plot sequentially on
            axes. Each column name corresponds to a RGB tuple to color that plot.
        highlight_lineage: Agent ID to plot lineage trace for
        highlight_color: Color to plot highlighted lineage with (default red)
        colony_scale: Whether to plot 95% percentile interval
    '''
    columns_to_include = list(set(columns_to_plot) | {
        'Volume', 'Condition', 'Time', 'Death', 'Agent ID'})
    data = data.loc[:, columns_to_include]

    # Convert to concentrations
    data = data.set_index(['Condition', 'Time', 'Death', 'Agent ID'])
    data = data.divide(data['Volume'], axis=0).drop(['Volume'], axis=1)
    data = data * COUNTS_PER_FL_TO_NANOMOLAR
    data = data.reset_index()
    data = data.set_index(['Condition'])
    
    ax_idx = 0
    # Only plot data between SPLIT_TIME and MAX_TIME if multiple
    # conditions in supplied data
    conditions = data.index.unique()
    if len(conditions) > 1:
        condition_1_mask = ((data.loc[conditions[0]]['Time'] 
            >= SPLIT_TIME) & (data.loc[conditions[0]]['Time'] 
            <= MAX_TIME))
        condition_1_data = data.loc[conditions[0]].loc[
            condition_1_mask, :]
        condition_2_mask = ((data.loc[conditions[1]]['Time'] 
            >= SPLIT_TIME) & (data.loc[conditions[1]]['Time'] 
            <= MAX_TIME))
        condition_2_data = data.loc[conditions[1]].loc[
            condition_2_mask, :]
        data = pd.concat([condition_1_data, condition_2_data])
    data = data.reset_index()

    # Convert time to minutes
    data.loc[:, 'Time'] /= 60
    # For '010010', return ['0', '01', '010', '0100', '010010']
    lineage_ids = list(itertools.accumulate(highlight_lineage))
    agent_ids = data.loc[:, 'Agent ID'].unique()
    background_ids = agent_ids[~np.isin(agent_ids, lineage_ids)]
    data = data.set_index('Agent ID')
    highlight_data = data.loc[lineage_ids, :]
    background_data = data.loc[background_ids, :]
    data = data.reset_index()

    if data.loc[:, 'Death'].sum() == 0:
        death_data = []
        grouped_data = data.groupby(['Condition', 'Agent ID'])
        data = data.set_index('Condition')
        for group in grouped_data:
            condition = group[0][0]
            agent_id = group[0][1]
            agent_data = group[1].reset_index()
            condition_data = data.loc[condition, :]
            if ((agent_id + '0' in condition_data.index) or
                (agent_id + '1' in condition_data.index)
            ):
                continue
            max_group_time = agent_data.loc[:, 'Time'].max()
            death_data.append(agent_data.loc[agent_data.loc[:, 'Time']
                ==max_group_time, :].reset_index())
        death_data = pd.concat(death_data)
        data = data.reset_index()
    else:
        death_data = data.loc[data.loc[:, 'Death'], :]
    for column, color in columns_to_plot.items():
        curr_ax = axes[ax_idx]
        ax_idx += 1
        if colony_scale:
            if len(conditions) > 1:
                palette = {conditions[0]: (0, 0, 0), conditions[1]: color}
            else:
                palette = {conditions[0]: color}
            # # Plot colony-scale error bars
            # g = sns.lineplot(
            #     data=data, x='Time', y=column, hue='Condition',
            #     ax=curr_ax, errorbar=('pi', 95), linewidth=0, legend=False,
            #     palette=palette, err_kws={'alpha': 0.2})
            # Plot each agent in highlighted lineage individually so
            # subsequent generations are discontinuous
            for background_id in background_ids:
                sns.lineplot(
                    data=background_data.loc[background_id, :],
                    x='Time', y=column, hue='Condition',
                    ax=curr_ax, legend=False, palette=palette,
                    linewidth=0.5)
            # Mark values where cell died
            g = sns.scatterplot(
                data=death_data, x='Time', y=column, hue='Condition', ax=g,
                markers=['X'], style='Death', legend=False, palette=palette)
        # If no highlight color is specified, use the color for that column
        if len(conditions) > 1:
            palette = {conditions[0]: (0, 0, 0)}
            if highlight_color:
                palette[conditions[1]] = highlight_color
            else:
                palette[conditions[1]] = color
        else:
            palette = {}
            if highlight_color:
                palette[conditions[0]] = highlight_color
            else:
                palette[conditions[0]] = color
        # Plot each agent in highlighted lineage individually so
        # subsequent generations are discontinuous
        for highlighted_agent in lineage_ids:
            sns.lineplot(
                data=highlight_data.loc[highlight_data.loc[
                    :, 'Agent ID']==highlighted_agent, :],
                x='Time', y=column, hue='Condition',
                ax=curr_ax, legend=False, palette=palette,
                linewidth=1)
        curr_ax.autoscale(enable=True, axis='both', tight='both')
        curr_ax.set_yticks(ticks=np.ceil(curr_ax.get_ylim()))
        curr_ax.set_xticks(ticks=np.ceil(curr_ax.get_xlim()))
        curr_ax.set_xlabel('Time (min)')
        sns.despine(ax=curr_ax, offset=3, trim=True)
