from typing import List, Dict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ecoli.plots.snapshots import plot_snapshots

from ecoli.analysis.antibiotics_colony import (
    CONDITION_GROUPINGS, METADATA, CONCENTRATION_COLUMNS, SPLIT_TIME)

def plot_generational_timeseries(
    data: Dict[str, Dict[int, pd.DataFrame]],
    out: bool = True,
    axes: List[plt.Axes] = None,
) -> None:
    '''For each generation of cells (identified by length of their agent ID),
    all generation start times are aligned and cells with long generation times
    are truncated at the median generation time. This aligned data is averaged
    across each generation to produce a bulk timeseries plot.

    Args:
        data: Nested dictionary with experimental condition on outer level and
            initial seed on inner level. Each seed has a DataFrame where each
            row is an agent and each column is a variable of interest (e.g.
            count of an mRNA). All DataFrames contain some metadata columns:
            'Time', 'Death' (True if agent about to die), 'Agent ID', and
            'Boundary' (only used for snapshot plots), 'Condition', and 'Seed'.
        out: Immediately save and close current figure if True.
        axes: If supplied, columns are plotted sequentially on these Axes.
    '''
    columns_to_plot = list(set(data['Glucose'][0]['df'].columns)
        - CONCENTRATION_COLUMNS)
    glc = [seed_data['df'].loc[:, columns_to_plot]
        for seed_data in data['Glucose'].values()]
    amp = [seed_data['df'].loc[:, columns_to_plot]
        for seed_data in data['Ampicillin (2 mg/L)'].values()]
    tet = [seed_data['df'].loc[:, columns_to_plot]
        for seed_data in data['Tetracycline (1.5 mg/L)'].values()]
    data = pd.concat(glc + amp + tet)
    agent_id_lengths = np.array([len(agent_id) for agent_id in data["Agent ID"]])
    n_generations = max(agent_id_lengths)
    previous_end_time = 0
    aligned_data = []
    # For each generation, assume start time = previous generation's end time
    # (0 for first gen.). Get median generation length and filter out all data
    # that exceeds that. Start time + median gen. length = new gen. start time
    for generation in range(1, n_generations +  1):
        generation_data = data.loc[agent_id_lengths == generation, :]
        grouped_data = generation_data.groupby(['Condition', 'Seed', 'Agent ID'])
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
    aligned_data = aligned_data.set_index(['Condition'])

    n_variables = len(columns_to_plot)
    if not axes:
        _, fresh_axes = plt.subplots(nrows=n_variables * 3, ncols=1, 
            sharex=True, figsize=(8, 2*n_variables))
        axes = np.ravel(fresh_axes)
    ax_idx = 0
    columns_to_plot = list(set(columns_to_plot) - METADATA)
    # Plot glucose alone but plot it with each antibiotic as control
    for grouping in CONDITION_GROUPINGS:
        curr_data = aligned_data.loc[grouping]
        # For antibiotic sims, limit glucose data to antibiotic run time
        if len(grouping) > 1:
            condition_1_data = aligned_data.loc[grouping[0]].loc[
                aligned_data.loc[grouping[0]]['Time'] >= SPLIT_TIME, :]
            condition_2_data = aligned_data.loc[grouping[1]].loc[
                aligned_data.loc[grouping[1]]['Time'] >= SPLIT_TIME, :]
            curr_data = pd.concat([condition_1_data, condition_2_data])
            
        for column in columns_to_plot:
            curr_ax = axes[ax_idx]
            ax_idx += 1
            g = sns.lineplot(
                data=curr_data, x='Time', y=column, hue='Condition', ax=curr_ax,
                errorbar=("pi", 95), estimator=np.median, legend=False)
            # Mark values where cell died
            death = curr_data.loc[curr_data['Death'], :]
            g = sns.scatterplot(
                data=death, x='Time', y=column, hue='Condition', ax=g,
                markers=['X'], style='Death', legend=False)
    if out:
        fig = g.get_figure()
        plt.tight_layout()
        fig.savefig('out/analysis/antibiotics_colony/' + 
            'generational_timeseries.png')
        plt.close(fig)


def plot_snapshot_timeseries(
    data: Dict[str, Dict[int, pd.DataFrame]]
) -> None:
    '''Plot a row of snapshot images that span a replicate for each condition.

    Args:
        data: Nested dictionary with experimental condition on outer level and
            initial seed on inner level. Each seed has a DataFrame where each
            row is an agent and each column is a variable of interest (e.g.
            count of an mRNA). All DataFrames contain some metadata columns:
            'Time', 'Death' (True if agent about to die), 'Agent ID', and
            'Boundary' (only used for snapshot plots), 'Condition', and 'Seed'.
            Each seed has two additional keys: 'bounds' is of the form [x, y]
            and gives the dimensions of the spatial environment and 'fields' is
            a dictionary timeseries of the 'fields' Store.
    '''
    for condition, seeds in data.items():
        # Plot first replicate for each condition by default
        condition_data = seeds[min(seeds.keys())]
        condition_df = condition_data['df']
        condition_df = condition_df.sort_values('Time')
        condition_fields = condition_data['fields']
        # Split data into 5 chunks of equal time for 5 snapshots
        time_chunks = np.array_split(condition_df['Time'].unique(), 4)
        split_times = [times[0] for times in time_chunks]
        split_times.append(time_chunks[-1][-1])
        condition_df = pd.concat([
            condition_df.loc[condition_df['Time']==time, :] for time in split_times
        ])
        condition_fields = {
            time: condition_fields[time]
            for time in condition_df['Time']
        }
        condition_bounds = condition_data['bounds']
        # Convert back to dictionary for snapshot plot function
        snapshot_data = {}         
        for time, agent_id, boundary in zip(
            condition_df['Time'], condition_df['Agent ID'], condition_df['Boundary']
        ):
            data_at_time = snapshot_data.setdefault(time, {})
            agent_at_time = data_at_time.setdefault(agent_id, {})
            agent_at_time['boundary'] = boundary
        snapshots_fig = plot_snapshots(
            agents=snapshot_data,
            fields=condition_fields,
            bounds=condition_bounds,
            # Glucose is the only significantly depleted molecule
            include_fields=['GLC[p]'],
            default_font_size=16,
            field_label_size=20,
            scale_bar_length=10,
            plot_width=2
        )
        snapshots_fig.subplots_adjust(wspace=0.7, hspace=0.1)
        plt.tight_layout()
        snapshots_fig.savefig('out/analysis/antibiotics_colony/' + 
            f'{condition.replace("/", "_")}_snapshots.png', bbox_inches='tight')
        plt.close(snapshots_fig)


def plot_concentration_timeseries(
    data: Dict[str, Dict[int, pd.DataFrame]],
    out: bool = True,
    axes: List[plt.Axes] = None,
) -> None:
    '''Normalize variables by volume to create bulk timeseries plot of
    concentrations.

    Args:
        data: Nested dictionary with experimental condition on outer level and
            initial seed on inner level. Each seed has a DataFrame where each
            row is an agent and each column is a variable of interest (e.g.
            count of an mRNA). All DataFrames contain some metadata columns:
            'Time', 'Death' (True if agent about to die), 'Agent ID', and
            'Boundary' (only used for snapshot plots), 'Condition', and 'Seed'.
        out: Immediately save and close current figure if True.
        axes: If supplied, columns are plotted sequentially on these Axes.
    '''
    columns_to_plot = list(set(data['Glucose'][0]['df'].columns)
        & CONCENTRATION_COLUMNS)
    glc = [seed_data['df'].loc[:, columns_to_plot + ['Volume', 'Condition', 'Time', 'Death']]
        for seed_data in data['Glucose'].values()]
    amp = [seed_data['df'].loc[:, columns_to_plot + ['Volume', 'Condition', 'Time', 'Death']]
        for seed_data in data['Ampicillin (2 mg/L)'].values()]
    tet = [seed_data['df'].loc[:, columns_to_plot + ['Volume', 'Condition', 'Time', 'Death']]
        for seed_data in data['Tetracycline (1.5 mg/L)'].values()]
    data = pd.concat(glc + amp + tet)
    # Convert to concentrations
    data = data.set_index(['Condition', 'Time', 'Death'])
    data = data.divide(data['Volume'], axis=0).drop(['Volume'], axis=1)
    data = data.reset_index()
    data = data.set_index(['Condition'])
    
    # if not axes:
    #     n_variables = len(columns_to_plot) * len(CONDITION_GROUPINGS)
    #     _, fresh_axes = plt.subplots(nrows=n_variables, ncols=1, 
    #         sharex=True, figsize=(4, 2*n_variables))
    #     axes = np.ravel(fresh_axes)
    # ax_idx = 0
    for grouping in CONDITION_GROUPINGS:
        condition_data = data.loc[grouping].copy()
        # For antibiotic sims, limit plot data to antibiotic run time
        if len(grouping) > 1:
            condition_1_data = condition_data.loc[grouping[0]].loc[
                condition_data.loc[grouping[0]]['Time'] >= SPLIT_TIME, :]
            condition_2_data = condition_data.loc[grouping[1]].loc[
                condition_data.loc[grouping[1]]['Time'] >= SPLIT_TIME, :]
            condition_data = pd.concat([condition_1_data, condition_2_data])
        condition_data = condition_data.reset_index()
        for column in columns_to_plot:
            # curr_ax = axes[ax_idx]
            # ax_idx += 1
            g = sns.lineplot(
                data=condition_data, x='Time', y=column,
                hue='Condition', legend=False,
                errorbar=("pi", 95), estimator=np.median,
                # ax=curr_ax
            )
            # Mark values where cell died
            death = condition_data.loc[condition_data['Death'], :]
            sns.scatterplot(
                data=death, x='Time', y=column, hue='Condition', ax=g,
                markers=['X'], style='Death', legend=False)
            if out:
                fig = g.get_figure()
                plt.tight_layout()
                fig.savefig('out/analysis/antibiotics_colony/' + 
                    f'{grouping[0].replace("/", "_")}_{column.replace("/", "_")}_concentration_timeseries.png')
                plt.close(fig)
