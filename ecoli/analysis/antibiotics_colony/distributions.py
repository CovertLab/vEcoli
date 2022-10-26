from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ecoli.analysis.antibiotics_colony import METADATA

def plot_death_distributions(
    data: Dict[str, Dict[int, pd.DataFrame]],
    out: bool = True,
    axes: List[plt.Axes] = None,
) -> None:
    '''Plot distributions of variables for cells one generation from death die
    vs cells that survive until the end of the simulation for ampicillin sims.

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
    # Only care about ampcillin sims at MIC
    data = [seed_data['df'] for seed_data in data['Ampicillin (2 mg/L)'].values()]
    data = pd.concat(data)
    # Average data over agent lifespan
    grouped_agent_data = data.groupby(by=['Seed', 'Agent ID'])
    columns_to_plot = set(data.columns) - METADATA
    column_agg = {column: np.mean for column in columns_to_plot}
    # If the cell dies, mark it as such
    column_agg['Death'] = 'max'
    # Get maximum time that agent is alive for
    column_agg['Time'] = 'max'
    avg_agent_data = grouped_agent_data.aggregate(column_agg)
    # Get data for cells that die
    dead_cell_data = avg_agent_data.loc[avg_agent_data['Death'], :]
    max_time = data["Time"].max()
    live_cell_data = avg_agent_data.loc[
        avg_agent_data['Time']==max_time, :]

    data = pd.concat([dead_cell_data, live_cell_data])

    if not axes:
        nrows = int(np.ceil(len(columns_to_plot) / 2))
        _, fresh_axes = plt.subplots(nrows=nrows, ncols=2, 
            figsize=(8, 3*nrows))
        axes = np.ravel(fresh_axes)
    ax_idx = 0
    for column in columns_to_plot:
        curr_ax = axes[ax_idx]
        ax_idx += 1
        sns.histplot(
            data=data, x=column, hue='Death', ax=curr_ax,
            stat='percent', common_norm=False, element='step')
    if out:
        fig = curr_ax.get_figure()
        plt.tight_layout()
        fig.savefig('out/analysis/antibiotics_colony/' + 
            f'{out}_amp_death_dist.png')
        plt.close(fig)


def plot_final_distributions(
    data: Dict[str, Dict[int, pd.DataFrame]],
    out: bool = True,
    axes: List[plt.Axes] = None,
) -> None:
    '''Plot data at the end of each simulation.

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
    # Get data at final time point for each condition
    amp_max_time = data['Ampicillin (2 mg/L)'][0]['df'].loc[:, 'Time'].max()
    amp_final = [
        amp_data['df'].loc[amp_data['df'].loc[:, 'Time']==amp_max_time, :]
        for amp_data in data['Ampicillin (2 mg/L)'].values()
    ]
    tet_max_time = data['Tetracycline (1.5 mg/L)'][0]['df'].loc[:, 'Time'].max()
    tet_final = [
        tet_data['df'].loc[tet_data['df'].loc[:, 'Time']==tet_max_time, :]
        for tet_data in data['Tetracycline (1.5 mg/L)'].values()
    ]
    glc_max_time = data['Glucose'][0]['df'].loc[:, 'Time'].max()
    glc_final = [
        glc_data['df'].loc[glc_data['df'].loc[:, 'Time']==glc_max_time, :]
        for glc_data in data['Glucose'].values()
    ]
    amp_final = pd.concat(amp_final + glc_final)
    tet_final = pd.concat(tet_final + glc_final)
    glc_final = pd.concat(glc_final)
    columns_to_plot = set(glc_final.columns) - METADATA
    if not axes:
        nrows = int(np.ceil(len(columns_to_plot) / 2)) * 3
        _, fresh_axes = plt.subplots(nrows=nrows, ncols=2, 
            figsize=(8, 3*nrows))
        axes = np.ravel(fresh_axes)
    ax_idx = 0
    for data in [glc_final, amp_final, tet_final]:
        for column in columns_to_plot:
            curr_ax = axes[ax_idx]
            ax_idx += 1
            sns.histplot(
                data=data, x=column, hue='Condition', ax=curr_ax,
                stat='percent', common_norm=False, element='step')
    if out:
        fig = curr_ax.get_figure()
        plt.tight_layout()
        fig.savefig('out/analysis/antibiotics_colony/' + 
            'final_dist.png')
        plt.close(fig)
