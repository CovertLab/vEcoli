from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ecoli.analysis.antibiotics_colony import MAX_TIME

def plot_death_distributions(
    data: pd.DataFrame,
    axes: List[plt.Axes] = None,
    columns_to_plot: Dict[str, tuple] = None
) -> None:
    '''Plot distributions of variables for cells one generation from death
    vs cells that survive until the end of the simulation (ampicillin only).

    Args:
        data: DataFrame where each row is an agent and each column is a variable
            of interest. Must have these columns: 'Time', 'Death', and 'Agent ID'.
            Include at most 1 condition and 1 seed.
        axes: Columns are plotted sequentially on these Axes.
        columns_to_plot: Dictionary of columns in data to plot sequentially on
            axes. Each column name corresponds to a RGB tuple to color that plot.
    '''
    # Average data over agent lifespan
    grouped_agent_data = data.groupby(by=['Agent ID'])
    column_agg = {column: np.mean for column in columns_to_plot}
    # If the cell dies, mark it as such
    column_agg['Death'] = 'max'
    # Get maximum time that agent is alive for
    column_agg['Time'] = 'max'
    avg_agent_data = grouped_agent_data.aggregate(column_agg)
    # Get data for cells that die
    dead_cell_data = avg_agent_data.loc[avg_agent_data['Death'], :]
    max_time = data['Time'].max()
    live_cell_data = avg_agent_data.loc[
        avg_agent_data['Time']==max_time, :]

    data = pd.concat([dead_cell_data, live_cell_data])
    ax_idx = 0
    for column, color in columns_to_plot.items():
        curr_ax = axes[ax_idx]
        ax_idx += 1
        palette = {True: (128, 128, 128), False: color}
        sns.histplot(
            data=data, x=column, hue='Death', ax=curr_ax,
            stat='percent', common_norm=False, element='step',
            palette=palette)


def plot_final_distributions(
    data: pd.DataFrame,
    axes: List[plt.Axes] = None,
    columns_to_plot: Dict[str, tuple] = None
) -> None:
    '''Plot data at the end of each simulation.

    Args:
        data: DataFrame where each row is an agent and each column is a variable
            of interest. Must have these columns: 'Time', 'Death', 'Agent ID',
            'Boundary', 'Condition', and 'Seed'. Include at most 2 conditions
            and 1 seed per condition. If more than one condition, assume first
            condition is a control and plot in gray.
        axes: Columns are plotted sequentially on these Axes.
        columns_to_plot: Dictionary of columns in data to plot sequentially on
            axes. Each column name corresponds to a RGB tuple to color that plot.
    '''
    data = data.loc[data.loc[:, 'Time'] == MAX_TIME]
    ax_idx = 0
    conditions = data.loc[:, 'Condition'].unique()
    for column, color in columns_to_plot.items():
        curr_ax = axes[ax_idx]
        ax_idx += 1
        palette = {
            conditions[0]: (128, 128, 128),
            conditions[1]: color}
        sns.histplot(
            data=data, x=column, hue='Condition', ax=curr_ax,
            stat='percent', common_norm=False, element='step',
            palette=palette)
