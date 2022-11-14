from typing import List, Dict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ecoli.analysis.antibiotics_colony import (
    MAX_TIME, SPLIT_TIME, DE_GENES, restrict_data)

def plot_colony_growth_rates(
    data: pd.DataFrame,
    ax: plt.Axes = None,
    palette: Dict[str, tuple] = None,
) -> None:
    '''Plot traces of total colony mass and total colony growth rate.

    Args:
        data: DataFrame where each row is an agent and each column is a variable
            of interest. Must have these columns: 'Time', 'Condition', and
            'Dry Mass'. The first experimental condition in the 'Condition'
            column is treated as a control (default gray). Include at most
            2 conditions and 1 seed per condition.
        axes: Instance of Matplotlib Axes to plot on
        palette: Dictionary mapping experimental conditions to RGB color tuples
    '''
    columns_to_include = ['Dry mass', 'Condition', 'Time']
    data = data.loc[:, columns_to_include]
    mask = (data.loc[:, 'Condition']=='Glucose') & (
        data.loc[:, 'Time'] > SPLIT_TIME) & (
        data.loc[:, 'Time'] <= MAX_TIME)
    remaining_glucose_data = data.loc[mask, :]
    data = restrict_data(data)
    data = pd.concat([data, remaining_glucose_data])
    data = data.groupby(['Condition', 'Time']).sum().reset_index()
    
    # Convert time to hours
    data.loc[:, 'Time'] = data.loc[:, 'Time'] / 3600

    sns.lineplot(
        data=data, x='Time', y='Dry mass',
        hue='Condition', ax=ax, palette='viridis', errorbar=None)
    # Set y-limits so that major ticks surround data
    curr_ylimits = np.log10([
        data.loc[:, 'Dry mass'].min(),
        data.loc[:, 'Dry mass'].max()])
    ax.set_ylim(10**np.floor(curr_ylimits[0]), 10**np.ceil(curr_ylimits[1]))
    max_x = np.round(data.loc[:, 'Time'].max(), 1)
    ticks = list(np.arange(0, max_x, 2, dtype=int)) + [max_x]
    labels = [str(tick) for tick in ticks]
    ax.set_xticks(ticks=ticks, labels=labels)
    # Log scale so linear means exponential growth
    ax.set(yscale='log')
    sns.despine(ax=ax, offset=3, trim=True)


def plot_final_fold_changes(
    data: pd.DataFrame,
    ax: plt.Axes = None,
    genes_to_plot: List[str] = None,
) -> None:
    '''Plot bar charts of final gene and monomer fold change for key genes
    regulated as part of tetracycline response.

    Args:
        data: DataFrame where each row is an agent and each column is a variable
            of interest. Must have these columns: 'Time', 'Condition', 'Seed',
            and 'Dry Mass'. The first experimental condition in the 'Condition'
            column is treated as a control. Include >1 seed for error bars.
        ax: Single instance of Matplotlib Axes to plot on.
        genes_to_plot: List of gene names to include in plot.
    '''
    data = data.loc[data.loc[:, 'Time'] <= MAX_TIME, :]
    mrnas = [f'{gene} mRNA' for gene in genes_to_plot if gene != 'MicF']
    monomers = [f'{gene[0].upper() + gene[1:]} monomer'
        for gene in genes_to_plot if gene != 'MicF']
    columns_to_plot = mrnas + monomers
    columns_to_include = list(set(columns_to_plot) |
        {'Condition', 'Seed'})
    data = data.loc[:, columns_to_include]
    # data = data.set_index(['Condition', 'Seed'])
    # data = data.divide(data['Volume'], axis=0).drop(['Volume'], axis=1)
    # data = data.reset_index()
    # Get average expression as concentration per condition
    data = data.groupby(['Condition', 'Seed']).mean()
    # Convert to fold change over glucose control
    data = data.loc[data.index[1]] / data.loc[data.index[0]]
    data['Source'] = 'Simulated'

    # Get literature fold change where applicable
    tet_degenes = DE_GENES.loc[:, ['Gene name', 'Fold change']].set_index(
        'Gene name').drop('MicF', axis=0).reset_index()
    tet_degenes = pd.melt(tet_degenes, id_vars=['Gene name'],
        value_name='Fold change (Tet./ Glc.)').drop('variable', axis=1)
    tet_degenes['Source'] = 'Literature'
    tet_degenes = tet_degenes.set_index('Gene name')
    tet_degenes.index = [f'{gene_name} mRNA' for gene_name in tet_degenes.index]
    data = pd.concat([data, tet_degenes])

    sns.barplot(data=data, x='Fold change (Tet./ Glc.)',
        y='Gene name', hue='Source', ax=ax, errorbar='sd')


def plot_vs_distance_from_center(
    data: pd.DataFrame,
    bounds: tuple = None,
    ax: plt.Axes = None,
    column_to_plot: str = None,
):
    '''Plot scatter plot of a given column vs distance from environment center.

    Args:
        data: DataFrame where each row is an agent and each column is a variable
            of interest. Supply at most 1 condition. Multiple seeds are OK.
        bounds: Tuple representing width and height of environment.
        axes: Single instance of Matplotlib Axes to plot on.
        column_to_plot: Column to plot against distance from center.
    '''
    data = data.loc[data.loc[:, column_to_plot] > 0, :]
    if len(data) == 0:
        return
    center = np.array(bounds) / 2
    locations = np.array(data.loc[:, 'Boundary'].apply(
        lambda x: x['location']).tolist())
    data['Distance'] = np.linalg.norm(locations-center, axis=1)

    sns.regplot(data=data, x='Distance', y=column_to_plot, ax=ax)
