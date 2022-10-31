from typing import List, Dict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ecoli.analysis.antibiotics_colony import MAX_TIME, SPLIT_TIME, DE_GENES

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
    data = data.set_index(['Condition'])

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
    
    # By default, plot first condition in gray and second in red
    if not palette:
        palette = {conditions[0]: (128, 128, 128)}
        if len(conditions) > 1:
            palette[conditions[1]] = (255, 0, 0)
    sns.lineplot(
        data=data, x='Time', y='Dry mass',
        hue='Condition', ax=ax, palette=palette)
    # Log scale so linear means exponential growth
    ax.set(yscale='log')


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
        axes: Single instance of Matplotlib Axes to plot on.
        genes_to_plot: List of gene names to include in plot.
    '''
    mrnas = [f'{gene} mRNA' for gene in genes_to_plot]
    monomers = [f'{gene.capitalize()} monomer' for gene in genes_to_plot]
    columns_to_plot = mrnas + monomers
    columns_to_include = list(set(columns_to_plot) +
        {'Condition', 'Volume', 'Seed'})
    data = data.loc[:, columns_to_include]
    data = data.set_index(['Condition', 'Seed'])
    data = data.divide(data['Volume'], axis=0).drop(['Volume'], axis=1)
    data = data.reset_index()
    # Get average expression as concentration per condition
    data = data.groupby(['Condition', 'Seed']).mean()
    # Convert to fold change over glucose control
    data = data.loc[data.index[1]] / data.loc[data.index[0]]
    data = pd.melt(data, var_name='Gene name',
        value_name='Fold change (Tet./ Glc.)')
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
