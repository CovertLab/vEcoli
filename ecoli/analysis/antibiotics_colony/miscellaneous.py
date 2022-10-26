from typing import List, Dict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ecoli.analysis.antibiotics_colony import (
    CONDITION_GROUPINGS, CONCENTRATION_COLUMNS, DE_GENES)

def plot_colony_growth_rates(
    data: Dict[str, Dict[int, pd.DataFrame]],
    out: bool = True,
    axes: List[plt.Axes] = None,
) -> None:
    '''Plot traces of total colony mass and total colony growth rate.

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
    columns_to_plot = ['Dry mass', 'Growth']
    # Get total dry mass and growth at each time point for each replicate
    glc = [seed_data['df'].loc[:, columns_to_plot + ['Condition', 'Time']
        ].groupby(['Condition', 'Time']).sum()
        for seed_data in data['Glucose'].values()]
    amp = [seed_data['df'].loc[:, columns_to_plot + ['Condition', 'Time']
        ].groupby(['Condition', 'Time']).sum()
        for seed_data in data['Ampicillin (2 mg/L)'].values()]
    tet = [seed_data['df'].loc[:, columns_to_plot + ['Condition', 'Time']
        ].groupby(['Condition', 'Time']).sum()
        for seed_data in data['Tetracycline (1.5 mg/L)'].values()]
    data = pd.concat(glc + amp + tet)

    if not axes:
        n_variables = 2 * len(CONDITION_GROUPINGS)
        _, fresh_axes = plt.subplots(nrows=n_variables, ncols=1, 
            sharex=True, figsize=(8, 2*n_variables))
        axes = np.ravel(fresh_axes)
    ax_idx = 0
    for grouping in CONDITION_GROUPINGS:
        condition_data = data.loc[grouping]
        condition_data = condition_data.reset_index()
        for column in columns_to_plot:
            curr_ax = axes[ax_idx]
            ax_idx += 1
            sns.lineplot(
                data=condition_data, x='Time', y=column,
                hue='Condition', ax=curr_ax)
            # Log scale so linear means exponential growth
            curr_ax.set(yscale='log')
    if out:
        fig = curr_ax.get_figure()
        plt.tight_layout()
        fig.savefig('out/analysis/antibiotics_colony/' + 
            'growth_rate_timeseries.png')
        plt.close(fig)


def plot_final_fold_changes(
    data: Dict[str, Dict[int, pd.DataFrame]],
    out: bool = True,
    ax: plt.Axes = None,
) -> None:
    '''Plot bar charts of final gene and monomer fold change for key genes
    regulated as part of tetracycline response.

    Args:
        data: Nested dictionary with experimental condition on outer level and
            initial seed on inner level. Each seed has a DataFrame where each
            row is an agent and each column is a variable of interest (e.g.
            count of an mRNA). All DataFrames contain some metadata columns:
            'Time', 'Death' (True if agent about to die), 'Agent ID', and
            'Boundary' (only used for snapshot plots), 'Condition', and 'Seed'.
        out: Immediately save and close current figure if True.
        ax: Single Matplotlib axes instance to plot on
    '''
    # Plot PBP1 mRNA fold change as a control (should not change significantly)
    mrnas = [column for column in CONCENTRATION_COLUMNS
        if 'mRNA' in column]
    monomers = [column for column in CONCENTRATION_COLUMNS
        if 'monomer' in column]
    columns_to_plot = mrnas + monomers
    # Get average expression as concentration at final time point per replicate
    glc_max_time = data['Glucose'][0]['df'].loc[:, 'Time'].max()
    glc = [seed_data['df'].loc[
            seed_data['df']['Time']==glc_max_time,
            columns_to_plot+['Volume', 'Seed']]
        for seed_data in data['Glucose'].values()]
    glc = pd.concat([glc_rep.divide(glc_rep['Volume'], axis=0).drop(
        ['Volume'], axis=1) for glc_rep in glc])
    glc = glc.groupby('Seed').mean()
    tet_max_time = data['Tetracycline (1.5 mg/L)'][0]['df'].loc[:, 'Time'].max()
    tet = [seed_data['df'].loc[
            seed_data['df']['Time']==tet_max_time,
            columns_to_plot+['Volume', 'Seed']]
        for seed_data in data['Tetracycline (1.5 mg/L)'].values()]
    tet = pd.concat([tet_rep.divide(tet_rep['Volume'], axis=0).drop(
        ['Volume'], axis=1) for tet_rep in tet])
    tet = tet.groupby('Seed').mean()
    # Convert to fold change over glucose control
    data = tet / glc
    data = pd.melt(data, var_name='Gene/Monomer',
        value_name='Fold change (Tet./ Glc.)')
    data['Source'] = 'Simulated'

    # Get literature fold change where applicable
    tet_degenes = DE_GENES.loc[:, ['Gene name', 'Fold change']].rename(
        columns={
            rna_id: f'{rna_id} mRNA'
            for rna_id in DE_GENES['Gene name']
            if rna_id != 'MicF'
        }
    )
    tet_degenes = pd.melt(tet_degenes, var_name='Gene/Monomer',
        value_name='Fold change (Tet./ Glc.)')
    tet_degenes['Source'] = 'Literature'
    data = pd.concat([data, tet_degenes])

    if not ax:
        n_variables = len(columns_to_plot)
        _, ax = plt.subplots(nrows=1, ncols=1, 
            figsize=(8, 2*n_variables))
    sns.histplot(data=data, x='Fold change (Tet./ Glc.)',
        y='Gene/Monomer', hue='Source', ax=ax)
    if out:
        fig = ax.get_figure()
        plt.tight_layout()
        fig.savefig('out/analysis/antibiotics_colony/' + 
            'fold_change.png')
        plt.close(fig)
