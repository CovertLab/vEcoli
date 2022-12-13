import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ecoli.analysis.antibiotics_colony import COUNTS_PER_FL_TO_NANOMOLAR
from ecoli.analysis.antibiotics_colony.timeseries import plot_tag_snapshots


def plot_exp_growth_rate(data, metadata):
    grouped_agents = data.groupby(['Condition', 'Agent ID'])
    new_data = []
    aggregate_data = {
        'active_ribo_concs': [],
        'growth_rates': [],
        'tet_concs': [],
    }
    for _, agent_data in grouped_agents:
        delta_t = np.diff(agent_data.loc[:, 'Time'], append=0)
        # Ignore cells for which less than 100 timepoints of data exist
        if len(delta_t) < 100:
            continue
        delta_t[-1] = delta_t[-2]
        dry_mass = agent_data.loc[:, 'Dry mass']
        mass_ratio = dry_mass[1:].to_numpy() / dry_mass[:-1].to_numpy()
        mass_ratio = np.append(mass_ratio, mass_ratio[-1])
        agent_data['Doubling rate'] = np.log2(mass_ratio) / delta_t * 3600
        new_data.append(agent_data)
        aggregate_data['active_ribo_concs'].append(
            (agent_data.loc[:, 'Active ribosomes'] /
            agent_data.loc[:, 'Volume']).mean() *
            COUNTS_PER_FL_TO_NANOMOLAR / 1000)
        aggregate_data['growth_rates'].append(
            agent_data.loc[:, 'Doubling rate'].mean())
        aggregate_data['tet_concs'].append(np.round(
            agent_data.loc[:, 'Initial external tet.'].mean() * 1000, 3))
    aggregate_data = pd.DataFrame(aggregate_data)
    data = pd.concat(new_data)
    cmap = matplotlib.colormaps['Greys']
    tet_min = aggregate_data.loc[:, 'tet_concs'].min()
    tet_max = aggregate_data.loc[:, 'tet_concs'].max()
    norm = matplotlib.colors.Normalize(
        vmin=1.5*tet_min-0.5*tet_max, vmax=tet_max)
    tet_concs = aggregate_data.loc[:, 'tet_concs'].unique()
    palette = {tet_conc: cmap(norm(tet_conc)) for tet_conc in tet_concs}
    palette[3.375] = (0, 0.4, 1)
    joint = sns.jointplot(data=aggregate_data, x='active_ribo_concs',
        y='growth_rates', hue='tet_concs', palette=palette, marginal_kws={
            'common_norm': False}, joint_kws={'edgecolors': 'face'})
    joint.ax_joint.set_ylim(0, joint.ax_joint.get_ylim()[1])
    sns.despine(offset=0.1, trim=True, ax=joint.ax_joint)
    legend = joint.ax_joint.legend(frameon=False)
    legend.set_title('Tetracycline (uM)')
    joint.ax_joint.set_xlabel('Active ribosomes (mM)')
    joint.ax_joint.set_ylabel('Doubling rate (1/hr)')
    plt.tight_layout()
    plt.savefig('out/analysis/paper_figures/growth_rate_variation.svg')
    plt.close()

    # Get log 2 fold change over mean glucose growth rate
    glucose_data = data.loc[data.loc[:, 'Condition'] == 'Glucose', :]
    mean_growth_rate = glucose_data.loc[:, 'Doubling rate'].mean()
    fc_col = 'Growth rate\n($\mathregular{log_2}$ fold change)'
    data.loc[:, fc_col] = np.log2(data.loc[:, 'Doubling rate'] / mean_growth_rate)

    # Only include data from glucose and tetracycline MIC
    data = data.loc[np.isin(data.loc[:, 'Condition'],
        ['Glucose', 'Tetracycline (1.5 mg/L)']), :]

    # Set up custom divergent colormap
    cmp = matplotlib.colors.LinearSegmentedColormap.from_list(
        'divergent', [(0, 0.4, 1), (1, 1, 1), (0.678, 0, 0.125)])
    norm = matplotlib.colors.Normalize(vmin=-2.5, vmax=2.5)
    plot_tag_snapshots(
        data=data, metadata=metadata, tag_colors={fc_col: {
            'cmp': cmp, 'norm': norm}}, snapshot_times=np.array([
            1.9, 3.2, 4.5, 5.8, 7.1]) * 3600, show_membrane=True)
