import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.api import anova_lm

from ecoli.analysis.antibiotics_colony import COUNTS_PER_FL_TO_NANOMOLAR
from ecoli.analysis.antibiotics_colony.timeseries import plot_tag_snapshots


def plot_exp_growth_rate(data, metadata):
    grouped_agents = data.groupby(['Condition', 'Agent ID'])
    new_data = []
    for _, agent_data in grouped_agents:
        delta_t = np.diff(agent_data.loc[:, 'Time'], append=0)
        # Ignore cells for which less than 10 timepoints of data exist
        # to avoid outliers from instability in first few timesteps
        if len(delta_t) < 10:
            continue
        delta_t[-1] = delta_t[-2]
        dry_mass = agent_data.loc[:, 'Dry mass']
        mass_ratio = dry_mass[1:].to_numpy() / dry_mass[:-1].to_numpy()
        mass_ratio = np.append(mass_ratio, mass_ratio[-1])
        agent_data['Doubling rate'] = np.log2(mass_ratio) / delta_t * 3600
        agent_data['active_ribo_concs'] = (agent_data.loc[
            :, 'Active ribosomes'] / agent_data.loc[:, 'Volume'] *
            COUNTS_PER_FL_TO_NANOMOLAR / 1000)
        agent_data['active_rnap_concs'] = (agent_data.loc[
            :, 'Active RNAP'] / agent_data.loc[:, 'Volume'] *
            COUNTS_PER_FL_TO_NANOMOLAR / 1000)
        agent_data['tet_concs'] = np.round(
            agent_data.loc[:, 'Initial external tet.'] * 1000, 3)
        new_data.append(agent_data)

    data = pd.concat(new_data)
    cmap = matplotlib.colormaps['Greys']
    tet_min = data.loc[:, 'tet_concs'].min()
    tet_max = data.loc[:, 'tet_concs'].max()
    norm = matplotlib.colors.Normalize(
        vmin=1.5*tet_min-0.5*tet_max, vmax=tet_max)
    tet_concs = data.loc[:, 'tet_concs'].unique()
    palette = {tet_conc: cmap(norm(tet_conc)) for tet_conc in tet_concs}
    palette[3.375] = (0, 0.4, 1)
    
    time_boundaries = np.linspace(11550, 26000, 4)
    time_boundaries[-1] = 26002
    cols_to_plot = ['active_ribo_concs', 'active_rnap_concs', 
        'Doubling rate', 'tet_concs', 'Agent ID', 'Condition']
    ylim = (0, 1.41)
    xlim = (4, 26)
    for i in range(3):
        time_filter = ((data.loc[:, 'Time'] >= time_boundaries[i]) &
            (data.loc[:, 'Time'] < time_boundaries[i+1]))
        filtered_data = data.loc[time_filter, cols_to_plot]
        mean_data = filtered_data.groupby(['Condition', 'Agent ID']).mean()
        joint = sns.jointplot(data=mean_data, x='active_ribo_concs',
            y='Doubling rate', hue='tet_concs', palette=palette, marginal_kws={
                'common_norm': False}, joint_kws={'edgecolors': 'face'}, height=4)
        joint.ax_joint.set_ylim(ylim)
        joint.ax_joint.set_xlim(xlim)
        if i == 0:
            sns.despine(offset=0.1, trim=True, ax=joint.ax_joint)
            sns.despine(trim=True, ax=joint.ax_marg_x, left=True)
            sns.despine(trim=True, ax=joint.ax_marg_y, bottom=True)
            legend = joint.ax_joint.legend(frameon=False, loc='lower left')
            legend.set_title('Tetracycline (uM)')
            joint.ax_joint.set_xlabel('Active ribosomes (mM)')
            joint.ax_joint.set_ylabel('Doubling rate (1/hr)')
        else:
            sns.despine(offset=0.1, left=True, trim=True, ax=joint.ax_joint)
            sns.despine(trim=True, ax=joint.ax_marg_x, left=True)
            sns.despine(trim=True, ax=joint.ax_marg_y, bottom=True)
            joint.ax_joint.set_xlabel('Active ribosomes (mM)')
            joint.ax_joint.yaxis.set_visible(False)
        plt.savefig(f'out/analysis/paper_figures/growth_rate_var_ribo_{i}.svg')
        plt.close()
    
    xlim = (0, 10)
    ylim = (0, 0.09)
    for i in range(3):
        time_filter = ((data.loc[:, 'Time'] >= time_boundaries[i]) &
            (data.loc[:, 'Time'] < time_boundaries[i+1]))
        filtered_data = data.loc[time_filter, cols_to_plot]
        mean_data = filtered_data.groupby(['Condition', 'Agent ID']).mean()
        mean_data['Ribo-normed doubling rate'] = (
            mean_data.loc[:, 'Doubling rate'] / 
            mean_data.loc[:, 'active_ribo_concs'])
        joint = sns.jointplot(data=mean_data, x='active_rnap_concs',
            y='Ribo-normed doubling rate', hue='tet_concs', palette=palette,
            marginal_kws={'common_norm': False}, joint_kws={
                'edgecolors': 'face'}, height=4)
        joint.ax_joint.set_ylim(ylim)
        joint.ax_joint.set_xlim(xlim)
        if i == 0:
            sns.despine(offset=3, trim=True, ax=joint.ax_joint)
            sns.despine(trim=True, ax=joint.ax_marg_x, left=True)
            sns.despine(trim=True, ax=joint.ax_marg_y, bottom=True)
            legend = joint.ax_joint.legend(frameon=False, loc='upper right')
            legend.set_title('Tetracycline (uM)')
            joint.ax_joint.set_xlabel('mRNA (mM)')
            joint.ax_joint.set_ylabel('Adj. doubling rate (1/hr/mM ribosome)')
        else:
            sns.despine(offset=3, left=True, trim=True, ax=joint.ax_joint)
            sns.despine(trim=True, ax=joint.ax_marg_x, left=True)
            sns.despine(trim=True, ax=joint.ax_marg_y, bottom=True)
            joint.ax_joint.legend().remove()
            joint.ax_joint.set_xlabel('mRNA (mM)')
            joint.ax_joint.yaxis.set_visible(False)
        plt.savefig(f'out/analysis/paper_figures/growth_rate_var_mrna_{i}.svg')
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
