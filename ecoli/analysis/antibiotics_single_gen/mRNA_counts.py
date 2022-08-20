import os
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.signal import savgol_filter

from matplotlib import pyplot as plt

from vivarium.core.emitter import timeseries_from_data

from ecoli.analysis.db import access
from ecoli.library.sim_data import LoadSimData

COLORS = ['b', 'g', 'r', 'c', 'm', 'y']

class Plot:
    """
    Compare counts of specified mRNAs over the course of a single
    generation. Calculates 95% CI for labelled replicate groups
    (e.g. control, with tetracycline, etc.) and plots mRNAs whose
    95% CIs overlap for <60% the shortest simulation time.
    
    Also plots gene copy number for context. Set flag norm to normalize
    by gene copy number. Supply parameters for Savitzky-Golary filter as:
        {
            'window_length': length of filter window,
            'polyorder': order of polynomial (must be < window_length)
        }
    
    # TODO: Implement norm flag
    """

    def __init__(self, exp_ids, labels=None, colors=None, mrnas_to_plot=[], 
                 out_file='out/analysis/mrna_counts.png', norm=True, savgol_args={}):
        labels = labels if labels else exp_ids
        if not colors:
            color_map = {}
            color_idx = 0
            for label in labels:
                if label not in color_map:
                    color_map[label] = COLORS[color_idx]
                    color_idx += 1
            colors = [color_map[label] for label in labels]
        self.mrnas_to_plot = mrnas_to_plot
        get_mrnas_to_plot = not self.mrnas_to_plot
        self.data = {}
        self.colors = colors
        self.norm = norm

        for exp_num, exp_id in enumerate(exp_ids):
            data, _, sim_config = access(exp_id, query=[
                ('agents', '0', 'listeners', 'mRNA_counts'),
                ('agents', '0', 'listeners', 'rna_synth_prob', 'gene_copy_number',)])
            data = timeseries_from_data(data)
            # Skip empty initial and 1 sec emits
            time = data['time'][2:]
            mrna_counts = np.array(data['agents']['0']['listeners']['mRNA_counts'][2:])
            gene_copy_num = np.array(data['agents']['0']['listeners']
                ['rna_synth_prob']['gene_copy_number'][2:])
            # Data only includes first generation (agent 0)
            time = np.array(time[:len(mrna_counts)])
            config = sim_config['metadata']
            # Apply sim_data modifications according to experiment config
            sim_data = LoadSimData(
                sim_data_path=config['sim_data_path'],
                seed=config['seed'],
                mar_regulon=config['mar_regulon'],
                rnai_data=config['process_configs'].get(
                    'ecoli-rna-interference')).sim_data
            
            # Get the names of mRNAs from sim_data
            is_mrna = sim_data.process.transcription.rna_data['is_mRNA']
            mrna_ids = sim_data.process.transcription.rna_data['id'][is_mrna]
            gene_copy_num = gene_copy_num[:, is_mrna]
            
            if norm:
                mrna_counts_norm = np.divide(mrna_counts, gene_copy_num)
                # micF-ompF duplex does not have a gene copy #
                mrna_counts = np.where(np.isfinite(mrna_counts_norm), 
                    mrna_counts_norm, mrna_counts)
            # Smooth data using Savitsky-Golary filter
            if savgol_args:
                mrna_counts = savgol_filter(
                    mrna_counts,
                    window_length=savgol_args['window_length'],
                    polyorder=savgol_args['polyorder'],
                    axis=0)
            
            if mrnas_to_plot:
                indices = np.nonzero(np.isin(mrna_ids, mrnas_to_plot))[0]
            else:
                indices = np.arange(len(mrna_ids))
            for index in indices:
                if mrna_ids[index] not in self.data:
                    self.data[mrna_ids[index]] = {}
                if labels[exp_num] not in self.data[mrna_ids[index]]:
                    self.data[mrna_ids[index]][labels[exp_num]] = {
                        'time': time,
                        'mrna_count': [],
                        'copy_num': [],
                        'color': self.colors[exp_num]
                    }
                # For each label, keep data up to length of shortest generation
                label_data = self.data[mrna_ids[index]][labels[exp_num]]
                if len(time) < len(label_data['time']):
                    label_data['time'] = time
                    for stored_idx in range(len(label_data['mrna_count'])):
                        label_data['mrna_count'][stored_idx] = label_data[
                            'mrna_count'][stored_idx][:len(time)]
                        label_data['copy_num'][stored_idx] = label_data[
                            'copy_num'][stored_idx][:len(time)]
                else:
                    time = time[:len(label_data['time'])]
                    mrna_counts = mrna_counts[:len(label_data['time'])]
                    gene_copy_num = gene_copy_num[:len(label_data['time'])]
                label_data['mrna_count'].append(mrna_counts[:, index])
                label_data['copy_num'].append(gene_copy_num[:, index])
            print(f"Done loading data for {exp_id}.")
        
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        self.out_file = out_file
        
        # Display common gene name
        rna_flat = pd.read_csv("reconstruction/ecoli/flat/rnas.tsv", 
            sep="\t", comment="#")
        self.rna_id_to_common = {
            rna_id + "[c]": common for rna_id, common in zip(
                rna_flat['id'], rna_flat['common_name'])
        }
        
        for mrna_id, mrna_counts in self.data.items():
            mrna_count_cis = []
            ci_labels = []
            for exp_label, exp_data in mrna_counts.items():
                exp_data['mrna_count'] = np.array(exp_data['mrna_count'])
                exp_data['copy_num'] = np.array(exp_data['copy_num'])
                if len(exp_data['mrna_count']) > 1:
                    mrna_count_avg = np.mean(exp_data['mrna_count'], axis=0)
                    copy_num_avg = np.mean(exp_data['copy_num'], axis=0)
                    # Generate 95% confidence intervals using t distribution
                    temp_mrna_count_ci = np.array(st.t.interval(0.95, 
                        len(exp_data['mrna_count'])-1, loc=mrna_count_avg,
                        scale=st.sem(exp_data['mrna_count'], axis=0)))
                    exp_data['mrna_count_ci'] = np.where(
                        np.isfinite(temp_mrna_count_ci),
                        temp_mrna_count_ci, mrna_count_avg)
                    for ci_label, ci in zip(ci_labels, mrna_count_cis):
                        # Calculate fraction where upper bound < lower bound or
                        # lower bound > upper bound, only including time of the
                        # shorter simulation
                        shared_time = np.min(
                            [len(ci[0]), len(exp_data['mrna_count_ci'][1])])
                        sig_frac = np.mean(
                            np.less(exp_data['mrna_count_ci'][1][:shared_time],
                                ci[0][:shared_time])
                            | np.greater(exp_data['mrna_count_ci'][0][:shared_time],
                                ci[1][:shared_time]))
                        if sig_frac >= 0.4:
                            print(f'{mrna_id}: {sig_frac*100}% significant ({ci_label} vs {exp_label})')
                        if (sig_frac >= 0.4 and
                            mrna_id not in self.mrnas_to_plot and
                            get_mrnas_to_plot):
                            self.mrnas_to_plot.append(mrna_id)
                    mrna_count_cis.append(exp_data['mrna_count_ci'])
                    ci_labels.append(exp_label)
                    exp_data['mrna_count'] = mrna_count_avg
                    exp_data['copy_num'] = copy_num_avg
                else:
                    exp_data['mrna_count'] = exp_data['mrna_count'][0]
                    exp_data['copy_num'] = exp_data['copy_num'][0]
            self.mrnas_to_plot.sort()
        print(f"Plotting {len(self.mrnas_to_plot)} mRNA counts.")
        self.plot_individual()        

    def plot_individual(self):
        fig_dim = int(np.ceil(np.sqrt(len(self.mrnas_to_plot))))
        fig, axs = plt.subplots(fig_dim, fig_dim,
            figsize=(fig_dim*3, fig_dim*3), sharex=True, dpi=600)
        axs = np.ravel(axs)
        if not self.norm:
            twin_axs = [axs[idx].twinx() for idx in range(len(axs))]
        print("Done creating subplots.")
        
        # Data for legend
        legend_labels = []
        legend_handles = []
        for i, mrna_id in enumerate(self.mrnas_to_plot):
            data = self.data[mrna_id]
            common = self.rna_id_to_common.get(mrna_id, mrna_id)
            for label, exp_data in data.items():
                axs[i].plot(exp_data['time'], exp_data['mrna_count'],
                    exp_data['color'], label=label)
                mrna_count_ci = exp_data.get('mrna_count_ci', [])
                if len(mrna_count_ci)>0:
                    axs[i].fill_between(exp_data['time'], mrna_count_ci[0],
                        mrna_count_ci[1], alpha=0.5, color=exp_data['color'])
                if not self.norm:
                    twin_axs[i].plot(exp_data['time'], exp_data['copy_num'],
                        color=exp_data['color'], linestyle='--', label=f'{label} copy #')
                axs[i].set_title(common)
            if len(legend_labels) < len(data):
                if not self.norm:
                    legend_handles, legend_labels = [(a + b) for a, b in zip(
                        axs[i].get_legend_handles_labels(), 
                        twin_axs[i].get_legend_handles_labels())]
                else:
                    legend_handles, legend_labels = axs[i].get_legend_handles_labels()
        fig.legend(legend_handles, legend_labels, loc='lower left', borderaxespad=0.,
            bbox_to_anchor=(0., 1.02, 1., .102), ncol=3, mode='expand')

        plt.tight_layout()
        print("Done with tight_layout.")
        plt.savefig(self.out_file, bbox_inches='tight')
        print("Done saving figure.")
        plt.close("all")
        return fig
    
if __name__ == "__main__":
    Plot([
        # 3.375 uM tet.
        "2022-08-15_17-44-04_988625+0000",
        "2022-08-15_18-33-06_362648+0000",
        "2022-08-15_21-49-24_819030+0000",
        "2022-08-17_00-30-10_058639+0000",
        # 0 uM tet.
        "2022-08-15_23-00-34_973906+0000",
        "2022-08-16_16-00-40_928301+0000",
        "2022-08-16_18-48-44_621220+0000",
        "2022-08-16_19-42-37_994009+0000",
        # Baseline
        "2022-08-16_21-09-39_797440+0000",
        "2022-08-16_22-04-54_797907+0000",
        "2022-08-16_23-04-23_500547+0000",
        "2022-08-17_00-21-24_422276+0000"
        ],
        ["3.375 uM tet", "3.375 uM tet", 
         "3.375 uM tet", "3.375 uM tet",
         "0 uM tet", "0 uM tet", 
         "0 uM tet", "0 uM tet",
         "base", "base", "base", "base"],
        savgol_args={
            'window_length': 50,
            'polyorder': 3
        },
        out_file='out/analysis/mrna_counts_test.png',
        norm=False
    )
