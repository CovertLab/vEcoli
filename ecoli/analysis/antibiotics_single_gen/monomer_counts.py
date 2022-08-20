import os
import numpy as np
import pandas as pd
import scipy.stats as st

from matplotlib import pyplot as plt

from vivarium.core.emitter import timeseries_from_data

from ecoli.analysis.db import access
from ecoli.library.sim_data import LoadSimData

COLORS = ['b', 'g', 'r', 'c', 'm', 'y']

class Plot:
    """
    Compare monomer counts (inc. complexed monomers) over the course
    of a single generation. Calculates 95% CI for labelled replicate
    groups (e.g. control, with tetracycline, etc.) and plots monomers
    whose 95% CIs overlap for <60% the shortest simulation time.
    
    Supply parameters for Savitzky-Golary filter as:
        {
            'window_length': length of filter window,
            'polyorder': order of polynomial (must be < window_length)
        }
    """

    def __init__(self, exp_ids, labels=None, colors=None, monomers_to_plot=[], 
                 out_file='out/analysis/monomer_counts.png'):
        labels = labels if labels else exp_ids
        if not colors:
            color_map = {}
            color_idx = 0
            for label in labels:
                if label not in color_map:
                    color_map[label] = COLORS[color_idx]
                    color_idx += 1
            colors = [color_map[label] for label in labels]
        self.monomers_to_plot = monomers_to_plot
        get_monomers_to_plot = not self.monomers_to_plot
        self.data = {}
        self.colors = colors
        all_monomer_ids = set()
        
        for exp_num, exp_id in enumerate(exp_ids):
            data, _, sim_config = access(
                exp_id, query=[('agents', '0', 'listeners', 'monomer_counts')])
            data = timeseries_from_data(data)
            time = data['time']
            data = np.array(data['agents']['0']['listeners']['monomer_counts'])
            # Data only includes first generation (agent 0)
            time = np.array(time[:len(data)])
            config = sim_config['metadata']
            # Apply sim_data modifications according to experiment config
            sim_data = LoadSimData(
                sim_data_path=config['sim_data_path'],
                seed=config['seed'],
                mar_regulon=config['mar_regulon'],
                rnai_data=config['process_configs'].get(
                    'ecoli-rna-interference')).sim_data
            # Get the names of monomers from sim_data
            monomer_ids = sim_data.process.translation.monomer_data["id"].tolist()
            for monomer_id in monomer_ids:
                all_monomer_ids.add(monomer_id)
            if monomers_to_plot:
                indices = np.nonzero(np.isin(monomer_ids, monomers_to_plot))[0]
            else:
                indices = np.arange(len(monomer_ids))
            for index in indices:
                if monomer_ids[index] not in self.data:
                    self.data[monomer_ids[index]] = {}
                if labels[exp_num] not in self.data[monomer_ids[index]]:
                    self.data[monomer_ids[index]][labels[exp_num]] = {
                        'time': time,
                        'count': [],
                        'color': self.colors[exp_num]
                    }
                # For each label, keep data up to length of shortest generation
                label_data = self.data[monomer_ids[index]][labels[exp_num]]
                if len(time) < len(label_data['time']):
                    label_data['time'] = time
                    for stored_idx in range(len(label_data['count'])):
                        label_data['count'][stored_idx] = label_data[
                            'count'][stored_idx][:len(time)]
                else:
                    time = time[:len(label_data['time'])]
                    data = data[:len(label_data['time'])]
                label_data['count'].append(data[:, index])
            print(f"Done loading data for {exp_id}.")
        
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        self.out_file = out_file
        
        # Display common protein name
        protein_flat = pd.read_csv("reconstruction/ecoli/flat/proteins.tsv", 
            sep="\t", comment="#")
        self.protein_id_to_common = {}
        for protein_id, common_name in zip(
            protein_flat['id'], protein_flat['common_name']):
            monomer_ids = [monomer_id for monomer_id in all_monomer_ids
                if protein_id in monomer_id]   
            for monomer_id in monomer_ids:
                self.protein_id_to_common[monomer_id] = common_name
        
        for monomer_id, count_data in self.data.items():
            count_cis = []
            ci_labels = []
            for exp_label, exp_data in count_data.items():
                exp_data['count'] = np.array(exp_data['count'])
                if len(exp_data['count']) > 1:
                    count_avg = np.mean(exp_data['count'], axis=0)
                    # Generate 95% confidence intervals using t distribution
                    temp_count_ci = np.array(st.t.interval(0.95, 
                        len(exp_data['count'])-1, loc=count_avg,
                        scale=st.sem(exp_data['count'], axis=0)))
                    exp_data['count_ci'] = np.where(
                        np.isfinite(temp_count_ci),
                        temp_count_ci, count_avg)
                    for ci_label, ci in zip(ci_labels, count_cis):
                        # Calculate fraction where upper bound < lower bound or
                        # lower bound > upper bound, only including time of the
                        # shorter simulation
                        shared_time = np.min(
                            [len(ci[0]), len(exp_data['count_ci'][1])])
                        sig_frac = np.mean(
                            np.less(exp_data['count_ci'][1][:shared_time],
                                ci[0][:shared_time])
                            | np.greater(exp_data['count_ci'][0][:shared_time],
                                ci[1][:shared_time]))
                        if sig_frac >= 0.4:
                            print(f'{monomer_id}: {sig_frac*100}% significant ({ci_label} vs {exp_label})')
                        if (sig_frac >= 0.4 and
                            monomer_id not in self.monomers_to_plot and
                            get_monomers_to_plot):
                            self.monomers_to_plot.append(monomer_id)
                    count_cis.append(exp_data['count_ci'])
                    ci_labels.append(exp_label)
                    exp_data['count'] = count_avg
                else:
                    exp_data['count'] = exp_data['count'][0]
            self.monomers_to_plot.sort()
        print(f"Plotting {len(self.monomers_to_plot)} monomer counts.")
        
        self.do_plot()

    def do_plot(self):
        fig_dim = int(np.ceil(np.sqrt(len(self.monomers_to_plot))))
        fig, axs = plt.subplots(fig_dim, fig_dim,
            figsize=(fig_dim*3, fig_dim*3), sharex=True, dpi=600)
        axs = np.ravel(axs)
        print("Done creating subplots.")
        
        # Data for legend
        legend_labels = []
        legend_handles = []
        for i, monomer_id in enumerate(self.monomers_to_plot):
            data = self.data[monomer_id]
            common = self.protein_id_to_common.get(monomer_id, monomer_id)
            
            for label, exp_data in data.items():
                axs[i].plot(exp_data['time'], exp_data['count'],
                    exp_data['color'], label=label)
                count_ci = exp_data.get('count_ci', [])
                if len(count_ci)>0:
                    axs[i].fill_between(exp_data['time'], count_ci[0],
                        count_ci[1], alpha=0.5, color=exp_data['color'])
                axs[i].set_title(common, fontsize=6)
            if len(legend_labels) < len(data):
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
        [
            "0 uM tet", "0 uM tet", 
            "0 uM tet", "0 uM tet",
            "base", "base", "base", "base"
         ],
        out_file='out/analysis/monomer_count.png'
    )