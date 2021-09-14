import os
import json
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors

from ecoli.library.logging import write_json
from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH

from ecoli.plots.plot_utils import reposition_ticks
from ecoli.plots.blame_utils import get_bulk_processes, extract_bulk, idx_array_from


def blame_plot(data, 
               topology,
               filename='out/ecoli_master/blame.png',
               selected_molecules=None,
               selected_processes=None,
               highlight_molecules=None,
               label_values=True,
               color_normalize="n"):
    """
    Given data from a simulation with logged updates (e.g. by running from CLI with --log_updates flag set),
    generates a heatmap where the columns are processes, rows are molecules, and
    cell colors reflect the average rate of change in a molecule over the whole simulation
    due to a particular process.

    Args:
        data: Data from a logged ecoli_master simulation.
        filename: The file to save the plot to. To skip writing to file, set this to None.
        selected_molecules: if not None, restricts to the specified molecules.
        selected_processes: if not None, restricts to the specified processes.
        highlight_molecules: A collection of molecules to highlight in red (or None).
        label_values: Whether to numerically label the heatmap cells with their values.
        color_normalize: whether to normalize values within (p)rocesses, (m)olecules, or (n)either.

    Returns:
        matplotlib axes and figure.
    """

    if 'log_update' not in data.keys():
        raise ValueError("Missing log_update in data; did you run simulation without logged updates?")

    max_t = data['time'][-1]

    bulk_idx, process_idx, plot_data = extract_bulk(data, get_bulk_processes(topology))
    plot_data = plot_data.toarray() / max_t  # convert counts to average rate

    # restrict to selected molecules and processes
    if selected_molecules:
        selected = np.isin(bulk_idx, selected_molecules)
        bulk_idx = bulk_idx[selected]
        plot_data = plot_data[selected, :]

    if selected_processes:
        selected = np.isin(process_idx, selected_processes)
        process_idx = process_idx[selected]
        plot_data = plot_data[:, selected]

    # exclude zero-change molecules
    nonzero_mols = np.sum(plot_data, axis=1) != 0
    bulk_idx = bulk_idx[nonzero_mols]
    plot_data = plot_data[nonzero_mols, :]

    # sort molecules by sum of absolute changes
    sorted_mol_idx = np.argsort(-np.sum(np.abs(plot_data), axis=1))
    bulk_idx = bulk_idx[sorted_mol_idx]
    plot_data = plot_data[sorted_mol_idx, :]

    n_molecules = plot_data.shape[1]
    n_processes = plot_data.shape[0]
    fig, axs = plt.subplots(2, 2,
                            gridspec_kw={'height_ratios': [n_processes, 1],
                                         'width_ratios': [n_molecules, 1]})

    ((main_ax, molecules_total_ax),
     (process_total_ax, total_total_ax)) = axs

    # Normalization within rows (molecules) or columns (processes)
    color_normalize = color_normalize.strip().lower()
    normalize_settings = {'p' : ('processes', "cols"),
                          'm' : ('molecules', "rows"),
                          'n' : (None, None)}
    norm_str, within = normalize_settings[color_normalize]

    title = (f"Average Change (#mol/sec) in Bulk due to each Process over {max_t} seconds\n"
             f"(non-zero only, logarithmic color scale{f' normalizing within {norm_str}' if norm_str else ''})")

    fig.set_size_inches(2 * (n_molecules + 3), (n_processes + 3) / 5)
    main_ax.imshow(-plot_data, aspect='auto', cmap=plt.get_cmap('seismic'),
                   norm=DivergingNormalize(within=within))

    # plot totals
    process_total = np.atleast_2d(plot_data.sum(axis=0))
    molecules_total = np.atleast_2d(plot_data.sum(axis=1))
    total_total = np.atleast_2d(plot_data.sum())

    process_total_ax.imshow(-process_total,
                            aspect='auto', cmap=plt.get_cmap('seismic'),
                            norm=DivergingNormalize())
    molecules_total_ax.imshow(-molecules_total.T,
                              aspect='auto', cmap=plt.get_cmap('seismic'),
                              norm=DivergingNormalize())

    total_total_ax.imshow(-total_total, aspect='auto', cmap=plt.get_cmap('seismic'), norm=SignNormalize())

    # show and rename ticks
    process_labels = [p.replace('_', '\n') for p in process_idx]

    main_ax.set_xticks(np.arange(plot_data.shape[1]))
    main_ax.set_yticks(np.arange(plot_data.shape[0]))
    main_ax.set_xticklabels(process_labels)
    main_ax.set_yticklabels(bulk_idx)

    molecules_total_ax.set_xticks([0])
    molecules_total_ax.set_yticks(np.arange(plot_data.shape[0]))
    molecules_total_ax.set_xticklabels(['TOTAL'])
    molecules_total_ax.set_yticklabels(bulk_idx)

    process_total_ax.set_xticks(np.arange(plot_data.shape[1]))
    process_total_ax.set_yticks([0])
    process_total_ax.set_xticklabels(process_labels)
    process_total_ax.set_yticklabels(['TOTAL'])

    total_total_ax.set_xticks([0])
    total_total_ax.set_yticks([0])
    total_total_ax.set_xticklabels(['TOTAL'])
    total_total_ax.set_yticklabels(['TOTAL'])

    # Put process ticks labels on correct sides
    reposition_ticks(main_ax, "top", "left")
    reposition_ticks(molecules_total_ax, "top", "right")
    reposition_ticks(total_total_ax, "bottom", "right")

    # Highlight selected molecules
    if highlight_molecules:
        highlight_idx = np.where(np.isin(bulk_idx, highlight_molecules))[0]
        for i in highlight_idx:
            main_ax.get_yticklabels()[i].set_color("red")
            molecules_total_ax.get_yticklabels()[i].set_color("red")

    # Label cells with numeric values
    if label_values:
        for i in range(plot_data.shape[0]):
            for j in range(plot_data.shape[1]):
                if plot_data[i, j] != 0:
                    main_ax.text(j, i, f'{sign_str(plot_data[i, j])}{plot_data[i, j]:.2f}/s',
                                 ha="center", va="center", color="w")

            val = molecules_total[0, i]
            if val != 0:
                molecules_total_ax.text(0, i, f'{sign_str(val)}{val:.2f}/s',
                                        ha="center", va="center", color="w")

        for i in range(plot_data.shape[1]):
            val = process_total[0, i]
            if val != 0:
                process_total_ax.text(i, 0, f'{sign_str(val)}{val:.2f}/s',
                                      ha="center", va="center", color="w")

        total_total_ax.text(0, 0, f'{sign_str(total_total[0, 0])}{total_total[0, 0]:.2f}/s',
                            ha="center", va="center", color="w")

    main_ax.set_title(title)
    fig.tight_layout()

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)

    return axs, fig


class SignNormalize(colors.Normalize):
    def __call__(self, value, clip=None):
        return (np.sign(value) + 1) / 2


class DivergingNormalize(colors.Normalize):
    def __init__(self, transform_log=True, within=None):
        self.transform_log = transform_log
        self.within = within
        super().__init__()

    def __call__(self, count_data):
        def diverging_color_normalize(data):
            # rescale logarithmically
            if self.transform_log:
                data[data > 0] = np.log(1 + data[data > 0])
                data[data < 0] = -np.log(1 - data[data < 0])

            # bring to [-1, 1]
            if (data < 0).sum() > 0:
                data[data < 0] /= -(data[data < 0].min())
            if (data > 0).sum() > 0:
                data[data > 0] /= data[data > 0].max()

            # scale to [0, 1]
            data += 1
            data /= 2

        if self.within is None:
            diverging_color_normalize(count_data)
        elif self.within == "rows":
            for row in range(count_data.shape[0]):
                diverging_color_normalize(count_data[row, :])
        elif self.within == "cols":
            for col in range(count_data.shape[1]):
                diverging_color_normalize(count_data[:, col])

        return count_data


def sign_str(val):
    return '-' if val < 0 else '+'


def test_blame():
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + "/test_configs/test_blame.json")
    data = sim.run()

    blame_plot(data, sim.ecoli.topology,
            'out/ecoli_master/blame_test.png',
            highlight_molecules=['PD00413[c]', 'PHOR-CPLX[c]'])


if __name__ == "__main__":
    test_blame()
