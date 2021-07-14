import os
import json
import numpy as np
from scipy.sparse import coo_matrix
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from ecoli.library.logging import write_json

from collections import Counter

from ecoli.composites.ecoli_master import ECOLI_TOPOLOGY

ecoli_topology = ECOLI_TOPOLOGY.copy()

def blame_plot(data, filename='out/ecoli_master/blame.png',
               selected_molecules=None,
               selected_processes=None,
               highlight_molecules=None,
               label_values=True,
               color_normalize="n"):
    """
    Given data from a logged simulation (e.g. by running python run_ecoli(blame=True)),
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
        raise ValueError("Missing log_update in data; did you call run_ecoli without blame=True?")

    max_t = data['time'][-1]

    bulk_idx, process_idx, plot_data = extract_bulk(data)
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


def extract_bulk(data):
    # Get relevant processes (those affecting bulk)
    bulk_processes = {}
    for process, ports in ecoli_topology.items():
        for port, path in ports.items():
            if 'bulk' in path:
                if process not in bulk_processes:
                    bulk_processes[process] = []

                bulk_processes[process].append(port)

    # Collect data into one dictionary
    collected_data = {}
    for process, updates in data['log_update'].items():
        if process not in bulk_processes.keys():
            break

        process_data = Counter()
        for port in updates.keys():
            if port not in bulk_processes[process]:
                break

            port_data = {k: np.sum(v) for k, v in updates[port].items()}
            process_data.update(port_data)

        collected_data[process] = dict(process_data)

    # convert dictionary to array
    collected_data = {process: idx_array_from(data) for process, data in collected_data.items()}

    bulk_indices = np.array([])
    for idx, _ in collected_data.values():
        bulk_indices = np.concatenate((bulk_indices, idx[~np.isin(idx, bulk_indices)]))

    col_i = 0
    row = []  # molecules
    col = []  # processes
    data_out = []  # update data
    process_idx = []
    sorter = np.argsort(bulk_indices)
    for process, (idx, value) in collected_data.items():
        if idx.size != 0:  # exclude processes with zero update
            rows = sorter[np.searchsorted(bulk_indices, idx, sorter=sorter)]
            row = np.concatenate((row, rows))
            col += [col_i] * len(rows)
            data_out = np.concatenate((data_out, value))
            process_idx.append(process)
            col_i += 1

    bulk_data = coo_matrix((data_out, (row.astype(int), np.array(col))))
    return bulk_indices, np.array(process_idx), bulk_data


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


def reposition_ticks(ax, x="bottom", y="left"):
    if x == "top":
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
    elif x == "bottom":
        ax.xaxis.tick_bottom()
        ax.xaxis.set_label_position('bottom')
    else:
        raise ValueError(f"{x} is not a valid place for x-ticks")

    if y == "left":
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position('left')
    elif y == "right":
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
    else:
        raise ValueError(f"{y} is not a valid place for y-ticks")


def idx_array_from(dictionary):
    idx = []
    values = []
    for k, v in dictionary.items():
        idx.append(k)
        values.append(v)

    return np.array(idx), np.array(values)


def test_blame():
    try:
        with open("data/blame_test_data.json") as f:
            data = json.load(f)
    except FileNotFoundError:  # save test data if it does not exist
        from ecoli.composites.ecoli_master import run_ecoli
        data = run_ecoli(blame=True, total_time=4)
        write_json('data/blame_test_data.json', data)

    blame_plot(data, 'out/ecoli_master/blame_test.png',
               highlight_molecules=['PD00413[c]'])


if __name__ == "__main__":
    test_blame()
