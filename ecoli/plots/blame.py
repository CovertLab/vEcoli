import os
import json
import unum
import numpy as np
from scipy.sparse import coo_matrix
from matplotlib import pyplot as plt
import matplotlib.colors as colors

from collections import Counter

from ecoli.composites.ecoli_master import ECOLI_TOPOLOGY


def blame_plot(data, filename='out/ecoli_master/blame.png',
               highlighted_molecules=None,
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
        highlighted_molecules: A collection of molecules to highlight in red (or None).
        label_values: Whether to numerically label the heatmap cells with their values.
        color_normalize: whether to normalize values within (p)rocesses, (m)olecules, or (n)either.

    Returns:
        matplotlib axes and figure.
    """

    if 'log_update' not in data.keys():
        raise ValueError("Missing log_update in data; did you run vivarium-ecoli without -blame?")

    title = "Average Change (#mol/sec) in Bulk due to each Process (non-zero only, logarithmic color-scale)"

    max_t = data['time'][-1]
    bulk_idx, process_idx, plot_data = extract_bulk(data)
    plot_data = plot_data.toarray() / max_t  # convert counts to average rate

    # exclude zero-change molecules
    nonzero_mols = np.sum(plot_data, axis=1) != 0
    bulk_idx = bulk_idx[nonzero_mols]
    plot_data = plot_data[nonzero_mols, :]

    # sort molecules by sum of absolute changes
    sorted_mol_idx = np.argsort(-np.sum(np.abs(plot_data), axis=1))
    bulk_idx = bulk_idx[sorted_mol_idx]
    plot_data = plot_data[sorted_mol_idx, :]

    # Normalization within rows (molecules) or columns (processes)
    normalized_data = plot_data.copy()

    color_normalize = color_normalize.strip().lower()
    if color_normalize.startswith("p"):
        title += f'\n (normalizing within processes)'

        for col in range(normalized_data.shape[1]):
            data_col = normalized_data[:, col]  # get view of this column

            data_col = diverging_color_normalize(data_col)

            assert np.all(data_col <= 1)
            assert np.all(data_col >= -1)

    elif color_normalize.startswith("m"):
        title += f'\n (normalizing within molecules)'

        for row in range(normalized_data.shape[0]):
            data_row = normalized_data[row, :]  # get view of this row

            data_row = diverging_color_normalize(data_row)

            assert np.all(data_row <= 1)
            assert np.all(data_row >= -1)
    else:
        normalized_data = diverging_color_normalize(normalized_data)

    n_molecules = normalized_data.shape[1]
    n_processes = normalized_data.shape[0]
    fig, axs = plt.subplots(2, 2,
                            gridspec_kw={'height_ratios' : [n_processes, 1],
                                         'width_ratios' : [n_molecules, 1]})

    main_ax = axs[0, 0]
    molecules_total_ax = axs[0, 1]
    process_total_ax = axs[1, 0]
    total_total_ax = axs[1, 1]

    fig.set_size_inches(2 * (n_molecules + 1), (n_processes + 1) / 5)
    main_ax.imshow(normalized_data,
                   aspect='auto',
                   cmap=plt.get_cmap('seismic'))

    # get totals
    process_total = np.atleast_2d(plot_data.sum(axis=0))
    molecules_total = np.atleast_2d(plot_data.sum(axis=1))
    total_total = np.atleast_2d(plot_data.sum())

    # normalize totals
    process_total_norm = diverging_color_normalize(process_total)
    molecules_total_norm = diverging_color_normalize(molecules_total)

    # plot totals
    process_total_ax.imshow(process_total_norm, aspect='auto', cmap=plt.get_cmap('seismic'))
    molecules_total_ax.imshow(molecules_total_norm.T, aspect='auto', cmap=plt.get_cmap('seismic'))

    class SignNormalize(colors.Normalize):
        def __call__(self, value, clip=None):
            return (np.sign(value) + 1) / 2

    total_total_ax.imshow(total_total, aspect='auto', cmap=plt.get_cmap('seismic'), norm=SignNormalize())

    # show and rename ticks
    process_labels = [p.replace('_', '\n') for p in process_idx]

    main_ax.set_xticks(np.arange(normalized_data.shape[1]))
    main_ax.set_yticks(np.arange(normalized_data.shape[0]))
    main_ax.set_xticklabels(process_labels)
    main_ax.set_yticklabels(bulk_idx)

    molecules_total_ax.set_xticks([0])
    molecules_total_ax.set_yticks(np.arange(normalized_data.shape[0]))
    molecules_total_ax.set_xticklabels(['TOTAL'])
    molecules_total_ax.set_yticklabels(bulk_idx)

    process_total_ax.set_xticks(np.arange(normalized_data.shape[1]))
    process_total_ax.set_yticks([0])
    process_total_ax.set_xticklabels(process_labels)
    process_total_ax.set_yticklabels(['TOTAL'])

    total_total_ax.set_xticks([0])
    total_total_ax.set_yticks([0])
    total_total_ax.set_xticklabels(['TOTAL'])
    total_total_ax.set_yticklabels(['TOTAL'])

    # Put process ticks labels on correct sides
    main_ax.xaxis.tick_top()
    main_ax.xaxis.set_label_position('top')
    molecules_total_ax.xaxis.tick_top()
    molecules_total_ax.xaxis.set_label_position('top')
    molecules_total_ax.yaxis.tick_right()
    molecules_total_ax.yaxis.set_label_position('right')
    total_total_ax.yaxis.tick_right()
    total_total_ax.yaxis.set_label_position('right')

    # Highlight selected molecules
    if highlighted_molecules:
        highlight_idx = np.where(np.isin(bulk_idx, highlighted_molecules))[0]
        for i in highlight_idx:
            main_ax.get_yticklabels()[i].set_color("red")
            molecules_total_ax.get_yticklabels()[i].set_color("red")

    # Label cells with numeric values
    if label_values:
        for i in range(plot_data.shape[0]):
            for j in range(plot_data.shape[1]):
                if plot_data[i, j] != 0:
                    main_ax.text(j, i, f'{"-" if plot_data[i, j] < 0 else "+"}{plot_data[i, j]:.2f}/s',
                                 ha="center", va="center", color="w")

            if molecules_total_norm[0, i] != 0:
                val = molecules_total[0, i]
                molecules_total_ax.text(0, i, f'{"-" if val < 0 else "+"}{val:.2f}/s',
                                        ha="center", va="center", color="w")

        for i in range(plot_data.shape[1]):
            if process_total_norm[0, i] != 0:
                val = process_total[0, i]
                process_total_ax.text(i, 0, f'{"-" if val < 0 else "+"}{val:.2f}/s',
                                      ha="center", va="center", color="w")

        total_total_ax.text(0, 0, f'{"-" if total_total[0, 0] < 0 else "+"}{total_total[0, 0]:.2f}/s',
                            ha="center", va="center", color="w")

    main_ax.set_title(title)
    fig.tight_layout()

    if filename:
        plt.savefig(filename)

    return main_ax, fig


def extract_bulk(data):
    # Get relevant processes (those affecting bulk)
    bulk_processes = {}
    for process, ports in ECOLI_TOPOLOGY.items():
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
    return bulk_indices, process_idx, bulk_data


def diverging_color_normalize(count_data, transform_log=True):
    count_data = count_data.copy()

    # rescale logarithmically
    if transform_log:
        count_data[count_data > 0] = np.log(1 + count_data[count_data > 0])
        count_data[count_data < 0] = -np.log(1 - count_data[count_data < 0])

    # bring back to [-1, 1]
    if (count_data < 0).sum() > 0:
        count_data[count_data < 0] /= -(count_data[count_data < 0].min())
    if (count_data > 0).sum() > 0:
        count_data[count_data > 0] /= count_data[count_data > 0].max()

    return count_data


def idx_array_from(dictionary):
    idx = []
    values = []
    for k, v in dictionary.items():
        idx.append(k)
        values.append(v)

    return np.array(idx), np.array(values)


def write_json(path, numpy_dict):
    INFINITY = float('inf')

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif obj == INFINITY:
                return '__INFINITY__'
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, unum.Unum):
                return float(obj)
            else:
                return super(NpEncoder, self).default(obj)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as outfile:
        json.dump(numpy_dict, outfile, cls=NpEncoder)


def test_blame():
    try:
        with open("data/blame_test_data.json") as f:
            data = json.load(f)
    except FileNotFoundError:  # save test data if it does not exist
        from ecoli.composites.ecoli_master import run_ecoli
        data = run_ecoli(blame=True, total_time=4)
        write_json('data/blame_test_data.json', data)

    blame_plot(data, 'out/ecoli_master/blame_test.png', highlighted_molecules=['PD00413[c]'])


if __name__ == "__main__":
    test_blame()
