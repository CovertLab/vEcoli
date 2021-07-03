import os
import json
import unum
import numpy as np
from scipy.sparse import coo_matrix
from matplotlib import pyplot as plt
import matplotlib.colors as colors

from collections import Counter

from ecoli.composites.ecoli_master import ECOLI_TOPOLOGY


# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def blame_plot(data, filename='out/ecoli_master/blame.png'):
    if 'log_update' not in data.keys():
        raise ValueError("Missing log_update in data; did you run vivarium-ecoli without -blame?")

    bulk_idx, process_idx, plot_data = extract_bulk(data)
    plot_data = plot_data.toarray()

    # exclude zero-change molecules
    nonzero_mols = np.sum(plot_data, axis=1) != 0
    bulk_idx = bulk_idx[nonzero_mols]
    plot_data = plot_data[nonzero_mols, :]

    # sort molecules by sum of absolute changes
    sorted_mol_idx = np.argsort(np.sum(np.abs(plot_data), axis=1))
    bulk_idx = bulk_idx[sorted_mol_idx]
    plot_data = plot_data[sorted_mol_idx, :]

    # normalize for each process
    # col_sums = np.sum(plot_data, axis=0)
    # col_sums[col_sums == 0] = 1
    # plot_data = plot_data / col_sums
    # min_val = np.min(plot_data)
    # max_val = np.max(plot_data)

    # normalize for each molecule
    row_sums = plot_data.sum(axis=1)
    row_sums[row_sums == 0] = 1
    plot_data = (plot_data.T / row_sums).T
    min_val = np.min(plot_data)
    max_val = np.max(plot_data)

    fig, ax = plt.subplots()
    fig.set_size_inches(2 * plot_data.shape[1], plot_data.shape[0] / 8)
    im = ax.imshow(plot_data,
                   aspect='auto',
                   cmap=plt.get_cmap('seismic'),
                   clim=(min_val, max_val),
                   norm=MidpointNormalize(midpoint=0, vmin=min_val, vmax=max_val))

    # show and rename ticks
    ax.set_xticks(np.arange(plot_data.shape[1]))
    ax.set_yticks(np.arange(plot_data.shape[0]))
    ax.set_xticklabels(process_idx)
    ax.set_yticklabels(bulk_idx)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
    #          rotation_mode="anchor")

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    # # Loop over data dimensions and create text annotations.
    # for i in range(len(vegetables)):
    #     for j in range(len(farmers)):
    #         text = ax.text(j, i, harvest[i, j],
    #                        ha="center", va="center", color="w")
    #
    # ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()

    plt.savefig(filename)


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

    bulk_indices = set()
    for idx, _ in collected_data.values():
        bulk_indices.update(idx)
    bulk_indices = np.array(list(bulk_indices))

    col_i = 0
    row = []  # molecules
    col = []  # processes
    data_out = []  # update data
    process_idx = []
    for process, (idx, value) in collected_data.items():
        rows = np.where(np.isin(bulk_indices, idx))[0]
        if rows.size != 0:
            row = np.concatenate((row, rows))
            col += [col_i for _ in range(len(rows))]
            data_out = np.concatenate((data_out, value))
            process_idx.append(process)
            col_i += 1

    bulk_data = coo_matrix((data_out, (row.astype(int), np.array(col))))
    return bulk_indices, process_idx, bulk_data


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

    blame_plot(data, 'out/ecoli_master/blame_test.png')


if __name__ == "__main__":
    test_blame()
