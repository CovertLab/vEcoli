import os
import json
import unum
import numpy as np
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt

from collections import Counter

from ecoli.composites.ecoli_master import ECOLI_TOPOLOGY

def blame_plot(data, filename='out/ecoli_master/blame.png'):
    data = extract_bulk(data)

    fig, ax = plt.subplots()
    im = ax.imshow(data)

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

            port_data = {k : np.sum(v) for k, v in updates[port].items()}
            process_data.update(port_data)

        collected_data[process] = dict(process_data)

    # convert dictionary to array
    bulk_indices = []
    for process, data in collected_data.items():
        bulk_indices += list(data.keys())
    bulk_indices = np.array(bulk_indices)

    row = []  # processes
    col = np.array([])  # molecules
    data_out = []
    for i, (process, data) in enumerate(collected_data.items()):
        cols = np.where(np.isin(bulk_indices, np.array(data.keys())))
        col = np.concatenate(col, cols)
        row += [i for x in range(len(cols))]
        data_out.append(data.values())

        import ipdb; ipdb.set_trace()

    return csr_matrix([data_out, [row, col]])


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



if __name__=="__main__":
    test_blame()