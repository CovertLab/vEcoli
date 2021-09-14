from collections import Counter
from functools import reduce

import numpy as np
from scipy.sparse import coo_matrix


def idx_array_from(dictionary):
    idx = []
    values = []
    for k, v in dictionary.items():
        idx.append(k)
        values.append(v)

    return np.array(idx), np.array(values)


def get_bulk_processes(topology):
    # Get relevant processes (those affecting bulk)
    bulk_processes = {}
    for process, ports in topology.items():
        for port, path in ports.items():
            if 'bulk' in path:
                if process not in bulk_processes:
                    bulk_processes[process] = []

                bulk_processes[process].append(port)
    
    return bulk_processes


def extract_bulk(data, bulk_processes):
    # Collect data into one dictionary
    collected_data = {}
    for process, updates in data['log_update'].items():
        if process not in bulk_processes.keys():
            continue

        process_data = Counter()
        for port in updates.keys():
            if port not in bulk_processes[process]:
                continue

            port_data = {k: np.sum(v) for k, v in updates[port].items()}
            process_data.update(port_data)

        collected_data[process] = dict(process_data)

    # convert dictionary to array
    collected_data = {process: idx_array_from(data) for process, data in collected_data.items()}

    bulk_indices = np.array([])
    for idx, _ in collected_data.values():
        bulk_indices = np.concatenate((bulk_indices, idx[~np.isin(idx, bulk_indices)]))

    col_i = 0
    row = np.array([])  # molecules
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