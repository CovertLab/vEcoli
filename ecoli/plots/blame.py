import numpy as np
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
    indexes = []
    for process, data in collected_data.items():
        indexes += list(data.keys())

    result = np.zeros((len(bulk_processes), len(indexes)))
    



    import ipdb; ipdb.set_trace()
    return collected_data