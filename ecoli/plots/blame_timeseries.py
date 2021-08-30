from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH
from ecoli.plots.blame_utils import get_bulk_processes, extract_bulk


def blame_timeseries(data,
                     topology,
                     molecules=None,
                     filename='out/ecoli_master/blame_timeseries.png'):

    if 'log_update' not in data.keys():
        raise ValueError(
            "Missing log_update in data; did you run simulation without logged updates?")

    max_t = data['time'][-1]

    bulk_processes = get_bulk_processes(topology)

    # Collect data into one dictionary
    collected_data = {}
    for process, updates in data['log_update'].items():
        if process not in bulk_processes.keys():
            break

        process_data = Counter()
        for port in updates.keys():
            if port not in bulk_processes[process]:
                break

            port_data = updates[port]
            # update process_data?

        collected_data[process] = dict(process_data)

    # convert dictionary to array
    collected_data = {process: idx_array_from(
        data) for process, data in collected_data.items()}

    # bulk_idx, process_idx, plot_data = extract_bulk(data, get_bulk_processes(topology))


def test_blame_timeseries():
    from vivarium.core.emitter import data_from_database, get_local_client

    EXPERIMENT_ID = "blame_timeseries_test_30/08/2021 17:34:09"

    if EXPERIMENT_ID:
        data, conf = data_from_database(EXPERIMENT_ID,
                                        get_local_client("localhost", "27017", "simulations"))
        topo = conf['topology']

    else:
        sim = EcoliSim.from_file(
            CONFIG_DIR_PATH + "/test_configs/test_blame.json")
        data = sim.run()
        topo = sim.ecoli.topology

    blame_timeseries(data, topo,
                     ["CPLX0-7452"],
                     'out/ecoli_master/test_blame_timeseries.png')


if __name__ == "__main__":
    test_blame_timeseries()
