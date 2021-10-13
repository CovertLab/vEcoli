import os
from collections import Counter
import numpy as np
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH
from ecoli.plots.blame_utils import get_bulk_processes, extract_bulk, idx_array_from


def blame_timeseries(data,
                     topology,
                     molecules,
                     filename='out/ecoli_master/blame_timeseries.png',
                     yscale='log'):

    if 'log_update' not in data.keys():
        raise ValueError(
            "Missing log_update in data; did you run simulation without logged updates?")

    # Collect data into one dictionary
    # of the form: {process : {molecule : timeseries}}
    bulk_processes = get_bulk_processes(topology)
    plot_data = {}
    for process, updates in data['log_update'].items():
        if process not in bulk_processes.keys():
            continue

        plot_data[process] = {}
        for port in updates.keys():
            if port not in bulk_processes[process]:
                continue

            port_data = updates[port]
            for k, v in port_data.items():
                if k in molecules:  # Only keep selected molecules
                    if k in plot_data[process]:
                        plot_data[process][k] += np.array(v)
                    else:
                        plot_data[process][k] = np.array(v)

    # Remove processes that do not affect any selected molecules
    plot_data = {process: {molecule: timeseries
                           for molecule, timeseries in data.items()
                           if not all(timeseries == 0)}
                 for process, data in plot_data.items()}
    plot_data = {process: data
                 for process, data in plot_data.items()
                 if data != {}}

    # Start plotting!
    time = data['time'][1:]
    max_t = data['time'][-1]

    # One subplot per molecule
    fig, axs = plt.subplots(len(molecules), 1)
    for i, molecule in enumerate(molecules):
        axs[i].set_title(f"Change in {molecule}")
        axs[i].set_ylabel("# molecules")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_xticks(time)
        axs[i].axhline(y=0, color="k", linestyle="--", alpha=0.5)

        # Need to keep track of separate totals for positive, negative
        # entries at each time step, so that positive entries get stacked above 0,
        # and negative entries get stacked below.
        total_pos = np.zeros_like(time)
        total_neg = np.zeros_like(time)
        for process, process_data in plot_data.items():
            if molecule in process_data:
                molecule_data = process_data[molecule]
                axs[i].bar(time, molecule_data,
                           bottom=np.where(molecule_data > 0,
                                           total_pos, total_neg),
                           label=process)
                total_pos += np.clip(molecule_data, 0, None)
                total_neg += np.clip(molecule_data, None, 0)

        axs[i].set_yscale(yscale)

        # Plot net change
        axs[i].legend(bbox_to_anchor=(1.04, 0.5),
                      loc="center left", borderaxespad=0)
        axs[i].plot(time, total_pos + total_neg,
                    color="k", label="net change")

    # Sizing and spacing
    fig.set_size_inches(4 + max_t,  # include space for legend(s)
                        3 * len(molecules))  # height prop. to number of plots
    fig.tight_layout(pad=2.0)

    # Save plot to file
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)

    # Return axis and figure
    return plt.gca(), plt.gcf()


def test_blame_timeseries():
    # TODO:
    # - make work with partitioning (currently, log-update port names need work to match non-partitioned case)
    # - add back processes
    # - get working with unique molecules (separate argument for unique)
    # - serializers

    from vivarium.core.emitter import (
        data_from_database, get_local_client, timeseries_from_data)

    EXPERIMENT_ID = None  # "d811d69e-0cf6-11ec-a1ab-00155df92294"

    if EXPERIMENT_ID:
        data, conf = data_from_database(EXPERIMENT_ID,
                                        get_local_client("localhost", "27017", "simulations"))
        data = timeseries_from_data(data)
        topo = conf['topology']

    else:
        sim = EcoliSim.from_file()
        # CONFIG_DIR_PATH + "/test_configs/test_blame.json")
        # sim.emitter = "database"
        # sim.initial_time = 1000
        sim.partition = False
        sim.log_updates = True
        sim.emit_topology = False
        sim.emit_processes = False
        sim.total_time = 10
        sim.exclude_processes = ["ecoli-two-component-system",
                                 "ecoli-chromosome-structure",
                                 "ecoli-polypeptide-elongation"]
        data = sim.run()
        topo = sim.ecoli.topology

    molecules = [
        "EG10841-MONOMER",
        "EG10321-MONOMER",
        "EG11545-MONOMER",
        "EG11967-MONOMER",
        "FLAGELLAR-MOTOR-COMPLEX",
        "G361-MONOMER",
        "CPLX0-7451",
        "CPLX0-7452"  # Final flagella molecule
    ]

    blame_timeseries(data, topo,
                     # ['WATER[c]', 'APORNAP-CPLX[c]', 'TRP[c]'], #, "CPLX0-7452"],
                     molecules,
                     'out/ecoli_master/test_blame_timeseries.png',
                     yscale="linear")


if __name__ == "__main__":
    test_blame_timeseries()
