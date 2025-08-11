import os
from duckdb import DuckDBPyConnection
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from typing import Any, Optional

from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH
from ecoli.library.schema import bulk_name_to_idx


def get_bulk_processes(topology):
    # Get relevant processes (those affecting bulk)
    bulk_processes = {}
    for process, ports in topology.items():
        # Only care about evolver molecule count changes
        if "_requester" in process:
            continue
        for port, path in ports.items():
            if "bulk" in path:
                if process not in bulk_processes:
                    bulk_processes[process] = []

                bulk_processes[process].append(port)

    return bulk_processes


def blame_plot(
    data,
    topology,
    bulk_ids,
    filename="out/ecoli_sim/blame.png",
    selected_molecules=None,
    selected_processes=None,
    highlight_molecules=None,
    label_values=True,
    color_normalize="n",
):
    """
    Given data from a simulation with logged updates (e.g. by running from CLI with --log_updates flag set),
    generates a heatmap where the columns are processes, rows are molecules, and
    cell colors reflect the average rate of change in a molecule over the whole simulation
    due to a particular process.

    Args:
        data: Data from a logged ecoli simulation.
        topology: Topology of logged ecoli simulation (e.g. sim.ecoli_experiment.topology).
        bulk_ids: Array of bulk IDs in correct order (can get from initial_state).
        filename: The file to save the plot to. To skip writing to file, set this to None.
        selected_molecules: if not None, restricts to the specified molecules.
        selected_processes: if not None, restricts to the specified processes.
        highlight_molecules: A collection of molecules to highlight in red (or None).
        label_values: Whether to numerically label the heatmap cells with their values.
        color_normalize: whether to normalize values within (p)rocesses, (m)olecules, or (n)either.

    Returns:
        matplotlib axes and figure.
    """

    if "log_update" not in data.keys():
        raise ValueError(
            "Missing log_update in data; did you run simulation without logged updates?"
        )

    max_t = data["time"][-1] - data["time"][0]

    included_procs, plot_data = extract_bulk(
        data, get_bulk_processes(topology), bulk_ids
    )
    plot_data = plot_data / max_t  # convert counts to average rate

    # restrict to selected molecules and processes
    if selected_molecules:
        plot_data = plot_data[np.isin(bulk_ids, selected_molecules), :]

    if selected_processes:
        plot_data = plot_data[:, np.isin(included_procs, selected_processes)]

    # exclude zero-change molecules
    nonzero_mols = np.sum(plot_data, axis=1) != 0
    bulk_ids = bulk_ids[nonzero_mols]
    plot_data = plot_data[nonzero_mols, :]

    # sort molecules by sum of absolute changes
    sorted_mol_ids = np.argsort(-np.sum(np.abs(plot_data), axis=1))
    bulk_ids = bulk_ids[sorted_mol_ids]
    plot_data = plot_data[sorted_mol_ids, :]

    n_molecules = plot_data.shape[1]
    n_processes = plot_data.shape[0]
    fig, axs = plt.subplots(
        2,
        2,
        gridspec_kw={
            "height_ratios": [n_processes, 1],
            "width_ratios": [n_molecules, 1],
        },
    )

    ((main_ax, molecules_total_ax), (process_total_ax, total_total_ax)) = axs

    # Normalization within rows (molecules) or columns (processes)
    color_normalize = color_normalize.strip().lower()
    normalize_settings = {
        "p": ("processes", "cols"),
        "m": ("molecules", "rows"),
        "n": (None, None),
    }
    norm_str, within = normalize_settings[color_normalize]

    title = (
        f"Average Change (#mol/sec) in Bulk due to each Process over {max_t} seconds\n"
        f"(non-zero only, logarithmic color scale{f' normalizing within {norm_str}' if norm_str else ''})"
    )

    if selected_molecules:
        fig.set_size_inches(
            2 * (n_molecules + 3) + 10, (n_processes + 3) / 5 + 10
        )  # Make margins larger
    else:
        fig.set_size_inches(2 * (n_molecules + 3), (n_processes + 3) / 5)
    main_ax.imshow(
        -plot_data,
        aspect="auto",
        cmap=plt.get_cmap("seismic"),
        norm=DivergingNormalize(within=within),
    )

    # plot totals
    process_total = np.atleast_2d(plot_data.sum(axis=0))
    molecules_total = np.atleast_2d(plot_data.sum(axis=1))
    total_total = np.atleast_2d(plot_data.sum())

    process_total_ax.imshow(
        -process_total,
        aspect="auto",
        cmap=plt.get_cmap("seismic"),
        norm=DivergingNormalize(),
    )
    molecules_total_ax.imshow(
        -molecules_total.T,
        aspect="auto",
        cmap=plt.get_cmap("seismic"),
        norm=DivergingNormalize(),
    )

    total_total_ax.imshow(
        -total_total, aspect="auto", cmap=plt.get_cmap("seismic"), norm=SignNormalize()
    )

    # show and rename ticks
    process_labels = [p.replace("_", "\n") for p in included_procs]

    main_ax.set_xticks(np.arange(plot_data.shape[1]))
    main_ax.set_yticks(np.arange(plot_data.shape[0]))
    main_ax.set_xticklabels(process_labels)
    main_ax.set_yticklabels(bulk_ids)

    molecules_total_ax.set_xticks([0])
    molecules_total_ax.set_yticks(np.arange(plot_data.shape[0]))
    molecules_total_ax.set_xticklabels(["TOTAL"])
    molecules_total_ax.set_yticklabels(bulk_ids)

    process_total_ax.set_xticks(np.arange(plot_data.shape[1]))
    process_total_ax.set_yticks([0])
    process_total_ax.set_xticklabels(process_labels)
    process_total_ax.set_yticklabels(["TOTAL"])

    total_total_ax.set_xticks([0])
    total_total_ax.set_yticks([0])
    total_total_ax.set_xticklabels(["TOTAL"])
    total_total_ax.set_yticklabels(["TOTAL"])

    # Put process ticks labels on correct sides
    reposition_ticks(main_ax, "top", "left")
    reposition_ticks(molecules_total_ax, "top", "right")
    reposition_ticks(total_total_ax, "bottom", "right")

    # Highlight selected molecules
    if highlight_molecules:
        highlight_idx = np.where(np.isin(bulk_ids, highlight_molecules))[0]
        for i in highlight_idx:
            main_ax.get_yticklabels()[i].set_color("red")
            molecules_total_ax.get_yticklabels()[i].set_color("red")

    # Label cells with numeric values
    if label_values:
        for i in range(plot_data.shape[0]):
            for j in range(plot_data.shape[1]):
                if plot_data[i, j] != 0:
                    main_ax.text(
                        j,
                        i,
                        f"{sign_str(plot_data[i, j])}{plot_data[i, j]:.2f}/s",
                        ha="center",
                        va="center",
                        color="w",
                    )

            val = molecules_total[0, i]
            if val != 0:
                molecules_total_ax.text(
                    0,
                    i,
                    f"{sign_str(val)}{val:.2f}/s",
                    ha="center",
                    va="center",
                    color="w",
                )

        for i in range(plot_data.shape[1]):
            val = process_total[0, i]
            if val != 0:
                process_total_ax.text(
                    i,
                    0,
                    f"{sign_str(val)}{val:.2f}/s",
                    ha="center",
                    va="center",
                    color="w",
                )

        total_total_ax.text(
            0,
            0,
            f"{sign_str(total_total[0, 0])}{total_total[0, 0]:.2f}/s",
            ha="center",
            va="center",
            color="w",
        )

    main_ax.set_title(title)
    fig.tight_layout()

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)

    return axs, fig


def extract_bulk(data, bulk_processes, bulk_ids):
    """
    Returns bulk updates in form of the array collected_data
    with dimensions (n_bulk_mols x n_processes), where n_processes
    is given by the keys that are shared by bulk_processes and
    data['log_update']. Shared processes are also returned in order.
    """
    included_procs = list(data["log_update"].keys() & bulk_processes.keys())
    collected_data = np.zeros((len(bulk_ids), len(included_procs)))
    # Apply bulk updates to a fake bulk count array
    # and retrieve final deltas for each process
    fake_bulk = np.zeros(len(bulk_ids), dtype=int)
    for proc_idx, process in enumerate(included_procs):
        updates = data["log_update"][process]
        for port in updates.keys():
            if port not in bulk_processes[process]:
                continue
            for update in updates[port]:
                for bulk_update in update:
                    fake_bulk[bulk_update[0]] += bulk_update[1]
                collected_data[:, proc_idx] += fake_bulk
                fake_bulk[:] = 0
    return included_procs, collected_data


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
    return "-" if val < 0 else "+"


def reposition_ticks(ax, x="bottom", y="left"):
    if x == "top":
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
    elif x == "bottom":
        ax.xaxis.tick_bottom()
        ax.xaxis.set_label_position("bottom")
    else:
        raise ValueError(f"{x} is not a valid place for x-ticks")

    if y == "left":
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position("left")
    elif y == "right":
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
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
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + "/test_configs/test_blame.json")
    sim.build_ecoli()
    bulk_ids = sim.generated_initial_state["agents"]["0"]["bulk"]["id"]
    sim.run()
    data = sim.query()
    data = {"time": data["time"], **data["agents"]["0"]}

    # TODO: Adapt this code to work with new Numpy update format
    blame_plot(
        data,
        sim.ecoli_experiment.topology["agents"]["0"],
        bulk_ids,
        "out/ecoli_sim/blame_test.png",
        highlight_molecules=["PD00413[c]", "PHOR-CPLX[c]"],
    )


def preprocess_data(data, bulk_ids, bulk_processes, molecules):
    """
    Prepares raw data for blame-timeseries plot.
    Returns data in the form time, process, values_array
    where time is a numpy array of times, process is a list of
    process names, and values_array is a numpy array of the form
    (molecule x time x process).
    """

    molecule_idx = bulk_name_to_idx(molecules, bulk_ids)
    processes = list(bulk_processes.keys())
    x = np.array(list(data.keys()))

    values_array = np.zeros([len(molecules), len(x), len(processes)])
    # Create a fake bulk array to apply all updates to for each process
    # at each timestep and get final count
    fake_bulk = np.zeros(len(bulk_ids))
    for j, timepoint in enumerate(x):
        for k, process in enumerate(processes):
            fake_bulk[:] = 0
            path = bulk_processes[process][0]
            logged = data[timepoint]["log_update"]
            if process not in logged:
                continue
            bulk_updates = logged[process].get(path, ())
            for bulk_update in bulk_updates:
                fake_bulk[bulk_update[0]] += bulk_update[1]
            values_array[:, j, k] = fake_bulk[molecule_idx]

    return x, processes, values_array


def signed_stacked_bar(ax, x, y, bar_labels):
    """
    ax: Axes object
    x: x values (1d array)
    y: y-values (len(x) columns by # stacked bars rows)

    Creates a stacked bar chart in the specified Axes, where
    y's with negative values represent bars below y=0, and
    y's with positive values represent bars above y=0.
    """
    # Need to keep track of separate totals for positive, negative
    # entries at each time step, so that positive entries get stacked above 0,
    # and negative entries get stacked below.
    total_pos = np.zeros(len(x), dtype=np.float64)
    total_neg = np.zeros_like(total_pos)

    for series in range(y.shape[1]):
        data = y[:, series]
        ax.bar(
            x,
            data,
            bottom=np.where(data > 0, total_pos, total_neg),
            label=bar_labels[series],
        )
        total_pos += np.clip(data, 0, None)
        total_neg += np.clip(data, None, 0)

    # Plot net change
    ax.plot(x, total_pos + total_neg, color="k", label="net change")


def blame_timeseries(
    data: dict,
    topology: dict,
    bulk_ids: list[str],
    molecules: list[str],
    filename: Optional[str] = None,
    yscale: str = "linear",
) -> tuple[mpl.axes.Axes, mpl.figure.Figure]:
    """
    Generates timeseries blame plots for the selected bulk molecules assuming
    that bulk data is an array of counts ordered by bulk_ids and saves to the
    specified output file. Timeseries blame plots show the change in molecule
    counts due to each process at each timestep. For convenience, exact count
    plots are included to the side.

    Example usage::

        sim = EcoliSim.from_file()
        sim.build_ecoli()
        sim.run()
        data = sim.query()
        data = {key: val['agents']['0'] for key, val in data.items()}
        store_configs = sim.ecoli_experiment.get_config()
        bulk_ids = store_configs['agents']['0']['bulk']['_properties']['metadata']
        blame_timeseries(data, sim.topology, bulk_ids
                        ['WATER[c]', 'APORNAP-CPLX[c]', 'TRP[c]'],
                        'out/ecoli_master/test_blame_timeseries.png',
                        yscale="linear")

    Args:
        data: Data from an experiment (for experiments with cell
            division, ensure that ``bulk`` is a top-level field in the
            sub-dictionaries for each time point)
        topology: Experiment topology (used to determine which processes
            are connected to ``bulk`` and how)
        bulk_ids: List (or array) of bulk molecule names in the order
            they appear in the structured bulk Numpy array (see :ref:`bulk`).
            Typically retrieved from simulation config metadata.
        molecules: List of bulk molecule names to plot data for
        filename: Path to save plot to (optional)
        yscale: See :py:func:`matplotlib.pyplot.yscale`

    Returns:
        Axes and figure
    """

    if "log_update" not in data[0.0].keys():
        raise ValueError(
            "Missing log_update store in data; did you run simulation without logged updates?"
        )

    # Collect data into one dictionary
    # of the form: {process : {molecule : timeseries}}
    bulk_processes = get_bulk_processes(topology)
    time, processes, values_array = preprocess_data(
        data, bulk_ids, bulk_processes, molecules
    )
    bulk_ids = np.array(bulk_ids)

    # Twp subplots per molecule (count, change)
    max_t = time.max()
    fig, axs = plt.subplots(
        len(molecules),
        2,
        figsize=(10 + np.sqrt(max_t), 3 * len(molecules)),
        gridspec_kw={"width_ratios": [1, 10 + np.sqrt(max_t)]},
    )
    axs = np.atleast_2d(axs)
    for i, molecule in enumerate(molecules):
        # Plot molecule count over time
        # molecule_data = data['bulk'][molecule]
        molecule_idx = np.where(bulk_ids == molecule)[0][0]
        molecule_data = np.array(
            [timepoint["bulk"][molecule_idx] for timepoint in data.values()]
        )
        axs[i, 0].set_title(f"Count of {molecule}", pad=20)
        axs[i, 0].set_ylabel("# molecules")
        axs[i, 0].set_xlabel("Time (s)")
        axs[i, 0].set_xticks(time)
        axs[i, 0].plot(time, molecule_data)

        # Plot change due to each process
        axs[i, 1].set_title(f"Change in {molecule}", pad=20)
        axs[i, 1].set_ylabel("# molecules")
        axs[i, 1].set_xlabel("Time (s)")
        axs[i, 1].set_xticks(time[1:])
        axs[i, 1].axhline(y=0, color="k", linestyle="--", alpha=0.5)

        y = values_array[i, 1 : len(time), :]
        signed_stacked_bar(axs[i, 1], time[1:], y, processes)
        axs[i, 1].set_yscale(yscale)

    axs[0, 1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    # Sizing and spacing
    # fig.set_size_inches(4 + np.sqrt(max_t),  # include space for legend(s)
    #                     3 * len(molecules))  # height prop. to number of plots
    fig.tight_layout(pad=2.0)

    # Save plot to file
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)

    # Return axis and figure
    return plt.gca(), plt.gcf()


def test_blame_timeseries():
    # TODO:
    # - add back processes
    # - get working with unique molecules (separate argument for unique)
    # - serializers

    from vivarium.core.emitter import (
        data_from_database,
        get_local_client,
        timeseries_from_data,
    )

    EXPERIMENT_ID = None  # "d811d69e-0cf6-11ec-a1ab-00155df92294"

    if EXPERIMENT_ID:
        data, conf = data_from_database(
            EXPERIMENT_ID, get_local_client("localhost", "27017", "simulations")
        )
        data = timeseries_from_data(data)
        topo = conf["topology"]

    else:
        sim = EcoliSim.from_file()
        # CONFIG_DIR_PATH + "/test_configs/test_blame.json")
        # sim.emitter = "database"
        sim.raw_output = True
        sim.log_updates = True
        sim.emit_topology = False
        sim.emit_processes = False
        sim.max_duration = 10
        sim.build_ecoli()
        bulk_ids = sim.generated_initial_state["agents"]["0"]["bulk"]["id"]
        sim.run()
        data = sim.query()
        topo = sim.ecoli_experiment.topology["agents"]["0"]

    # molecules = [
    #     "EG10841-MONOMER",
    #     "EG10321-MONOMER",
    #     "EG11545-MONOMER",
    #     "EG11967-MONOMER",
    #     "FLAGELLAR-MOTOR-COMPLEX",
    #     "G361-MONOMER",
    #     "CPLX0-7451",
    #     "CPLX0-7452",  # Final flagella molecule
    # ]

    data = {time: time_data["agents"]["0"] for time, time_data in data.items()}

    blame_timeseries(
        data,
        topo,
        bulk_ids,
        ["CPD-12261[p]"],  # + molecules,
        "out/ecoli_master/murein_blame.png",
        yscale="linear",
    )


def plot(
    params: dict[str, Any],
    conn: DuckDBPyConnection,
    history_sql: str,
    config_sql: str,
    success_sql: str,
    sim_data_paths: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_name: str,
):
    # TODO: Write analysis script using DuckDB
    raise NotImplementedError("Still need to write analysis script using DuckDB!")


if __name__ == "__main__":
    test_blame()
