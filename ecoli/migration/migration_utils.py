from vivarium.core.experiment import Experiment

from ecoli.composites.ecoli_master import get_state_from_file


def run_ecoli_process(process, topology, total_time=2):
    """
    load a single ecoli process, run it, and return the update

    Args:
        process: an initialized Process
        topology: (dict) topology for the Process, from ecoli_master
        total_time: (optional) run time. defaults at 2 seconds -- the default time of wcEcoli

    Returns:
        an update from the perspective of the Process.
    """

    # get initial state from file
    # TODO -- get wcecoli_t0
    initial_state = get_state_from_file(path='data/wcecoli_t10.json')

    # make an experiment
    experiment_config = {
        'processes': {process.name: process},
        'topology': {process.name: topology},
        'initial_state': initial_state}
    experiment = Experiment(experiment_config)

    # Get update from process.
    # METHOD 1
    path, process = list(experiment.process_paths.items())[0]
    store = experiment.state.get_path(path)

    # translate the values from the tree structure into the form
    # that this process expects, based on its declared topology
    states = store.outer.schema_topology(process.schema, store.topology)

    update = experiment.invoke_process(
        process,
        path,
        total_time,
        states)

    actual_update = update.get()
    return actual_update


def readout_diffs(a, b, names=None):
    if len(a) != len(b):
        raise ValueError(f"Length of a does not match length of b ({len(a)} != {len(b)})")

    if names is None:
        names = list(map(str, range(len(a))))

    if len(names) != len(a):
        raise ValueError(f"Length of names does not match length of a ({len(names)} != {len(a)})")

    diffs = a - b

    name_pad = max(4, max(map(len, names))) + 1
    diffs_pad = max(4, max(map(lambda x : len(str(x)), diffs))) + 1
    result = "Name".center(name_pad) + ' | ' + "Diff".center(diffs_pad) + '\n'
    result += '=' * (name_pad + diffs_pad + 3) + '\n'
    for name, diff in zip(names, diffs):
        result += str(name).ljust(name_pad) + ' : ' + str(diff).rjust(diffs_pad) + '\n'

    return result



def percent_error(actual, expected):
    return abs((actual - expected) / expected)