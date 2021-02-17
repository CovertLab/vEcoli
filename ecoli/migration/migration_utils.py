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


def percent_error(actual, expected):
    return abs((actual - expected) / expected)