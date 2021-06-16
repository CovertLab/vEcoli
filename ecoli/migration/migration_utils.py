import numpy as np
from vivarium.core.experiment import Experiment

from ecoli.composites.ecoli_master import get_state_from_file


def run_ecoli_process(
        process,
        topology,
        total_time=2,
        initial_time=0,
        initial_state=None):
    """
    load a single ecoli process, run it, and return the update

    Args:
        process: an initialized Process
        topology: (dict) topology for the Process, from ecoli_master
        total_time: (optional) run time. defaults at 2 seconds -- the default time of wcEcoli

    Returns:
        an update from the perspective of the Process.
    """
    if not initial_state:
        # get initial state from file
        initial_state = get_state_from_file(
            path=f'data/wcecoli_t{initial_time}.json')

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


def array_diffs_report(a, b, names=None, sort_by="absolute", sort_with=np.abs):
    """
    Returns a report of the differences between numpy arrays a and b, including both
    absolute and relative differences.

    Args:
        a: First array.
        b: Second array.
        names: Optional array of names for the entries in a/b. If not provided, the diffs will simply be numbered.
        sort_by: Which values to sort the the differences by, if any. Can be "absolute", "relative", or None.
        sort_with: Function to apply to diffs prior to sorting. For example,
                   the default of np.abs can be substituted with the identity function if negative
                   diffs should be considered less than positive diffs.

    Returns: String representing a report of the differences between a and b.
    """

    if len(a) != len(b):
        raise ValueError(f"Length of a does not match length of b ({len(a)} != {len(b)})")

    if names is None:
        names = list(map(str, range(len(a))))

    if len(names) != len(a):
        raise ValueError(f"Length of names does not match length of a ({len(names)} != {len(a)})")

    diffs = a - b
    r_diffs = diffs / b

    if sort_by is not None:
        if sort_by == "absolute":
            order = np.argsort(-sort_with(diffs))
        elif sort_by == "relative":
            order = np.argsort(-sort_with(r_diffs))

    diffs = diffs[order]
    r_diffs = r_diffs[order]
    names = [names[i] for i in order]

    headings = ["Name", "Diff", "Relative Diff"]
    name_pad = max(len(headings[0]), max(map(len, names))) + 1
    diffs_pad = max(len(headings[1]), max(map(lambda x : len(str(x)), diffs))) + 1
    r_diffs_pad = max(len(headings[2]), max(map(lambda x : len(str(x)), r_diffs))) + 1
    paddings = [name_pad, diffs_pad, r_diffs_pad]

    result = f"{np.nonzero(diffs)[0].shape[0]} / {len(diffs)} entries have differences.\n"
    result += f"Maximum absolute difference is {np.max(diffs)}.\n"
    result += f"Maximum relative difference is {np.max(r_diffs[~np.isnan(r_diffs)])}.\n\n"

    result += ' | '.join(map(lambda t: t[0].center(t[1]),
                             zip(headings, paddings))) + '\n'
    result += '=' * (sum(paddings) + 2*len(' | ')) + '\n'
    for name, diff, r_diff in zip(names, diffs, r_diffs):
        result += (str(name).ljust(name_pad) + ' : ' +
                   str(diff).center(diffs_pad) + ' : ' +
                   str(r_diff).rjust(r_diffs_pad) + '\n')

    return result



def percent_error(actual, expected):
    return abs((actual - expected) / expected)


class ComparisonTestSuite:
    '''
    Class for running comparison tests in a consistent and reproducible way.
    The key function here is

        ComparisonTestSuite.run_tests(vivEcoli_update, wcEcoli_update, ...)

    which expects the two updates to compare in dictionary form, with the same structure.

    To create a ComparisonTestSuite, first define your expectations in a test structure.
    This should be a dictionary of the same structure as the two updates to be compared,
    but with the leaf nodes replaced with a (two-input) "test function" that compares
    the corresponding elements of the two updates by

    1. returning False if they are different (something Truth-y otherwise), and
    2. (optionally) returning additional diagnostic information.

    For example:

        def scalar_equal(v1, v2):
            return v1==v2, f"Difference (actual-expected) is {v1-v2}"

    Creating a ComparisonTestSuite with fail_loudly = True causes it to stop testing
    as soon as it encounters the first error. Otherwise, failures occur silently.

    Results of testing can be viewed using ComparisonTestSuite.dump_report(), or
    by examining ComparisonTestSuite.report manually.

    To raise an AssertionError after running
    '''

    def __init__(self, test_structure, fail_loudly=False):
        self.test_structure = test_structure
        self.fail_loudly = fail_loudly
        self.report = {}

    def run_tests(self, vivEcoli_update, wcEcoli_update, verbose=False):
        failures

        def run_tests_iter(vivEcoli_update, wcEcoli_update, test_structure,
                           verbose, level=0):
            for k, v in test_structure.items():
                if verbose:
                    print('.'*level + f'Testing {k}...')

                # check structure is correct:
                if k not in vivEcoli_update:
                    raise ValueError(f"Key {k} is missing from vivarium-ecoli update.")
                if k not in wcEcoli_update:
                    raise ValueError(f"Key {k} is missing from wcEcoli update.")

                # run test if at a leaf node, else recurse
                if callable(v):
                    try:
                        print('.'*level + 'Running test\n' +
                              '.'*level + f'\t{v}\n' +
                              '.'*level + 'for element {k}.')
                        test_result = v(vivEcoli_update[k], wcEcoli_update[k])
                        #self.report[k] = test_result if test_result is not None else True
                    except AssertionError as e:
                        self.failure = e
                        if self.fail_loudly:
                            raise e
                        if verbose:
                            print(e)

                        #self.report[k]=False
                else:
                    run_tests_iter(vivEcoli_update[k],
                                   wcEcoli_update[k],
                                   test_structure[k],
                                   verbose,
                                   level+1)

        run_tests_iter(vivEcoli_update, wcEcoli_update, self.test_structure, verbose)


def array_equal(arr1, arr2):
    return np.array_equal(arr1, arr2, equal_nan=True)

def scalar_equal(v1, v2):
    return v1==v2, f"Difference (actual-expected) is {v1-v2}"

def run_all(*tests):
    def _run_all(arr1, arr2):
        result = []
        for test in tests:
            result.append(test(arr1, arr2))

        result = (all([x[0] if isinstance(x, tuple) else x for x in result]),
                  [x[1] if isinstance(x, tuple) else "" for x in result])

        return result

    return _run_all
