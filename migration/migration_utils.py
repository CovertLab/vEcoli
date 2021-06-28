import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency, ttest_ind
from vivarium.core.engine import Engine

from ecoli.composites.ecoli_master import get_state_from_file

PERCENT_ERROR_THRESHOLD = 0.05
PVALUE_THRESHOLD = 0.05

def get_process_state(process, topology, initial_state):
    # make an experiment
    experiment_config = {
        'processes': {process.name: process},
        'topology': {process.name: topology},
        'initial_state': initial_state}
    experiment = Engine(experiment_config)

    # Get update from process.
    path, process = list(experiment.process_paths.items())[0]
    store = experiment.state.get_path(path)

    # translate the values from the tree structure into the form
    # that this process expects, based on its declared topology
    states = store.outer.schema_topology(process.schema, store.topology)
    return states, experiment


def run_ecoli_process(
        process,
        topology,
        total_time=2,
        initial_time=0,
        initial_state=None,
):
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
    experiment = Engine(experiment_config)

    # Get update from process.
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

    1. returning False if they are different (True otherwise), and
    2. (optionally) returning additional diagnostic information as the
       second element of a tuple.

    For example:

        def scalar_equal(v1, v2):
            return v1==v2, f"Difference (actual-expected) is {v1-v2}"

    Creating a ComparisonTestSuite with fail_loudly = True causes it to stop testing
    as soon as it encounters the first error. Otherwise, failures occur silently.

    Results of testing can be viewed using ComparisonTestSuite.dump_report(), or
    by examining ComparisonTestSuite.report manually.

    To raise an AssertionError if necessary after running tests with fail_loudly=False,
    use ComparisonTestSuite.fail().

    I recommend to use fail_loudly=False, verbose=True while writing tests for ease of debugging,
    and fail_loudly=True, verbose=False once tests are passing, for efficiency.
    '''

    def __init__(self, test_structure, fail_loudly=False):
        self.test_structure = test_structure
        self.fail_loudly = fail_loudly
        self.report = {}
        self.failure = False

    def run_tests(self, vivEcoli_update, wcEcoli_update, verbose=False):
        self.report = {}

        def run_tests_iter(vivEcoli_update, wcEcoli_update, test_structure,
                           verbose, level=0):
            report = {}
            for k, v in test_structure.items():
                indent = "." * level
                if verbose:
                    print(f'{indent}Testing {k}...')

                # check structure is correct:
                if k not in vivEcoli_update:
                    raise ValueError(f"Key {k} is missing from vivarium-ecoli update.")
                if k not in wcEcoli_update:
                    raise ValueError(f"Key {k} is missing from wcEcoli update.")

                # run test if at a leaf node, else recurse
                if callable(v):
                    print(
                        f'{indent}\n'
                        f'{indent}Running test\n'
                        f'{indent}\t{v.__name__}\n'
                        f'{indent}for element {k}.')
                    test_result = v(vivEcoli_update[k], wcEcoli_update[k])
                    report[k] = test_result

                    passed = test_result[0] if isinstance(test_result, tuple) else test_result
                    if verbose:
                        print(f'{indent}{"PASSED" if passed else "FAILED"} test "{v.__name__}" for {k}:')
                        if isinstance(test_result, tuple):
                            print(f'{indent}\t{test_result[1]}')

                    if not passed:
                        self.failure = True
                        if self.fail_loudly:
                            assert passed, f'{indent}FAILED test "{v.__name__}" for {k}.'

                    print(indent)
                else:
                    report[k] = run_tests_iter(vivEcoli_update[k],
                                               wcEcoli_update[k],
                                               test_structure[k],
                                               verbose,
                                               level+1)
            return report

        self.report = run_tests_iter(vivEcoli_update, wcEcoli_update, self.test_structure, verbose)

    def dump_report(self):
        def dump_report_iter(test_structure, report):
            n_passed = 0
            n_tests = 0

            for k, v in report.items():
                if not isinstance(v, dict):
                    passed = v[0] if isinstance(v, tuple) else v
                    print(f'{"PASSED" if passed else "FAILED"} test for {k}:')
                    if isinstance(v, tuple):
                        print(f'\t{v[1]}')

                    n_passed += passed
                    n_tests += 1
                else:
                    p, t = dump_report_iter(test_structure[k], report[k])
                    n_passed += p
                    n_tests += t

            return (n_passed, n_tests)

        n_passed, n_tests = dump_report_iter(self.test_structure, self.report)
        print(f"Passed {n_passed}/{n_tests} tests.")


    def fail(self):
        assert not self.failure, "Failed one or more tests."


# Common tests for use with ComparisonTestSuite ========================================================================

def array_equal(arr1, arr2):
    return (np.array_equal(arr1, arr2, equal_nan=True),
            f"Total difference (actual-expected) is {np.sum(np.abs(arr1-arr2))}")

def scalar_equal(v1, v2):
    return v1 == v2, f"Difference (actual-expected) is {v1-v2}"

def array_almost_equal(arr1, arr2):
    p_errors = np.abs(arr1 - arr2) / np.abs(arr2)
    p_errors[np.isnan(p_errors)] = 0

    return (np.all(p_errors <= PERCENT_ERROR_THRESHOLD),
            f"Max error = {np.max(p_errors):.4f}")

def scalar_almost_equal(v1, v2):
    return percent_error(v1, v2) < PERCENT_ERROR_THRESHOLD or np.isnan(pe), f"Percent error = {percent_error(v1, v2):.4f}"

def custom_array_comp(percent_error_threshold = 0.05):
    def _array_almost_equal(arr1, arr2):
        p_errors = np.abs(arr1 - arr2) / np.abs(arr2)
        p_errors[np.isnan(p_errors)] = 0

        return (np.all(p_errors <= percent_error_threshold),
                f"Max error = {np.max(p_errors):.4f}")

    return _array_almost_equal

def custom_scalar_comp(percent_error_threshold = 0.05):
    def _array_almost_equal(arr1, arr2):
        pe = np.sum(np.abs(arr1 - arr2) / arr2)
        return pe < percent_error_threshold, f"Percent error = {pe:.4f}"

    return _array_almost_equal

def good_fit(dist1, dist2):
    chi2, p, _, _ = chi2_contingency([dist1, dist2])
    return p > PVALUE_THRESHOLD, f"Chi^2 test, X^2 = {chi2:.4f}, p = {p:.4f}"

def same_means(dist1, dist2):
    result = ttest_ind(dist1, dist2, equal_var=False)
    return (result.pvalue > PVALUE_THRESHOLD,
            f"Two-sample T-test, t = {result.statistic:.4f}, p = {result.pvalue:.4f}")

def stochastic_equal(dist1, dist2):
    u, p = mannwhitneyu(dist1, dist2)
    return p > PVALUE_THRESHOLD, f"Mann-Whitney U, U = {u}, p = {p:.4f}"

def array_diffs_report_test(filename, names=None, sort_by="absolute", sort_with=np.abs):
    def _array_diffs_report_test(a, b):
        result = array_diffs_report(a, b, names, sort_by, sort_with)
        with open(filename, 'w') as f:
            f.write(result)
        return True
    return _array_diffs_report_test

# Common transformations ===============================================================================================

def pseudocount(arr):
    return arr + 1

# Test composition and modification ====================================================================================

def run_all(*tests):
    '''
    Function for composing tests. Passes if all input tests pass.

    Args:
        *tests: the tests to be run.

    Returns: A test function that returns a tuple of
             1. a boolean for whether all tests passed, and
             2. a list of all diagnostic information for the tests.

    '''
    def _run_all(arr1, arr2):
        result = []
        for test in tests:
            result.append(test(arr1, arr2))

        result = (all([x[0] if isinstance(x, tuple) else x for x in result]),
                  [x[1] if isinstance(x, tuple) else "" for x in result])

        return result

    return _run_all

def one_of(*tests):
    '''
    For composing tests. Equivalent to asserting that at least one of the specified tests passes.

    Args:
        *tests: the test functions to use.

    Returns: a test function, returning a tuple of
             1. boolean representing whether at least one test passed, followed by
             2. a list of all the diagnostic information from all tests.
    '''

    def _one_of(arr1, arr2):
        result = []
        for test in tests:
            result.append(test(arr1, arr2))

        result = (all([x[0] if isinstance(x, tuple) else x for x in result]),
                  [x[1] if isinstance(x, tuple) else None for x in result])

        return result

    return _one_of

def transform_and_run(transformation, test, transform_first=True, transform_second=True):
    '''
    Allows preprocessing of elements prior to testing. transformation(val) does the transformation
    prior to running the specified test.

    Args:
        transformation: list of functions to preprocess each element to be compared
        test: the test to run on transformed elements

    Returns: A test function.
    '''

    def _transform_and_run(v1, v2):
        if transform_first:
            v1 = transformation(v1)

        if transform_second:
            v2 = transformation(v2)

        return test(v1, v2)

    return _transform_and_run
