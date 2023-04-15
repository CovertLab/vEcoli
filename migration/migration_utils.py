import json
import copy
import numpy as np
from unum import Unum
import numbers
from scipy.stats import mannwhitneyu, chi2_contingency, ttest_ind, bartlett
from vivarium.core.engine import Engine, view_values
from vivarium.library.dict_utils import deep_merge

from wholecell.utils import units
from ecoli.states.wcecoli_state import get_state_from_file
from migration import LOAD_SIM_DATA

PERCENT_ERROR_THRESHOLD = 0.05
PVALUE_THRESHOLD = 0.05

def get_process_state(process, topology, initial_state):
    # make an experiment
    experiment_config = {
        'processes': {process.name: process},
        'topology': {process.name: topology},
        'initial_state': initial_state}
    experiment = Engine(**experiment_config)

    # Get update from process.
    path, process = list(experiment.process_paths.items())[0]
    store = experiment.state.get_path(path)

    # translate the values from the tree structure into the form
    # that this process expects, based on its declared topology
    topology_view = store.outer.schema_topology(process.schema, store.topology)
    states = view_values(topology_view)
    return states, experiment


def run_non_partitioned_process(
    process,
    topology,
    initial_time=0,
    initial_state=None,
):
    """
    Load a single ecoli process, run it, and return the update.
    NOTE: Only use for non-partitioned processes

    Args:
        process: an initialized Process
        topology: (dict) topology for the Process, from ecoli_master
        initial_time: (int) used to determine which partitioned count to load
        initial_state: (dict) used to initialize simulation

    Returns:
        an update from the perspective of the Process.
    """
    # make an experiment
    experiment_config = {
        'processes': {process.name: process},
        'topology': {process.name: topology},
        'initial_state': initial_state}
    experiment = Engine(**experiment_config)

    # Reset bulk counts to "partitioned counts" so final counts
    # can be directly compared to wcEcoli (some processes do not
    # connect to bulk store so skip it)
    try:
        bulk_array = experiment.state.get_path(('bulk',)).get_value()
        with open(f"data/migration/bulk_partitioned_t{initial_time}.json") as f:
            bulk_partitioned = json.load(f)
        proc_name = type(process).__name__
        if proc_name in bulk_partitioned:
            bulk_array['count'] = bulk_partitioned[proc_name]
        bulk_exists = True
    except:
        bulk_exists = False

    # Get update from process.
    path = list(experiment.process_paths.keys())[0]
    update, store = experiment._calculate_update(path, process, 2)
    update = update.get()

    # Leave non-bulk updates as is
    actual_update = {k: v for k, v in update.items() if k != 'bulk'}

    # Get final bulk counts to compare to wcEcoli
    if bulk_exists:
        experiment.apply_update(update, store)
        bulk_counts = experiment.state.get_path(('bulk',)).get_value()['count']
        if 'bulk' in update:
            actual_update['bulk'] = bulk_counts
    
    return actual_update


def run_partitioned_process(
    process,
    topology,
    initial_time=0,
    initial_state=None,
):
    """Given an initial state dictionary, run the calculate_request method of
    a process, then load a json of partitioned molecule counts to be merged
    into the process state before running the evolve_state method."""
    # Cache bulk_total count for later
    bulk_total = initial_state['bulk']['count'].copy()

    # make an experiment
    experiment_config = {
        'processes': {process.name: process},
        'topology': {process.name: topology},
        'initial_state': initial_state}
    experiment = Engine(**experiment_config)

    # Get process path
    path = list(experiment.process_paths.keys())[0]
    # Test calculate_request
    process.request_only = True
    process.evolve_only = False
    requests, store = experiment._calculate_update(path, process, 2)
    bulk_array = experiment.state.get_path(('bulk',)).get_value()
    # Set bulk counts to 0 for requested count after applying update
    bulk_array['count'] = 0
    experiment.apply_update(requests.get(), store)
    actual_requests = bulk_array['count'].copy()

    # Test evolve_state
    process.request_only = False
    process.evolve_only = True
    store, states = experiment._process_state(path)
    # Make process see wcEcoli partitioned molecule counts but
    # do not modify bulk_total port values
    states['bulk_total'] = bulk_total
    with open(f"data/migration/bulk_partitioned_t"
              f"{initial_time}.json") as f:
        bulk_partitioned = json.load(f)
    bulk_array['count'] = bulk_partitioned[type(process).__name__]
    update, store = experiment._process_update(
        path, process, store, states, 2)
    update = update.get()

    # Leave non-bulk updates as is
    actual_update = {k: v for k, v in update.items() if k != 'bulk'}

    # Get final bulk counts to compare to wcEcoli
    experiment.apply_update(update, store)
    bulk_counts = experiment.state.get_path(('bulk',)).get_value()['count']
    if 'bulk' in update:
        actual_update['bulk'] = bulk_counts

    return actual_requests, actual_update


def run_and_compare(init_time, process_class, partition=True, layer=0, post=False):
    # Set time parameters
    init_time = init_time

    # Create process, experiment, loading in initial state from file.
    config = LOAD_SIM_DATA.get_config_by_name(process_class.name)
    config['seed'] = 0
    process = process_class(config)

    process.is_step = lambda: False
    
    if post:
        initial_state = get_state_from_file(
            path=f'data/migration/wcecoli_t{init_time}_before_post.json')
    else:
        initial_state = get_state_from_file(
            path=f'data/migration/wcecoli_t{init_time}_before_layer_{layer}.json')
    initial_state['deriver_skips'] = {
        'chromosome_structure': False,
        'metabolism': False,
    }

    # Complexation sets seed weirdly
    if process_class.__name__ == 'Complexation':
        from arrow import StochasticSystem
        process.system = StochasticSystem(process.stoichiometry, random_seed=0)
    # Polypeptide initiation requires effective_elongation_rate listener
    elif process_class.__name__ == 'PolypeptideInitiation':
        listener_file = f'data/migration/wcecoli_listeners_t{init_time}.json'
        with open(listener_file, 'r') as f:
            listeners = json.load(f)
        elong_rate = listeners['ribosome_data']['effective_elongation_rate']
        initial_state['listeners']['ribosome_data'] = {
            'effective_elongation_rate': elong_rate}
    # Metabolism requires gtp_to_hydrolyze
    elif process_class.__name__ == 'Metabolism':
        updates_file = f'data/migration/process_updates_t{init_time}.json'
        with open(updates_file, 'r') as f:
            proc_updates = json.load(f)
        gtp_hydro = proc_updates['PolypeptideElongation']['gtp_to_hydrolyze']
        initial_state['process_state'] = {'polypeptide_elongation': {
            'gtp_to_hydrolyze': gtp_hydro}}

    if partition:
        # run the process and get an update
        actual_request, actual_update = run_partitioned_process(
            process, process_class.topology,
            initial_time=init_time, initial_state=initial_state)

        # Compare requested bulk counts
        with open(f"data/migration/bulk_requested_t{init_time}.json", 'r') as f:
            wc_request = json.load(f)
        assert np.all(actual_request == wc_request[process_class.__name__])
    else:
        actual_update = run_non_partitioned_process(
            process, process_class.topology,
            initial_time=init_time, initial_state=initial_state)
    # Compare unique molecule updates
    with open(f"data/migration/process_updates_t{init_time}.json", 'r') as f:
        wc_update = json.load(f)
    if process_class.__name__ in wc_update:
        # wcEcoli and viv-ecoli generate unique indices very differently
        assert recursive_compare(actual_update, wc_update[process_class.__name__],
            check_keys_strict=False, ignore_keys={'unique_index', 'RNAP_index'})
    # Compare listener values
    with open(f"data/migration/wcecoli_listeners_t{init_time}.json", 'r') as f:
        wc_listeners = json.load(f)
    actual_listeners = actual_update.get('listeners', {})
    # Certain listeners are at higher level in viv-ecoli
    if 'monomer_counts' in actual_listeners:
        actual_listeners = {'monomer_counts': actual_listeners}
    if 'mRNA_counts' in actual_listeners:
        actual_listeners = {'mrna_counts': actual_listeners}
    # Gene copy number determined by listener after everything updates in wcEcoli
    # TODO: Move this out of TfBinding process
    assert recursive_compare(actual_listeners, wc_listeners,
        check_keys_strict=False, ignore_keys={'growth', 'gene_copy_number'})


def recursive_compare(d1, d2, level='root', include_callable=False, 
                      ignore_keys=set(), check_keys_strict=True):
    """Recursively compare 2 dictionaries, printing any differences
    and their paths. Does not compare callable values by default.
    Can exclude specific dictionary keys from comparison or prevent
    failure when keys are missing."""
    is_equal = True
    if isinstance(d1, dict) and isinstance(d2, dict):
        if d1.keys() != d2.keys():
            s1 = set(d1.keys())
            s2 = set(d2.keys())
            print('{:<20} + {} - {}'.format(level, s1-s2, s2-s1))
            common_keys = s1 & s2 - ignore_keys
            if check_keys_strict:
                if s1-s2-ignore_keys or s2-s1-ignore_keys:
                    is_equal = False
        else:
            common_keys = set(d1.keys()) - ignore_keys

        for k in common_keys:
            is_equal = recursive_compare(d1[k], d2[k], 
                level='{}.{}'.format(level, k), 
                include_callable=include_callable, 
                ignore_keys=ignore_keys,
                check_keys_strict=check_keys_strict) and is_equal

    elif isinstance(d1, list) and isinstance(d2, list):
        if len(d1) != len(d2):
            print('{:<20} len1={}; len2={}'.format(level, len(d1), len(d2)))
            is_equal = False
        common_len = min(len(d1), len(d2))

        for i in range(common_len):
            is_equal = recursive_compare(d1[i], d2[i], 
                level='{}[{}]'.format(level, i),
                include_callable=include_callable, 
                ignore_keys=ignore_keys,
                check_keys_strict=check_keys_strict) and is_equal
            
    elif isinstance(d1, np.ndarray) or isinstance(d2, np.ndarray):
        if not np.array_equal(d1, d2):
            print('{:<20} {} != {}'.format(level, d1, d2))
            return False
    
    elif isinstance(d1, Unum) and isinstance(d2, Unum):
        return recursive_compare(d1.asNumber(), d2.asNumber(), 
            level='{}'.format(level), 
            include_callable=include_callable, 
            ignore_keys=ignore_keys,
            check_keys_strict=check_keys_strict)
    
    elif isinstance(d1, numbers.Number) and isinstance(d2, numbers.Number):
        # Floats are precise to 15 digits
        if not np.isclose(d1, d2, equal_nan=True, rtol=1e-15):
            print('{:<20} {} != {}'.format(level, d1, d2))
            return False
            
    elif isinstance(d1, set) and isinstance(d2, set):
        if len(d1 - d2) != 0:
            print('{:<20} {} in 1st but not 2nd set'.format(level, d1 - d2))
            return False
    
    elif callable(d1) and callable(d2) and not include_callable:
        return
    
    else:
        try:
            if str(d1) != str(d2):
                print('{:<20} {} != {}'.format(level, d1, d2))
                return False
        except ValueError:
            print('{:<20} {} != {}'.format(level, d1, d2))
            return False
    return is_equal


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

    diffs = np.abs(a - b)
    r_diffs = diffs / np.maximum(np.abs(a), np.abs(b))
    r_diffs[np.isnan(r_diffs)] = 0

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
    denominator = max(abs(actual), abs(expected))
    if denominator > 0:
        pe = np.divide(abs(actual - expected), denominator)
    else:
        pe = 0
    return pe

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
    2. (optionally) returning additional diagnostic information, as the
       second element of a tuple.

    For example:

        def scalar_equal(v1, v2):
            return v1==v2, f"Difference (actual-expected) is {v1-v2}"

    Leaf nodes can also be lists of such test functions, in which case all the specified
    tests will be run.

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
            indent = "  " * level
            if callable(test_structure):
                if verbose:
                    print(f'{indent}\n'
                          f'{indent}Running test\n'
                          f'{indent}\t{test_structure.__name__}')
                test_result = test_structure(vivEcoli_update, wcEcoli_update)
                report = {test_structure.__name__: test_result}

                passed = test_result[0] if isinstance(test_result, tuple) else test_result
                if verbose:
                    print(f'{indent}{"PASSED" if passed else "FAILED"} test "{test_structure.__name__}":')
                    if isinstance(test_result, tuple):
                        print(f'{indent}\t{test_result[1]}')

                if not passed:
                    self.failure = True
                    if self.fail_loudly:
                        assert passed, f'{indent}FAILED test "{test_structure.__name__}"'
            elif isinstance(test_structure, list):
                report = [run_tests_iter(vivEcoli_update,
                                         wcEcoli_update,
                                         test,
                                         verbose,
                                         level + 1)
                          for test in test_structure]
            else: # assumed to be dictionary
                for k, v in test_structure.items():
                    if verbose:
                        print(f'{indent}Testing {k}...')

                    # check structure is correct:
                    if k not in vivEcoli_update:
                        raise ValueError(f"Key {k} is missing from vivarium-ecoli update.")
                    if k not in wcEcoli_update:
                        raise ValueError(f"Key {k} is missing from wcEcoli update.")

                    report[k] = run_tests_iter(vivEcoli_update[k],
                                               wcEcoli_update[k],
                                               test_structure[k],
                                               verbose,
                                               level + 1)

            return report

        self.report = run_tests_iter(vivEcoli_update, wcEcoli_update, self.test_structure, verbose)

    def dump_report(self):

        def dump_report_iter(test_structure, report, level=0):
            indent = '  '*level

            if isinstance(test_structure, dict):
                for k, v in test_structure.items():
                    print(f'{indent}Tests for {k}:')
                    dump_report_iter(v, report[k], level + 1)
            elif isinstance(test_structure, list):
                for test, result in zip(test_structure, report):
                    dump_report_iter(test, result, level)
            else:
                for k, v in report.items():
                    passed = v[0] if isinstance(v, tuple) else v
                    print(f'{indent}{"PASSED" if passed else "FAILED"} test "{k}":')

                    if isinstance(v, tuple):
                        print(f'{indent}\t{v[1]}')

                    dump_report_iter.n_passed += passed
                    dump_report_iter.n_tests += 1

        dump_report_iter.n_passed = 0
        dump_report_iter.n_tests = 0
        dump_report_iter(self.test_structure, self.report)
        print(f"Passed {dump_report_iter.n_passed}/{dump_report_iter.n_tests} tests.")

    def fail(self):
        assert not self.failure, "Failed one or more tests."


# Common tests for use with ComparisonTestSuite ========================================================================
def equal(a, b):
    return np.all(a==b)

def equal_len(a, b):
    return len(a)==len(b)

def len_almost_equal(a, b):
    return np.isclose(len(a), len(b), rtol=0.05, atol=1)

def array_equal(arr1, arr2):
    return (np.array_equal(arr1, arr2, equal_nan=True),
            f"Total difference (actual-expected) is {np.sum(np.abs(arr1-arr2))}")

def scalar_equal(v1, v2):
    return v1 == v2, f"Difference (actual-expected) is {v1-v2}"

def array_almost_equal(arr1, arr2):
    p_errors = np.abs(arr1 - arr2) / np.maximum(np.abs(arr1), np.abs(arr2))
    p_errors[np.isnan(p_errors)] = 0

    return (np.all(p_errors <= PERCENT_ERROR_THRESHOLD),
            f"Max error = {np.max(p_errors):.4f}")

def scalar_almost_equal(v1, v2, custom_threshold=None):
    pe = percent_error(v1, v2)
    if custom_threshold:
        return pe < custom_threshold or np.isnan(pe), f"Percent error = {pe:.4f}"
    return pe < PERCENT_ERROR_THRESHOLD or np.isnan(pe), f"Percent error = {pe:.4f}"

def custom_array_comp(percent_error_threshold = 0.05):
    def _array_almost_equal(arr1, arr2):
        p_errors = np.abs(arr1 - arr2) / np.maximum(np.abs(arr1), np.abs(arr2))
        p_errors[np.isnan(p_errors)] = 0

        return (np.all(p_errors <= percent_error_threshold),
                f"Max error = {np.max(p_errors):.4f}")

    return _array_almost_equal

def custom_scalar_comp(percent_error_threshold = 0.05):
    def _scalar_almost_equal(v1, v2):
        pe = percent_error(v1, v2)
        return pe < PERCENT_ERROR_THRESHOLD or np.isnan(pe), f"Percent error = {pe:.4f}"

    return _scalar_almost_equal

def good_fit(dist1, dist2):
    chi2, p, _, _ = chi2_contingency([dist1, dist2])
    return p > PVALUE_THRESHOLD, f"Chi^2 test, X^2 = {chi2:.4f}, p = {p:.4f}"

def same_means(dist1, dist2):
    var_result = bartlett(dist1, dist2)
    same_var = var_result.pvalue > PVALUE_THRESHOLD
    result = ttest_ind(dist1, dist2, equal_var=same_var)
    return (result.pvalue > PVALUE_THRESHOLD,
            f"Two-sample T-test, t = {result.statistic:.4f}, p = {result.pvalue:.4f}"
            f" (Assuming {'' if same_var else 'un'}equal variances based on Bartlett test, p={var_result.pvalue:.4f})")

def stochastic_equal(dist1, dist2):
    u, p = mannwhitneyu(dist1, dist2)
    return p > PVALUE_THRESHOLD, f"Mann-Whitney U, U = {u}, p = {p:.4f}"

def stochastic_equal_array(arr1, arr2):
    u_stats, p_vals = mannwhitneyu(arr1, arr2)
    return p_vals > PVALUE_THRESHOLD, f"Mann-Whitney U, (U,p) = {list(zip(u_stats, [round(p, 4) for p in p_vals]))}"

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

def some_of(*tests):
    '''
    For composing tests. Equivalent to asserting that at least one of the specified tests passes.

    Args:
        *tests: the test functions to use.

    Returns: a test function, returning a tuple of
             1. boolean representing whether at least one test passed, followed by
             2. a list of all the diagnostic information from all tests.
    '''

    def _some_of(arr1, arr2):
        result = []
        for test in tests:
            result.append(test(arr1, arr2))

        result = (all([x[0] if isinstance(x, tuple) else x for x in result]),
                  [x[1] if isinstance(x, tuple) else None for x in result])

        return result

    return _some_of

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
