import json
import numpy as np
from unum import Unum
import numbers
import warnings
from scipy.stats import mannwhitneyu, chi2_contingency, ttest_ind, bartlett
from vivarium.core.engine import Engine, view_values, _process_update
from vivarium.library.dict_utils import deep_merge

from wholecell.utils import units
from ecoli.library.json_state import get_state_from_file
from migration import LOAD_SIM_DATA, LOAD_SIM_DATA_NO_OPERONS

PERCENT_ERROR_THRESHOLD = 0.05
PVALUE_THRESHOLD = 0.05


def get_process_state(process, topology, initial_state):
    # make an experiment
    experiment_config = {
        "processes": {process.name: process},
        "topology": {process.name: topology},
        "initial_state": initial_state,
    }
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
    process, topology, initial_time=0, initial_state=None, data_prefix="data/migration"
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
        "processes": {process.name: process},
        "topology": {process.name: topology},
        "initial_state": initial_state,
    }
    experiment = Engine(**experiment_config)

    # Get update from process.
    path = list(experiment.process_paths.keys())[0]
    store, states = experiment._process_state(path)

    # Reset bulk counts to "partitioned counts" so final counts
    # can be directly compared to wcEcoli (some processes do not
    # connect to bulk store so skip it)
    try:
        states["bulk_total"] = initial_state["bulk"]["count"].copy()
        with open(f"{data_prefix}/bulk_partitioned_t{initial_time}.json") as f:
            bulk_partitioned = json.load(f)
        states["bulk"].flags.writeable = True
        states["bulk"]["count"] = bulk_partitioned[type(process).__name__]
        states["bulk"].flags.writeable = False
        bulk_exists = True
    except KeyError:
        bulk_exists = False
    update, store = _process_update(path, process, store, states, 1)

    update = update.get()

    # Leave non-bulk updates as is
    actual_update = {k: v for k, v in update.items() if k != "bulk"}

    # Get final bulk counts to compare to wcEcoli
    if bulk_exists:
        experiment.apply_update(update, store)
        bulk_counts = experiment.state.get_path(("bulk",)).get_value()["count"]
        if "bulk" in update:
            actual_update["bulk"] = bulk_counts

    return actual_update


def run_partitioned_process(
    process, topology, initial_time=0, initial_state=None, data_prefix="data/migration"
):
    """Given an initial state dictionary, run the calculate_request method of
    a process, then load a json of partitioned molecule counts to be merged
    into the process state before running the evolve_state method."""
    # Cache bulk_total count for later
    bulk_total = initial_state["bulk"]["count"].copy()

    # make an experiment
    experiment_config = {
        "processes": {process.name: process},
        "topology": {process.name: topology},
        "initial_state": initial_state,
    }
    experiment = Engine(**experiment_config)

    # Get process path
    path = list(experiment.process_paths.keys())[0]
    # Test calculate_request
    process.request_only = True
    process.evolve_only = False
    requests, store = experiment._calculate_update(path, process, 1)
    bulk_array = experiment.state.get_path(("bulk",)).get_value()
    # Create copy of bulk array with floating point counts so
    # processes can return non-integer requests
    new_dtype = np.dtype(
        [("count", "float64")]
        + [(name, dtype) for name, dtype in bulk_array.dtype.descr if name != "count"]
    )
    float_bulk = np.empty(bulk_array.shape, dtype=new_dtype)
    for name in bulk_array.dtype.names:
        float_bulk[name] = (
            bulk_array[name] if name != "count" else bulk_array[name].astype(float)
        )
    # Leave non-bulk entries in request as is
    requests = requests.get()
    request_misc = {k: v for k, v in requests.items() if k != "bulk"}
    # Set bulk counts to 0 for requested count after applying update
    float_bulk["count"] = 0.0
    experiment.state["bulk"].value = float_bulk
    experiment.apply_update(requests, store)
    actual_requests = float_bulk["count"].copy()

    # Test evolve_state
    experiment.state["bulk"] = bulk_array
    process.request_only = False
    process.evolve_only = True
    store, states = experiment._process_state(path)
    # Make process see wcEcoli partitioned molecule counts but
    # do not modify bulk_total port values
    states["bulk_total"] = bulk_total
    with open(f"{data_prefix}/bulk_partitioned_t{initial_time}.json") as f:
        bulk_partitioned = json.load(f)
    states["bulk"].flags.writeable = True
    states["bulk"]["count"] = bulk_partitioned[type(process).__name__]
    states["bulk"].flags.writeable = False
    update, store = _process_update(path, process, store, states, 1)
    update = update.get()

    # Leave non-bulk updates as is
    actual_update = {k: v for k, v in update.items() if k != "bulk"}

    # Get final bulk counts to compare to wcEcoli
    experiment.apply_update(update, store)
    bulk_counts = experiment.state.get_path(("bulk",)).get_value()["count"]
    if "bulk" in update:
        actual_update["bulk"] = bulk_counts

    # Merge non-bulk requests with evolve_state update
    actual_update = deep_merge(request_misc, actual_update)

    return actual_requests, actual_update


def run_and_compare(
    init_time, process_class, partition=True, layer=0, post=False, operons=True
):
    # Set time parameters
    init_time = init_time

    # Create process, experiment, loading in initial state from file.
    if process_class.name == "replication_data_listener":
        config = {"time_step": 1}
    else:
        if operons:
            config = LOAD_SIM_DATA.get_config_by_name(process_class.name)
        else:
            config = LOAD_SIM_DATA_NO_OPERONS.get_config_by_name(process_class.name)
    config["seed"] = 0
    process = process_class(config)

    process.is_step = lambda: False

    if operons:
        data_prefix = "data/migration"
    else:
        data_prefix = "data/migration_no_operons"

    if post:
        initial_state = get_state_from_file(
            path=f"{data_prefix}/wcecoli_t{init_time}_before_post.json"
        )
        # By this point the clock process would have advanced the global time
        initial_state["global_time"] = init_time + 1
    else:
        initial_state = get_state_from_file(
            path=f"{data_prefix}/wcecoli_t{init_time}_before_layer_{layer}.json"
        )

    # Complexation sets seed weirdly
    if process_class.__name__ == "Complexation":
        from stochastic_arrow import StochasticSystem

        process.system = StochasticSystem(process.stoichiometry, random_seed=0)
    # Metabolism requires gtp_to_hydrolyze
    elif process_class.__name__ == "Metabolism":
        updates_file = f"{data_prefix}/process_updates_t{init_time}.json"
        with open(updates_file, "r") as f:
            proc_updates = json.load(f)
        gtp_hydro = proc_updates["PolypeptideElongation"]["process_state"][
            "gtp_to_hydrolyze"
        ]
        aa_exchange_rates = (
            units.mol
            / units.L
            / units.s
            * np.array(
                proc_updates["PolypeptideElongation"]["process_state"][
                    "aa_exchange_rates"
                ]
            )
        )
        initial_state["process_state"] = {
            "polypeptide_elongation": {
                "gtp_to_hydrolyze": gtp_hydro,
                "aa_exchange_rates": aa_exchange_rates,
                "aa_count_diff": proc_updates["PolypeptideElongation"]["process_state"][
                    "aa_count_diff"
                ],
            }
        }
        process.aa_targets = proc_updates["Metabolism"]["process_state"]["aa_targets"]
    # MassListener needs initial masses to calculate fold changes
    elif process_class.__name__ == "MassListener":
        t0_file = f"{data_prefix}/wcecoli_t0.json"
        with open(t0_file, "r") as f:
            t0_data = json.load(f)
        t0_mass = t0_data["listeners"]["mass"]
        process.first_time_step = False
        process.dryMassInitial = t0_mass["dry_mass"]
        process.proteinMassInitial = t0_mass["protein_mass"]
        process.rnaMassInitial = t0_mass["rna_mass"]
        process.smallMoleculeMassInitial = t0_mass["smallMolecule_mass"]
        process.timeInitial = 0
        process.match_wcecoli = True
    # Ribosome data listener requires transcription data
    elif process_class.__name__ == "RibosomeData":
        with open(f"{data_prefix}/wcecoli_listeners_t{init_time}.json", "r") as f:
            wc_listeners = json.load(f)
        initial_state["listeners"]["ribosome_data"]["rRNA_initiated_TU"] = wc_listeners[
            "ribosome_data"
        ]["rRNA_initiated_TU"]
        initial_state["listeners"]["ribosome_data"]["rRNA_init_prob_TU"] = wc_listeners[
            "ribosome_data"
        ]["rRNA_init_prob_TU"]
    # RNA synth prob listener requires transcription data
    elif process_class.__name__ == "RnaSynthProb":
        with open(f"{data_prefix}/wcecoli_listeners_t{init_time}.json", "r") as f:
            wc_listeners = json.load(f)
        initial_state["listeners"]["rna_synth_prob"] = {
            "total_rna_init": wc_listeners["rna_synth_prob"]["total_rna_init"],
            "n_bound_TF_per_TU": wc_listeners["rna_synth_prob"]["n_bound_TF_per_TU"],
            "actual_rna_synth_prob": wc_listeners["rna_synth_prob"][
                "actual_rna_synth_prob"
            ],
            "target_rna_synth_prob": wc_listeners["rna_synth_prob"][
                "target_rna_synth_prob"
            ],
        }
    # RNAP data listener requires transcription data
    elif process_class.__name__ == "RnapData":
        with open(f"{data_prefix}/wcecoli_listeners_t{init_time}.json", "r") as f:
            wc_listeners = json.load(f)
        initial_state["listeners"]["rnap_data"] = {
            "rna_init_event": wc_listeners["rnap_data"]["rna_init_event"]
        }

    if partition:
        # run the process and get an update
        actual_request, actual_update = run_partitioned_process(
            process,
            process_class.topology,
            initial_time=init_time,
            initial_state=initial_state,
            data_prefix=data_prefix,
        )

        # Compare requested bulk counts
        with open(f"{data_prefix}/bulk_requested_t{init_time}.json", "r") as f:
            wc_request = json.load(f)
        assert np.all(actual_request == wc_request[process_class.__name__])
    else:
        actual_update = run_non_partitioned_process(
            process,
            process_class.topology,
            initial_time=init_time,
            initial_state=initial_state,
            data_prefix=data_prefix,
        )
    if process_class.__name__ == "PolypeptideElongation":
        actual_update["process_state"] = actual_update["process_state"][
            "polypeptide_elongation"
        ]
        actual_update["process_state"]["aa_exchange_rates"] = actual_update[
            "process_state"
        ]["aa_exchange_rates"].asNumber()
    # Sort delete indices to match wcEcoli sorted indices
    for unique_update in actual_update.get("unique", {}).values():
        if "delete" in unique_update:
            unique_update["delete"] = np.sort(unique_update["delete"])
    # Compare updates
    with open(f"{data_prefix}/process_updates_t{init_time}.json", "r") as f:
        wc_update = json.load(f)
    if process_class.__name__ in wc_update:
        # wcEcoli and viv-ecoli generate unique indices very differently
        # Also do not compare metabolism exchanges because wcEcoli accumulates
        # those over entire runtime but migration test runs for one timestep
        assert recursive_compare(
            actual_update,
            wc_update[process_class.__name__],
            check_keys_strict=False,
            ignore_keys={"unique_index", "RNAP_index", "exchange"},
        )
    # Compare listener values
    with open(f"{data_prefix}/wcecoli_listeners_t{init_time}.json", "r") as f:
        wc_listeners = json.load(f)
    wc_listeners["unique_molecule_counts"] = dict(
        zip(
            sorted(
                LOAD_SIM_DATA.sim_data.internal_state.unique_molecule.unique_molecule_definitions
            ),
            wc_listeners["unique_molecule_counts"]["unique_molecule_counts"],
        )
    )
    actual_listeners = actual_update.get("listeners", {})
    # Certain listeners are at higher level in viv-ecoli
    if "monomer_counts" in actual_listeners:
        actual_listeners = {"monomer_counts": actual_listeners}
    if "mRNA_counts" in actual_listeners:
        actual_listeners = {"mrna_counts": actual_listeners}
    # Gene copy number determined by listener after everything updates in wcEcoli
    # Empty fork collisions are a new concept in vEcoli
    assert recursive_compare(
        actual_listeners,
        wc_listeners,
        check_keys_strict=False,
        ignore_keys={
            "growth",
            "gene_copy_number",
            "n_empty_fork_collisions",
            "empty_fork_collision_coordinates",
        },
    )


def recursive_compare(
    d1,
    d2,
    level="root",
    include_callable=False,
    ignore_keys=set(),
    check_keys_strict=True,
):
    """Recursively compare 2 dictionaries, printing any differences
    and their paths. Does not compare callable values by default.
    Can exclude specific dictionary keys from comparison or prevent
    failure when keys are missing."""
    is_equal = True
    if isinstance(d1, dict) and isinstance(d2, dict):
        if d1.keys() != d2.keys():
            s1 = set(d1.keys())
            s2 = set(d2.keys())
            print("{:<20} + {} - {}".format(level, s1 - s2, s2 - s1))
            common_keys = s1 & s2 - ignore_keys
            if check_keys_strict:
                if s1 - s2 - ignore_keys or s2 - s1 - ignore_keys:
                    is_equal = False
        else:
            common_keys = set(d1.keys()) - ignore_keys

        for k in common_keys:
            is_equal = (
                recursive_compare(
                    d1[k],
                    d2[k],
                    level="{}.{}".format(level, k),
                    include_callable=include_callable,
                    ignore_keys=ignore_keys,
                    check_keys_strict=check_keys_strict,
                )
                and is_equal
            )

    elif isinstance(d1, list) and isinstance(d2, list):
        if len(d1) != len(d2):
            print("{:<20} len1={}; len2={}".format(level, len(d1), len(d2)))
            is_equal = False
        common_len = min(len(d1), len(d2))

        for i in range(common_len):
            is_equal = (
                recursive_compare(
                    d1[i],
                    d2[i],
                    level="{}[{}]".format(level, i),
                    include_callable=include_callable,
                    ignore_keys=ignore_keys,
                    check_keys_strict=check_keys_strict,
                )
                and is_equal
            )

    elif isinstance(d1, np.ndarray) or isinstance(d2, np.ndarray):
        try:
            np.testing.assert_array_almost_equal_nulp(d1, d2)
        except AssertionError as e:
            warn_str = "{:<20} {} != {} {}".format(level, d1, d2, e)
            print(warn_str)
            warnings.warn(warn_str)
            return np.allclose(d1, d2, rtol=1e-7, atol=1e-12)
        except TypeError:
            if not np.array_equal(d1, d2):
                print("{:<20} {} != {}".format(level, d1, d2))
                return False

    elif isinstance(d1, Unum) and isinstance(d2, Unum):
        return recursive_compare(
            d1.asNumber(),
            d2.asNumber(),
            level="{}".format(level),
            include_callable=include_callable,
            ignore_keys=ignore_keys,
            check_keys_strict=check_keys_strict,
        )

    elif isinstance(d1, numbers.Number) and isinstance(d2, numbers.Number):
        try:
            np.testing.assert_array_almost_equal_nulp(d1, d2, 5)
        except AssertionError as e:
            if not (np.isnan(d1) & np.isnan(d2)):
                warn_str = "{:<20} {} != {} {}".format(level, d1, d2, e)
                print(warn_str)
                warnings.warn(warn_str)
                return np.isclose(d1, d2, rtol=1e-7, atol=1e-12)

    elif isinstance(d1, set) and isinstance(d2, set):
        if len(d1 - d2) != 0:
            print("{:<20} {} in 1st but not 2nd set".format(level, d1 - d2))
            return False

    elif callable(d1) and callable(d2) and not include_callable:
        return

    else:
        try:
            if str(d1) != str(d2):
                print("{:<20} {} != {}".format(level, d1, d2))
                return False
        except ValueError:
            print("{:<20} {} != {}".format(level, d1, d2))
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
        raise ValueError(
            f"Length of a does not match length of b ({len(a)} != {len(b)})"
        )

    if names is None:
        names = list(map(str, range(len(a))))

    if len(names) != len(a):
        raise ValueError(
            f"Length of names does not match length of a ({len(names)} != {len(a)})"
        )

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
    diffs_pad = max(len(headings[1]), max(map(lambda x: len(str(x)), diffs))) + 1
    r_diffs_pad = max(len(headings[2]), max(map(lambda x: len(str(x)), r_diffs))) + 1
    paddings = [name_pad, diffs_pad, r_diffs_pad]

    result = (
        f"{np.nonzero(diffs)[0].shape[0]} / {len(diffs)} entries have differences.\n"
    )
    result += f"Maximum absolute difference is {np.max(diffs)}.\n"
    result += (
        f"Maximum relative difference is {np.max(r_diffs[~np.isnan(r_diffs)])}.\n\n"
    )

    result += (
        " | ".join(map(lambda t: t[0].center(t[1]), zip(headings, paddings))) + "\n"
    )
    result += "=" * (sum(paddings) + 2 * len(" | ")) + "\n"
    for name, diff, r_diff in zip(names, diffs, r_diffs):
        result += (
            str(name).ljust(name_pad)
            + " : "
            + str(diff).center(diffs_pad)
            + " : "
            + str(r_diff).rjust(r_diffs_pad)
            + "\n"
        )

    return result


def percent_error(actual, expected):
    denominator = max(abs(actual), abs(expected))
    if denominator > 0:
        pe = np.divide(abs(actual - expected), denominator)
    else:
        pe = 0
    return pe


class ComparisonTestSuite:
    """
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
    """

    def __init__(self, test_structure, fail_loudly=False):
        self.test_structure = test_structure
        self.fail_loudly = fail_loudly
        self.report = {}
        self.failure = False

    def run_tests(self, vivEcoli_update, wcEcoli_update, verbose=False):
        self.report = {}

        def run_tests_iter(
            vivEcoli_update, wcEcoli_update, test_structure, verbose, level=0
        ):
            report = {}
            indent = "  " * level
            if callable(test_structure):
                if verbose:
                    print(
                        f"{indent}\n"
                        f"{indent}Running test\n"
                        f"{indent}\t{test_structure.__name__}"
                    )
                test_result = test_structure(vivEcoli_update, wcEcoli_update)
                report = {test_structure.__name__: test_result}

                passed = (
                    test_result[0] if isinstance(test_result, tuple) else test_result
                )
                if verbose:
                    print(
                        f'{indent}{"PASSED" if passed else "FAILED"} test "{test_structure.__name__}":'
                    )
                    if isinstance(test_result, tuple):
                        print(f"{indent}\t{test_result[1]}")

                if not passed:
                    self.failure = True
                    if self.fail_loudly:
                        assert passed, (
                            f'{indent}FAILED test "{test_structure.__name__}"'
                        )
            elif isinstance(test_structure, list):
                report = [
                    run_tests_iter(
                        vivEcoli_update, wcEcoli_update, test, verbose, level + 1
                    )
                    for test in test_structure
                ]
            else:  # assumed to be dictionary
                for k, v in test_structure.items():
                    if verbose:
                        print(f"{indent}Testing {k}...")

                    # check structure is correct:
                    if k not in vivEcoli_update:
                        raise ValueError(
                            f"Key {k} is missing from vivarium-ecoli update."
                        )
                    if k not in wcEcoli_update:
                        raise ValueError(f"Key {k} is missing from wcEcoli update.")

                    report[k] = run_tests_iter(
                        vivEcoli_update[k],
                        wcEcoli_update[k],
                        test_structure[k],
                        verbose,
                        level + 1,
                    )

            return report

        self.report = run_tests_iter(
            vivEcoli_update, wcEcoli_update, self.test_structure, verbose
        )

    def dump_report(self):
        def dump_report_iter(test_structure, report, level=0):
            indent = "  " * level

            if isinstance(test_structure, dict):
                for k, v in test_structure.items():
                    print(f"{indent}Tests for {k}:")
                    dump_report_iter(v, report[k], level + 1)
            elif isinstance(test_structure, list):
                for test, result in zip(test_structure, report):
                    dump_report_iter(test, result, level)
            else:
                for k, v in report.items():
                    passed = v[0] if isinstance(v, tuple) else v
                    print(f'{indent}{"PASSED" if passed else "FAILED"} test "{k}":')

                    if isinstance(v, tuple):
                        print(f"{indent}\t{v[1]}")

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
    return np.all(a == b)


def equal_len(a, b):
    return len(a) == len(b)


def len_almost_equal(a, b):
    return np.isclose(len(a), len(b), rtol=0.05, atol=1)


def array_equal(arr1, arr2):
    return (
        np.array_equal(arr1, arr2, equal_nan=True),
        f"Total difference (actual-expected) is {np.sum(np.abs(arr1 - arr2))}",
    )


def scalar_equal(v1, v2):
    return v1 == v2, f"Difference (actual-expected) is {v1 - v2}"


def array_almost_equal(arr1, arr2):
    p_errors = np.abs(arr1 - arr2) / np.maximum(np.abs(arr1), np.abs(arr2))
    p_errors[np.isnan(p_errors)] = 0

    return (
        np.all(p_errors <= PERCENT_ERROR_THRESHOLD),
        f"Max error = {np.max(p_errors):.4f}",
    )


def scalar_almost_equal(v1, v2, custom_threshold=None):
    pe = percent_error(v1, v2)
    if custom_threshold:
        return pe < custom_threshold or np.isnan(pe), f"Percent error = {pe:.4f}"
    return pe < PERCENT_ERROR_THRESHOLD or np.isnan(pe), f"Percent error = {pe:.4f}"


def custom_array_comp(percent_error_threshold=0.05):
    def _array_almost_equal(arr1, arr2):
        p_errors = np.abs(arr1 - arr2) / np.maximum(np.abs(arr1), np.abs(arr2))
        p_errors[np.isnan(p_errors)] = 0

        return (
            np.all(p_errors <= percent_error_threshold),
            f"Max error = {np.max(p_errors):.4f}",
        )

    return _array_almost_equal


def custom_scalar_comp(percent_error_threshold=0.05):
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
    return (
        result.pvalue > PVALUE_THRESHOLD,
        f"Two-sample T-test, t = {result.statistic:.4f}, p = {result.pvalue:.4f}"
        f" (Assuming {'' if same_var else 'un'}equal variances based on Bartlett test, p={var_result.pvalue:.4f})",
    )


def stochastic_equal(dist1, dist2):
    u, p = mannwhitneyu(dist1, dist2)
    return p > PVALUE_THRESHOLD, f"Mann-Whitney U, U = {u}, p = {p:.4f}"


def stochastic_equal_array(arr1, arr2):
    u_stats, p_vals = mannwhitneyu(arr1, arr2)
    return (
        p_vals > PVALUE_THRESHOLD,
        f"Mann-Whitney U, (U,p) = {list(zip(u_stats, [round(p, 4) for p in p_vals]))}",
    )


def array_diffs_report_test(filename, names=None, sort_by="absolute", sort_with=np.abs):
    def _array_diffs_report_test(a, b):
        result = array_diffs_report(a, b, names, sort_by, sort_with)
        with open(filename, "w") as f:
            f.write(result)
        return True

    return _array_diffs_report_test


# Common transformations ===============================================================================================


def pseudocount(arr):
    return arr + 1


# Test composition and modification ====================================================================================


def some_of(*tests):
    """
    For composing tests. Equivalent to asserting that at least one of the specified tests passes.

    Args:
        *tests: the test functions to use.

    Returns: a test function, returning a tuple of
             1. boolean representing whether at least one test passed, followed by
             2. a list of all the diagnostic information from all tests.
    """

    def _some_of(arr1, arr2):
        result = []
        for test in tests:
            result.append(test(arr1, arr2))

        result = (
            all([x[0] if isinstance(x, tuple) else x for x in result]),
            [x[1] if isinstance(x, tuple) else None for x in result],
        )

        return result

    return _some_of


def transform_and_run(
    transformation, test, transform_first=True, transform_second=True
):
    """
    Allows preprocessing of elements prior to testing. transformation(val) does the transformation
    prior to running the specified test.

    Args:
        transformation: list of functions to preprocess each element to be compared
        test: the test to run on transformed elements

    Returns: A test function.
    """

    def _transform_and_run(v1, v2):
        if transform_first:
            v1 = transformation(v1)

        if transform_second:
            v2 = transformation(v2)

        return test(v1, v2)

    return _transform_and_run
