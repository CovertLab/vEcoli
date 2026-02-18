"""
The parca, aka parameter calculator.

TODO: establish a controlled language for function behaviors (i.e. create* set* fit*)
TODO: functionalize so that values are not both set and returned from some methods
"""

import binascii
import copy as copy_module
import functools
import json
import os
import pickle
import time
import traceback
from typing import Callable, Any

from stochastic_arrow import StochasticSystem
import numpy as np
import scipy.optimize
import scipy.sparse

from ecoli.library.initial_conditions import create_bulk_container
from ecoli.library.schema import bulk_name_to_idx, counts
from reconstruction.ecoli.parca_condition_specs import (
    build_condition_cell_specs,
    get_condition_expression_and_concentrations,
)
from reconstruction.ecoli.parca_promoter_fitting import (
    fitPromoterBoundProbability,
    fitLigandConcentrations,
    calculatePromoterBoundProbability,
    calculateRnapRecruitment,
)
# Note: parca_stages module provides the @stage decorator for future unification
# of legacy and pure modes. Currently both paths are maintained for stability.
from reconstruction.ecoli.parca_updates import FittingOptions
from reconstruction.ecoli.simulation_data import SimulationDataEcoli
from wholecell.utils import parallelization, units
from wholecell.utils.fitting import normalize, masses_and_counts_for_homeostatic_target


# Fitting parameters
# NOTE: This threshold is arbitrary but relaxing it too much can slow doubling time.
FITNESS_THRESHOLD = 1e-9
MAX_FITTING_ITERATIONS = 200
N_SEEDS = 10

BASAL_EXPRESSION_CONDITION = "M9 Glucose minus AAs"

# Smoke mode: reduced conditions for fast testing (~20-30 min instead of 2-4 hours)
# These are the minimal conditions needed for a valid sim_data
SMOKE_CONDITIONS = {"basal", "with_aa"}
# TFs required by SMOKE_CONDITIONS (from condition_defs.tsv)
SMOKE_TFS = {
    "CPLX-125",        # trpR - with_aa active
    "MONOMER0-162",    # tyrR - with_aa active
    "CPLX0-228",       # argR - with_aa active
    "MONOMER0-155",    # lrp - with_aa active
    "CPLX0-7796",      # metJ - with_aa active
    "CPLX0-7669",      # argP - with_aa inactive
    "PUTA-CPLX",       # putA - with_aa inactive
    "EG12123-MONOMER", # lrhA - with_aa inactive
}

VERBOSE = 1

COUNTS_UNITS = units.dmol
VOLUME_UNITS = units.L
MASS_UNITS = units.g
TIME_UNITS = units.s

functions_run = []


def get_structural_signature(
    obj: Any,
    max_depth: int = 4,
    _depth: int = 0,
    _max_keys: int = 20,
    _max_attrs: int = 30,
) -> Any:
    """
    Generate a JSON-serializable structural signature of an object.

    Captures types, shapes, and keys without actual numeric data.
    Useful for debugging and comparing sim_data across stages/modes.

    Args:
        obj: Object to analyze (sim_data, dict, array, etc.)
        max_depth: Maximum recursion depth (default 4 for performance)
        _depth: Current recursion depth (internal)
        _max_keys: Maximum dict keys to include per level
        _max_attrs: Maximum object attributes to include per level

    Returns:
        JSON-serializable dict describing the structure
    """
    if _depth > max_depth:
        return {"_truncated": True, "_type": type(obj).__name__}

    # Handle None
    if obj is None:
        return None

    # Handle callable early (skip methods)
    if callable(obj) and not isinstance(obj, type):
        return {"_type": "callable", "name": getattr(obj, '__name__', '?')}

    # Handle numpy arrays (fast path - these are common)
    if isinstance(obj, np.ndarray):
        sig = {
            "_type": "ndarray",
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
        }
        if obj.dtype.names:
            sig["fields"] = list(obj.dtype.names)[:20]  # Limit field names
        return sig

    # Handle scipy sparse matrices
    if scipy.sparse.issparse(obj):
        return {
            "_type": f"sparse.{type(obj).__name__}",
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "nnz": obj.nnz,
        }

    # Handle units (wholecell.utils.units)
    if hasattr(obj, 'asNumber') and hasattr(obj, 'units'):
        try:
            val = obj.asNumber()
            if isinstance(val, np.ndarray):
                return {
                    "_type": "units.array",
                    "units": str(obj.units()),
                    "shape": list(val.shape),
                    "dtype": str(val.dtype),
                }
            else:
                return {"_type": "units.scalar", "units": str(obj.units())}
        except Exception:
            return {"_type": "units.unknown"}

    # Handle strings (fast path)
    if isinstance(obj, str):
        return {"_type": "str", "_len": len(obj)}

    # Handle numbers (fast path)
    if isinstance(obj, (int, float, bool)):
        return {"_type": type(obj).__name__}

    # Handle bytes
    if isinstance(obj, bytes):
        return {"_type": "bytes", "_len": len(obj)}

    # Handle dicts
    if isinstance(obj, dict):
        sig = {"_type": "dict", "_len": len(obj)}
        sorted_keys = sorted(obj.keys(), key=str)[:_max_keys]
        if len(obj) > _max_keys:
            sig["_truncated_keys"] = True
        sig["keys"] = {
            str(k): get_structural_signature(
                obj[k], max_depth, _depth + 1, _max_keys, _max_attrs
            )
            for k in sorted_keys
        }
        return sig

    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        type_name = "list" if isinstance(obj, list) else "tuple"
        sig = {"_type": type_name, "_len": len(obj)}
        if len(obj) == 0:
            return sig
        # Sample first element only for efficiency
        sig["sample_element"] = get_structural_signature(
            obj[0], max_depth, _depth + 1, _max_keys, _max_attrs
        )
        # Check types of first few elements
        sample_types = set(type(x).__name__ for x in obj[:10])
        if len(sample_types) == 1:
            sig["element_type"] = sample_types.pop()
        else:
            sig["element_types"] = list(sample_types)
        return sig

    # Handle sets
    if isinstance(obj, (set, frozenset)):
        sample = []
        try:
            sample = sorted([str(x) for x in list(obj)[:5]])
        except Exception:
            pass
        return {
            "_type": "set" if isinstance(obj, set) else "frozenset",
            "_len": len(obj),
            "sample": sample,
        }

    # Handle objects with __dict__ (custom classes like SimulationDataEcoli)
    if hasattr(obj, '__dict__'):
        sig = {"_type": type(obj).__name__}
        attrs = {}
        attr_count = 0
        # Use __dict__ directly for speed instead of dir()
        attr_names = sorted(obj.__dict__.keys()) if hasattr(obj, '__dict__') else []
        for attr_name in attr_names:
            if attr_name.startswith('_'):
                continue
            if attr_count >= _max_attrs:
                sig["_truncated_attrs"] = True
                break
            try:
                attr_val = obj.__dict__.get(attr_name)
                if attr_val is None or callable(attr_val):
                    continue
                attrs[attr_name] = get_structural_signature(
                    attr_val, max_depth, _depth + 1, _max_keys, _max_attrs
                )
                attr_count += 1
            except Exception:
                attrs[attr_name] = {"_error": "could not access"}
                attr_count += 1
        if attrs:
            sig["attributes"] = attrs
        return sig

    # Fallback
    return {"_type": type(obj).__name__}


def write_structural_signature(
    sim_data: Any,
    cell_specs: dict,
    stage_name: str,
    intermediates_dir: str,
) -> None:
    """
    Write structural signatures for sim_data and cell_specs to JSON files.

    Args:
        sim_data: The SimulationDataEcoli object
        cell_specs: The cell specifications dict
        stage_name: Name of the current stage (e.g., 'initialize')
        intermediates_dir: Directory to write signature files
    """
    if not intermediates_dir:
        return

    os.makedirs(intermediates_dir, exist_ok=True)

    signature = {
        "stage": stage_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sim_data": get_structural_signature(sim_data, max_depth=3),
        "cell_specs": get_structural_signature(cell_specs, max_depth=3),
    }

    signature_file = os.path.join(intermediates_dir, f"signature_{stage_name}.json")
    with open(signature_file, "w") as f:
        json.dump(signature, f, indent=2, default=str)


def make_fitting_options(**kwargs) -> FittingOptions:
    """
    Create a FittingOptions instance from kwargs.

    This helper extracts fitting-related parameters from kwargs and packages
    them into a FittingOptions dataclass for cleaner function signatures.

    Args:
        **kwargs: May contain any of the FittingOptions fields:
            - disable_ribosome_capacity_fitting
            - disable_rnapoly_capacity_fitting
            - variable_elongation_transcription
            - variable_elongation_translation
            - cpus
            - cache_dir

    Returns:
        FittingOptions instance with values from kwargs or defaults
    """
    return FittingOptions(
        disable_ribosome_capacity_fitting=kwargs.get("disable_ribosome_capacity_fitting", False),
        disable_rnapoly_capacity_fitting=kwargs.get("disable_rnapoly_capacity_fitting", False),
        variable_elongation_transcription=kwargs.get("variable_elongation_transcription", True),
        variable_elongation_translation=kwargs.get("variable_elongation_translation", False),
        cpus=kwargs.get("cpus", 1),
        cache_dir=kwargs.get("cache_dir"),
    )


def fitSimData_1(raw_data, pure_mode=False, **kwargs):
    """
    Fits parameters necessary for the simulation based on the knowledge base

    Inputs:
            raw_data (KnowledgeBaseEcoli) - knowledge base consisting of the
                    necessary raw data
            pure_mode (bool) - if True, use pure function implementations that
                    return update objects instead of mutating sim_data in-place.
                    This makes data flow explicit and enables better testing/caching.
            cpus (int) - number of processes to use (if > 1, use multiprocessing)
            smoke (bool) - if True, fit only the minimal set of TFs needed for
                    'basal' and 'with_aa' conditions (~20-30 min vs 2-4 hours).
                    Produces valid sim_data for testing/development.
            debug (bool) - DEPRECATED: use smoke instead. If True, fit only one
                    arbitrarily-chosen transcription factor. This mode is BROKEN
                    and will fail at promoter_binding due to shape mismatches.
            save_intermediates (bool) - if True, save the state (sim_data and cell_specs)
                    to disk in intermediates_directory after each Parca step
            intermediates_directory (str) - path to the directory to save intermediate
                    sim_data and cell_specs files to
            load_intermediate (str) - the function name of the Parca step to load
                    sim_data and cell_specs from; functions prior to and including this
                    will be skipped but all following functions will run
            variable_elongation_transcription (bool) - enable variable elongation
                    for transcription
            variable_elongation_translation (bool) - enable variable elongation for
                    translation
            disable_ribosome_capacity_fitting (bool) - if True, ribosome expression
                    is not fit to protein synthesis demands
            disable_rnapoly_capacity_fitting (bool) - if True, RNA polymerase
                    expression is not fit to protein synthesis demands
            cache_dir (str) - path to the directory to save cached data for
                    affinities of RNAs binding to endoRNases

    Note:
        The fitting parameters (variable_elongation_*, disable_*_capacity_fitting,
        cpus, cache_dir) can also be passed as a FittingOptions instance via
        the 'fitting_options' kwarg. Use make_fitting_options(**kwargs) to create one.
    """
    # Reset the function run tracking for intermediate loading
    global functions_run
    functions_run = []

    sim_data = SimulationDataEcoli()
    cell_specs = {}

    if pure_mode:
        # Pure mode: call compute_* functions directly and apply updates externally
        # This makes data flow explicit - each stage returns updates, we apply them
        result = compute_initialize(sim_data, cell_specs, raw_data=raw_data, **kwargs)
        sim_data, cell_specs = apply_stage_result(sim_data, cell_specs, result)

        result = compute_input_adjustments(sim_data, cell_specs, **kwargs)
        sim_data, cell_specs = apply_stage_result(sim_data, cell_specs, result)

        result = compute_basal_specs(sim_data, cell_specs, **kwargs)
        sim_data, cell_specs = apply_stage_result(sim_data, cell_specs, result)

        result = compute_tf_condition_specs(sim_data, cell_specs, **kwargs)
        sim_data, cell_specs = apply_stage_result(sim_data, cell_specs, result)

        result = compute_fit_condition(sim_data, cell_specs, **kwargs)
        sim_data, cell_specs = apply_stage_result(sim_data, cell_specs, result)

        result = compute_promoter_binding(sim_data, cell_specs, **kwargs)
        sim_data, cell_specs = apply_stage_result(sim_data, cell_specs, result)

        result = compute_adjust_promoters(sim_data, cell_specs, **kwargs)
        sim_data, cell_specs = apply_stage_result(sim_data, cell_specs, result)

        result = compute_set_conditions(sim_data, cell_specs, **kwargs)
        sim_data, cell_specs = apply_stage_result(sim_data, cell_specs, result)

        result = compute_final_adjustments(sim_data, cell_specs, **kwargs)
        sim_data, cell_specs = apply_stage_result(sim_data, cell_specs, result)
    else:
        # Legacy mode: functions mutate sim_data and cell_specs in-place
        sim_data, cell_specs = initialize(sim_data, cell_specs, raw_data=raw_data, **kwargs)
        sim_data, cell_specs = input_adjustments(sim_data, cell_specs, **kwargs)
        sim_data, cell_specs = basal_specs(sim_data, cell_specs, **kwargs)
        sim_data, cell_specs = tf_condition_specs(sim_data, cell_specs, **kwargs)
        sim_data, cell_specs = fit_condition(sim_data, cell_specs, **kwargs)
        sim_data, cell_specs = promoter_binding(sim_data, cell_specs, **kwargs)
        sim_data, cell_specs = adjust_promoters(sim_data, cell_specs, **kwargs)
        sim_data, cell_specs = set_conditions(sim_data, cell_specs, **kwargs)
        sim_data, cell_specs = final_adjustments(sim_data, cell_specs, **kwargs)

    if sim_data is None:
        raise ValueError(
            "sim_data is not specified.  Check that the"
            f" load_intermediate function ({kwargs.get('load_intermediate')})"
            " is correct and matches a function to be run."
        )

    return sim_data


def save_state(func):
    """
    Wrapper for functions called in fitSimData_1() to allow saving and loading
    of sim_data and cell_specs at different points in the parameter calculation
    pipeline.  This is useful for development in order to skip time intensive
    steps that are not required to recalculate in order to work with the desired
    stage of parameter calculation.

    This wrapper expects arguments in the kwargs passed into a wrapped function:
            save_intermediates (bool): if True, the state (sim_data and cell_specs)
                    will be saved to disk in intermediates_directory
            intermediates_directory (str): path to the directory to save intermediate
                    sim_data and cell_specs files to
            load_intermediate (str): the name of the function to load sim_data and
                    cell_specs from, functions prior to and including this will be
                    skipped but all following functions will run
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        load_intermediate = kwargs.get("load_intermediate")
        intermediates_dir = kwargs.get("intermediates_directory", "")

        # Files to save to or load from
        sim_data_file = os.path.join(intermediates_dir, f"sim_data_{func_name}.cPickle")
        cell_specs_file = os.path.join(
            intermediates_dir, f"cell_specs_{func_name}.cPickle"
        )

        # Run the wrapped function if the function to load is not specified or was already loaded
        if load_intermediate is None or load_intermediate in functions_run:
            start = time.time()
            sim_data, cell_specs = func(*args, **kwargs)
            end = time.time()
            print(f"Ran {func_name} in {end - start:.0f} s")
        # Load the saved results from the wrapped function if it is set to be loaded
        elif load_intermediate == func_name:
            if not os.path.exists(sim_data_file) or not os.path.exists(cell_specs_file):
                raise IOError(
                    f"Could not find intermediate files ({sim_data_file}"
                    f" or {cell_specs_file}) to load. Make sure to save intermediates"
                    " before trying to load them."
                )
            with open(sim_data_file, "rb") as f:
                sim_data = pickle.load(f)
            with open(cell_specs_file, "rb") as f:
                cell_specs = pickle.load(f)
            print(f"Loaded sim_data and cell_specs for {func_name}")
        # Skip running or loading if a later function will be loaded
        else:
            print(f"Skipped {func_name}")
            sim_data = None
            cell_specs = {}

        # Save the current state of the parameter calculator after the function to disk
        if (
            kwargs.get("save_intermediates", False)
            and intermediates_dir != ""
            and sim_data is not None
        ):
            os.makedirs(intermediates_dir, exist_ok=True)
            with open(sim_data_file, "wb") as f:
                pickle.dump(sim_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(cell_specs_file, "wb") as f:
                pickle.dump(cell_specs, f, protocol=pickle.HIGHEST_PROTOCOL)
            sim_data_size = os.path.getsize(sim_data_file) / (1024 * 1024)
            cell_specs_size = os.path.getsize(cell_specs_file) / (1024 * 1024)
            print(f"Saved data for {func_name}: sim_data={sim_data_size:.1f}MB, cell_specs={cell_specs_size:.1f}MB")

            # Write structural signature for debugging/comparison
            write_structural_signature(sim_data, cell_specs, func_name, intermediates_dir)

        # Record which functions have been run to know if the loaded function has run
        functions_run.append(func_name)

        return sim_data, cell_specs

    return wrapper


@save_state
def initialize(sim_data, cell_specs, raw_data=None, **kwargs):
    """Initialize sim_data from raw_data."""
    sim_data.initialize(
        raw_data=raw_data,
        basal_expression_condition=BASAL_EXPRESSION_CONDITION,
    )

    return sim_data, cell_specs


@save_state
def input_adjustments(sim_data, cell_specs, debug=False, smoke=False, **kwargs):
    """
    Apply input adjustments to sim_data.

    This stage filters TF conditions for smoke/debug mode and applies various
    adjustments to translation efficiencies, RNA expression, and degradation rates.

    Args:
        sim_data: The SimulationDataEcoli object to modify
        cell_specs: The cell specifications dict
        debug: DEPRECATED - use smoke instead. Broken mode.
        smoke: If True, fit only minimal TFs for basal + with_aa conditions

    Returns:
        Tuple of (sim_data, cell_specs)
    """
    # Limit the number of conditions that are being fit so that execution time decreases
    if smoke:
        print(
            f"SMOKE MODE: Fitting only {len(SMOKE_TFS)} TFs for {len(SMOKE_CONDITIONS)} conditions"
        )
        # Filter TF conditions to only those needed for smoke conditions
        # This limits which TFs are fit in tf_condition_specs and fit_condition
        sim_data.tf_to_active_inactive_conditions = {
            k: v for k, v in sim_data.tf_to_active_inactive_conditions.items()
            if k in SMOKE_TFS
        }
        # Filter combined conditions to smoke subset
        # This limits which combined conditions are built in buildCombinedConditionCellSpecifications
        sim_data.condition_active_tfs = {
            k: [tf for tf in v if tf in SMOKE_TFS]
            for k, v in sim_data.condition_active_tfs.items()
            if k in SMOKE_CONDITIONS
        }
        sim_data.condition_inactive_tfs = {
            k: [tf for tf in v if tf in SMOKE_TFS]
            for k, v in sim_data.condition_inactive_tfs.items()
            if k in SMOKE_CONDITIONS
        }
        # NOTE: Do NOT filter sim_data.conditions or condition_to_doubling_time here!
        # The TF-specific conditions (e.g., PUTA-CPLX__active) are needed by
        # buildTfConditionCellSpecifications and are already limited by filtering
        # tf_to_active_inactive_conditions above.
        print(f"  TFs: {sorted(sim_data.tf_to_active_inactive_conditions.keys())}")
        print(f"  Combined conditions: {sorted(sim_data.condition_active_tfs.keys())}")
    elif debug:
        print(
            "Warning: Running the Parca in debug mode - not all conditions will be fit.\n"
            "         This mode is BROKEN and will fail at promoter_binding.\n"
            "         Use --smoke instead for fast testing."
        )
        key = list(sim_data.tf_to_active_inactive_conditions.keys())[0]
        sim_data.tf_to_active_inactive_conditions = {
            key: sim_data.tf_to_active_inactive_conditions[key]
        }

    # Make adjustments for metabolic enzymes
    setTranslationEfficiencies(sim_data)
    set_balanced_translation_efficiencies(sim_data)
    setRNAExpression(sim_data)
    setRNADegRates(sim_data)
    setProteinDegRates(sim_data)

    return sim_data, cell_specs


@save_state
def basal_specs(
    sim_data,
    cell_specs,
    disable_ribosome_capacity_fitting=False,
    disable_rnapoly_capacity_fitting=False,
    variable_elongation_transcription=True,
    variable_elongation_translation=False,
    **kwargs,
):
    """Build basal cell specifications."""
    cell_specs = buildBasalCellSpecifications(
        sim_data,
        variable_elongation_transcription,
        variable_elongation_translation,
        disable_ribosome_capacity_fitting,
        disable_rnapoly_capacity_fitting,
    )

    # Set expression based on ppGpp regulation from basal expression
    sim_data.process.transcription.set_ppgpp_expression(sim_data)
    # TODO (Travis): use ppGpp expression in condition fitting

    # Modify other properties
    # Compute Km's
    Km = setKmCooperativeEndoRNonLinearRNAdecay(
        sim_data, cell_specs["basal"]["bulkContainer"], kwargs.get("cache_dir")
    )
    n_transcribed_rnas = len(sim_data.process.transcription.rna_data)
    sim_data.process.transcription.rna_data["Km_endoRNase"] = Km[:n_transcribed_rnas]
    sim_data.process.transcription.mature_rna_data["Km_endoRNase"] = Km[
        n_transcribed_rnas:
    ]

    ## Calculate and set maintenance values
    # ----- Growth associated maintenance -----
    fitMaintenanceCosts(sim_data, cell_specs["basal"]["bulkContainer"])

    return sim_data, cell_specs


@save_state
def tf_condition_specs(
    sim_data,
    cell_specs,
    cpus=1,
    disable_ribosome_capacity_fitting=False,
    disable_rnapoly_capacity_fitting=False,
    variable_elongation_transcription=True,
    variable_elongation_translation=False,
    **kwargs,
):
    """Build TF condition cell specifications."""
    # Limit the number of CPUs before printing it to stdout.
    cpus = parallelization.cpus(cpus)

    # Apply updates to cell_specs from buildTfConditionCellSpecifications for each TF condition
    conditions = list(sorted(sim_data.tf_to_active_inactive_conditions))
    args = [
        (
            sim_data,
            tf,
            variable_elongation_transcription,
            variable_elongation_translation,
            disable_ribosome_capacity_fitting,
            disable_rnapoly_capacity_fitting,
        )
        for tf in conditions
    ]
    apply_updates(
        buildTfConditionCellSpecifications, args, conditions, cell_specs, cpus
    )

    for conditionKey in cell_specs:
        if conditionKey == "basal":
            continue

        sim_data.process.transcription.rna_expression[conditionKey] = cell_specs[
            conditionKey
        ]["expression"]
        sim_data.process.transcription.rna_synth_prob[conditionKey] = cell_specs[
            conditionKey
        ]["synthProb"]
        sim_data.process.transcription.cistron_expression[conditionKey] = cell_specs[
            conditionKey
        ]["cistron_expression"]
        sim_data.process.transcription.fit_cistron_expression[conditionKey] = (
            cell_specs[conditionKey]["fit_cistron_expression"]
        )

    buildCombinedConditionCellSpecifications(
        sim_data,
        cell_specs,
        variable_elongation_transcription,
        variable_elongation_translation,
        disable_ribosome_capacity_fitting,
        disable_rnapoly_capacity_fitting,
    )

    return sim_data, cell_specs


@save_state
def fit_condition(sim_data, cell_specs, cpus=1, **kwargs):
    """Fit conditions to calculate bulk distributions and translation supply rates."""
    # Apply updates from fitCondition to cell_specs for each fit condition
    conditions = list(sorted(cell_specs))
    args = [(sim_data, cell_specs[condition], condition) for condition in conditions]
    apply_updates(fitCondition, args, conditions, cell_specs, cpus)

    for condition_label in sorted(cell_specs):
        nutrients = sim_data.conditions[condition_label]["nutrients"]
        if nutrients not in sim_data.translation_supply_rate:
            sim_data.translation_supply_rate[nutrients] = cell_specs[condition_label][
                "translation_aa_supply"
            ]

    return sim_data, cell_specs


@save_state
def promoter_binding(sim_data, cell_specs, **kwargs):
    """Fit promoter bound probability."""
    if VERBOSE > 0:
        print("Fitting promoter binding")
    # noinspection PyTypeChecker
    fitPromoterBoundProbability(sim_data, cell_specs)

    return sim_data, cell_specs


@save_state
def adjust_promoters(sim_data, cell_specs, **kwargs):
    """Adjust promoters by fitting ligand concentrations and calculating RNAP recruitment."""
    # noinspection PyTypeChecker
    fitLigandConcentrations(sim_data, cell_specs)
    calculateRnapRecruitment(sim_data, cell_specs)

    return sim_data, cell_specs


@save_state
def set_conditions(sim_data, cell_specs, **kwargs):
    """Set condition-specific parameters."""
    sim_data.process.transcription.rnaSynthProbFraction = {}
    sim_data.process.transcription.rnapFractionActiveDict = {}
    sim_data.process.transcription.rnaSynthProbRProtein = {}
    sim_data.process.transcription.rnaSynthProbRnaPolymerase = {}
    sim_data.process.transcription.rnaPolymeraseElongationRateDict = {}
    sim_data.expectedDryMassIncreaseDict = {}
    sim_data.process.translation.ribosomeElongationRateDict = {}
    sim_data.process.translation.ribosomeFractionActiveDict = {}

    for condition_label in sorted(cell_specs):
        condition = sim_data.conditions[condition_label]
        nutrients = condition["nutrients"]

        if VERBOSE > 0:
            print("Updating mass in condition {}".format(condition_label))
        spec = cell_specs[condition_label]

        concDict = sim_data.process.metabolism.concentration_updates.concentrations_based_on_nutrients(
            media_id=nutrients
        )
        concDict.update(
            sim_data.mass.getBiomassAsConcentrations(
                sim_data.condition_to_doubling_time[condition_label]
            )
        )

        avgCellDryMassInit, fitAvgSolublePoolMass = rescaleMassForSolubleMetabolites(
            sim_data,
            spec["bulkContainer"],
            concDict,
            sim_data.condition_to_doubling_time[condition_label],
        )

        if VERBOSE > 0:
            print("{} to {}".format(spec["avgCellDryMassInit"], avgCellDryMassInit))

        spec["avgCellDryMassInit"] = avgCellDryMassInit
        spec["fitAvgSolublePoolMass"] = fitAvgSolublePoolMass

        mRnaSynthProb = sim_data.process.transcription.rna_synth_prob[condition_label][
            sim_data.process.transcription.rna_data["is_mRNA"]
        ].sum()
        tRnaSynthProb = sim_data.process.transcription.rna_synth_prob[condition_label][
            sim_data.process.transcription.rna_data["is_tRNA"]
        ].sum()
        rRnaSynthProb = sim_data.process.transcription.rna_synth_prob[condition_label][
            sim_data.process.transcription.rna_data["is_rRNA"]
        ].sum()

        if len(condition["perturbations"]) == 0:
            if nutrients not in sim_data.process.transcription.rnaSynthProbFraction:
                sim_data.process.transcription.rnaSynthProbFraction[nutrients] = {
                    "mRna": mRnaSynthProb,
                    "tRna": tRnaSynthProb,
                    "rRna": rRnaSynthProb,
                }

            if nutrients not in sim_data.process.transcription.rnaSynthProbRProtein:
                prob = sim_data.process.transcription.rna_synth_prob[condition_label][
                    sim_data.process.transcription.rna_data[
                        "includes_ribosomal_protein"
                    ]
                ]
                sim_data.process.transcription.rnaSynthProbRProtein[nutrients] = prob

            if (
                nutrients
                not in sim_data.process.transcription.rnaSynthProbRnaPolymerase
            ):
                prob = sim_data.process.transcription.rna_synth_prob[condition_label][
                    sim_data.process.transcription.rna_data["includes_RNAP"]
                ]
                sim_data.process.transcription.rnaSynthProbRnaPolymerase[nutrients] = (
                    prob
                )

            if nutrients not in sim_data.process.transcription.rnapFractionActiveDict:
                frac = sim_data.growth_rate_parameters.get_fraction_active_rnap(
                    spec["doubling_time"]
                )
                sim_data.process.transcription.rnapFractionActiveDict[nutrients] = frac

            if (
                nutrients
                not in sim_data.process.transcription.rnaPolymeraseElongationRateDict
            ):
                rate = sim_data.growth_rate_parameters.get_rnap_elongation_rate(
                    spec["doubling_time"]
                )
                sim_data.process.transcription.rnaPolymeraseElongationRateDict[
                    nutrients
                ] = rate

            if nutrients not in sim_data.expectedDryMassIncreaseDict:
                sim_data.expectedDryMassIncreaseDict[nutrients] = spec[
                    "avgCellDryMassInit"
                ]

            if nutrients not in sim_data.process.translation.ribosomeElongationRateDict:
                rate = sim_data.growth_rate_parameters.get_ribosome_elongation_rate(
                    spec["doubling_time"]
                )
                sim_data.process.translation.ribosomeElongationRateDict[nutrients] = (
                    rate
                )

            if nutrients not in sim_data.process.translation.ribosomeFractionActiveDict:
                frac = sim_data.growth_rate_parameters.get_fraction_active_ribosome(
                    spec["doubling_time"]
                )
                sim_data.process.translation.ribosomeFractionActiveDict[nutrients] = (
                    frac
                )

    return sim_data, cell_specs


@save_state
def final_adjustments(sim_data, cell_specs, **kwargs):
    """Final adjustments for RNA attenuation, ppGpp, and supply constants."""
    # Adjust expression for RNA attenuation
    sim_data.process.transcription.calculate_attenuation(sim_data, cell_specs)

    # Adjust ppGpp regulated expression after conditions have been fit for physiological constraints
    sim_data.process.transcription.adjust_polymerizing_ppgpp_expression(sim_data)
    sim_data.process.transcription.adjust_ppgpp_expression_for_tfs(sim_data)

    # Set supply constants for amino acids based on condition supply requirements
    average_basal_container = create_bulk_container(sim_data, n_seeds=5)
    average_with_aa_container = create_bulk_container(
        sim_data, condition="with_aa", n_seeds=5
    )
    sim_data.process.metabolism.set_phenomological_supply_constants(sim_data)
    sim_data.process.metabolism.set_mechanistic_supply_constants(
        sim_data, cell_specs, average_basal_container, average_with_aa_container
    )
    sim_data.process.metabolism.set_mechanistic_export_constants(
        sim_data, cell_specs, average_basal_container
    )
    sim_data.process.metabolism.set_mechanistic_uptake_constants(
        sim_data, cell_specs, average_with_aa_container
    )

    # Set ppGpp reaction parameters
    sim_data.process.transcription.set_ppgpp_kinetics_parameters(
        average_basal_container, sim_data.constants
    )

    return sim_data, cell_specs


def apply_updates(
    func: Callable[..., dict],
    args: list[tuple],
    labels: list[str],
    dest: dict,
    cpus: int,
):
    """
    Use multiprocessing (if cpus > 1) to apply args to a function to get
    dictionary updates for a destination dictionary.

    Args:
            func: function to call with args
            args: list of args to apply to func
            labels: label for each set of args for exception information
            dest: destination dictionary that will be updated with results
                    from each function call
            cpus: number of cpus to use
    """

    if cpus > 1:
        print("Starting {} Parca processes".format(cpus))

        # Apply args to func
        pool = parallelization.pool(cpus)
        results = {label: pool.apply_async(func, a) for label, a in zip(labels, args)}
        pool.close()
        pool.join()

        # Check results from function calls and update dest
        failed = []
        for label, result in results.items():
            if result.successful():
                dest.update(result.get())
            else:
                # noinspection PyBroadException
                try:
                    result.get()
                except Exception:
                    traceback.print_exc()
                    failed.append(label)

        # Cleanup
        if failed:
            raise RuntimeError(
                "Error(s) raised for {} while using multiple processes".format(
                    ", ".join(failed)
                )
            )
        pool = None
        print("End parallel processing")
    else:
        for a in args:
            dest.update(func(*a))


def buildBasalCellSpecifications(
    sim_data,
    variable_elongation_transcription=True,
    variable_elongation_translation=False,
    disable_ribosome_capacity_fitting=False,
    disable_rnapoly_capacity_fitting=False,
):
    """
    Creates cell specifications for the basal condition by fitting expression.
    Relies on expressionConverge() to set the expression and update masses.

    Inputs
    ------
    - disable_ribosome_capacity_fitting (bool) - if True, ribosome expression
    is not fit
    - disable_rnapoly_capacity_fitting (bool) - if True, RNA polymerase
    expression is not fit

    Requires
    --------
    - Metabolite concentrations based on 'minimal' nutrients
    - 'basal' RNA expression
    - 'basal' doubling time

    Modifies
    --------
    - Average mass values of the cell
    - cistron expression
    - RNA expression and synthesis probabilities

    Returns
    --------
    - dict {'basal': dict} with the following keys in the dict from key 'basal':
            'concDict' {metabolite_name (str): concentration (float with units)} -
                    dictionary of concentrations for each metabolite with a concentration
            'fit_cistron_expression' (array of floats) - hypothetical expression for
                    each RNA cistron post-fit, total normalized to 1, if all
                    transcription units were monocistronic
            'expression' (array of floats) - expression for each RNA, total normalized to 1
            'doubling_time' (float with units) - cell doubling time
            'synthProb' (array of floats) - synthesis probability for each RNA,
                    total normalized to 1
            'avgCellDryMassInit' (float with units) - average initial cell dry mass
            'fitAvgSolubleTargetMolMass' (float with units) - the adjusted dry mass
                    of the soluble fraction of a cell
            - bulkContainer (np.ndarray object) - Two columns: 'id' for name and 'count'
                    for expected counts based on expression of all bulk molecules

    Notes
    -----
    - TODO - sets sim_data attributes and returns values - change to only return values
    """
    # Get initial expression and concentration data for basal condition
    expression, conc_dict, doubling_time, _, _ = get_condition_expression_and_concentrations(
        sim_data, "basal", condition_type="basal"
    )

    # Build cell specs using unified function
    cell_specs = build_condition_cell_specs(
        sim_data,
        "basal",
        expression,
        conc_dict,
        doubling_time,
        Km=None,
        cistron_expression=None,
        variable_elongation_transcription=variable_elongation_transcription,
        variable_elongation_translation=variable_elongation_translation,
        disable_ribosome_capacity_fitting=disable_ribosome_capacity_fitting,
        disable_rnapoly_capacity_fitting=disable_rnapoly_capacity_fitting,
        expressionConverge=expressionConverge,
    )

    # Update sim_data mass (basal-specific)
    avgCellDryMassInit = cell_specs["basal"]["avgCellDryMassInit"]
    fitAvgSolubleTargetMolMass = cell_specs["basal"]["fitAvgSolubleTargetMolMass"]

    sim_data.mass.avg_cell_dry_mass_init = avgCellDryMassInit
    sim_data.mass.avg_cell_dry_mass = (
        sim_data.mass.avg_cell_dry_mass_init
        * sim_data.mass.avg_cell_to_initial_cell_conversion_factor
    )
    sim_data.mass.avg_cell_water_mass_init = (
        sim_data.mass.avg_cell_dry_mass_init
        / sim_data.mass.cell_dry_mass_fraction
        * sim_data.mass.cell_water_mass_fraction
    )
    sim_data.mass.fitAvgSolubleTargetMolMass = fitAvgSolubleTargetMolMass

    # Update sim_data expression (basal-specific)
    sim_data.process.transcription.rna_expression["basal"][:] = cell_specs["basal"]["expression"]
    sim_data.process.transcription.rna_synth_prob["basal"][:] = cell_specs["basal"]["synthProb"]
    sim_data.process.transcription.fit_cistron_expression["basal"] = cell_specs["basal"]["fit_cistron_expression"]

    return cell_specs


def buildTfConditionCellSpecifications(
    sim_data,
    tf,
    variable_elongation_transcription=True,
    variable_elongation_translation=False,
    disable_ribosome_capacity_fitting=False,
    disable_rnapoly_capacity_fitting=False,
):
    """
    Creates cell specifications for a given transcription factor by
    fitting expression. Will set for the active and inactive TF condition.
    Relies on expressionConverge() to set the expression and masses.
    Uses fold change data relative to the 'basal' condition to determine
    expression for a given TF.

    Inputs
    ------
    - tf (str) - label for the transcription factor to fit (eg. 'CPLX-125')
    - disable_ribosome_capacity_fitting (bool) - if True, ribosome expression
    is not fit
    - disable_rnapoly_capacity_fitting (bool) - if True, RNA polymerase
    expression is not fit

    Requires
    --------
    - Metabolite concentrations based on nutrients for the TF
    - Adjusted 'basal' cistron expression
    - Doubling time for the TF
    - Fold changes in expression for each gene given the TF

    Returns
    --------
    - dict {tf + '__active'/'__inactive': dict} with the following keys in each dict:
            'concDict' {metabolite_name (str): concentration (float with units)} -
                    dictionary of concentrations for each metabolite with a concentration
            'expression' (array of floats) - expression for each RNA, total normalized to 1
            'doubling_time' (float with units) - cell doubling time
            'synthProb' (array of floats) - synthesis probability for each RNA,
                    total normalized to 1
            'cistron_expression' (array of floats) - hypothetical expression for
                    each RNA cistron, calculated from basal cistron expression levels
                    and fold change data
            'fit_cistron_expression' (array of floats) - hypothetical expression for
                    each RNA cistron post-fit, total normalized to 1, if all
                    transcription units were monocistronic
            'avgCellDryMassInit' (float with units) - average initial cell dry mass
            'fitAvgSolubleTargetMolMass' (float with units) - the adjusted dry mass
                    of the soluble fraction of a cell
            - bulkContainer (np.ndarray object) - Two columns: 'id' for name and 'count'
                    for expected counts based on expression of all bulk molecules
    """
    cell_specs = {}
    for tf_state in ["__active", "__inactive"]:
        condition_key = tf + tf_state

        # Get expression and concentration data for the TF condition
        expression, conc_dict, doubling_time, cistron_expression, Km = (
            get_condition_expression_and_concentrations(
                sim_data,
                condition_key,
                condition_type="tf",
                tf=tf,
                tf_state=tf_state,
                expressionFromConditionAndFoldChange=expressionFromConditionAndFoldChange,
            )
        )

        # Build cell specs using unified function
        condition_specs = build_condition_cell_specs(
            sim_data,
            condition_key,
            expression,
            conc_dict,
            doubling_time,
            Km=Km,
            cistron_expression=cistron_expression,
            variable_elongation_transcription=variable_elongation_transcription,
            variable_elongation_translation=variable_elongation_translation,
            disable_ribosome_capacity_fitting=disable_ribosome_capacity_fitting,
            disable_rnapoly_capacity_fitting=disable_rnapoly_capacity_fitting,
            expressionConverge=expressionConverge,
        )
        cell_specs.update(condition_specs)

    return cell_specs


def buildCombinedConditionCellSpecifications(
    sim_data,
    cell_specs,
    variable_elongation_transcription=True,
    variable_elongation_translation=False,
    disable_ribosome_capacity_fitting=False,
    disable_rnapoly_capacity_fitting=False,
):
    """
    Creates cell specifications for sets of transcription factors being active.
    These sets include conditions like 'with_aa' or 'no_oxygen' where multiple
    transcription factors will be active at the same time.

    Inputs
    ------
    - cell_specs {condition (str): dict} - information about each individual
    transcription factor condition
    - disable_ribosome_capacity_fitting (bool) - if True, ribosome expression
    is not fit
    - disable_rnapoly_capacity_fitting (bool) - if True, RNA polymerase
    expression is not fit

    Requires
    --------
    - Metabolite concentrations based on nutrients for the condition
    - Adjusted 'basal' RNA expression
    - Doubling time for the combined condition
    - Fold changes in expression for each gene given the TF

    Modifies
    --------
    - cell_specs dictionary for each combined condition
    - RNA expression and synthesis probabilities for each combined condition

    Notes
    -----
    - TODO - determine how to handle fold changes when multiple TFs change the
    same gene because multiplying both fold changes together might not be
    appropriate
    """
    for condition_key in sim_data.condition_active_tfs:
        # Skip 'basal' condition
        if condition_key == "basal":
            continue

        # Get expression and concentration data for combined condition
        expression, conc_dict, doubling_time, cistron_expression, Km = (
            get_condition_expression_and_concentrations(
                sim_data,
                condition_key,
                condition_type="combined",
                expressionFromConditionAndFoldChange=expressionFromConditionAndFoldChange,
            )
        )

        # Build cell specs using unified function
        condition_specs = build_condition_cell_specs(
            sim_data,
            condition_key,
            expression,
            conc_dict,
            doubling_time,
            Km=Km,
            cistron_expression=cistron_expression,
            variable_elongation_transcription=variable_elongation_transcription,
            variable_elongation_translation=variable_elongation_translation,
            disable_ribosome_capacity_fitting=disable_ribosome_capacity_fitting,
            disable_rnapoly_capacity_fitting=disable_rnapoly_capacity_fitting,
            expressionConverge=expressionConverge,
        )
        cell_specs.update(condition_specs)

        # Update sim_data expression (combined-specific)
        sim_data.process.transcription.rna_expression[condition_key] = cell_specs[condition_key]["expression"]
        sim_data.process.transcription.rna_synth_prob[condition_key] = cell_specs[condition_key]["synthProb"]
        sim_data.process.transcription.cistron_expression[condition_key] = cell_specs[condition_key]["cistron_expression"]
        sim_data.process.transcription.fit_cistron_expression[condition_key] = cell_specs[condition_key]["fit_cistron_expression"]


def expressionConverge(
    sim_data,
    expression,
    concDict,
    doubling_time,
    Km=None,
    conditionKey=None,
    variable_elongation_transcription=True,
    variable_elongation_translation=False,
    disable_ribosome_capacity_fitting=False,
    disable_rnapoly_capacity_fitting=False,
):
    """
    Iteratively fits synthesis probabilities for RNA. Calculates initial
    expression based on gene expression data and makes adjustments to match
    physiological constraints for ribosome and RNAP counts. Relies on
    fitExpression() to converge

    Inputs
    ------
    - expression (array of floats) - expression for each RNA, normalized to 1
    - concDict {metabolite (str): concentration (float with units of mol/volume)} -
    dictionary for concentrations of each metabolite with location tag
    - doubling_time (float with units of time) - doubling time
    - Km (array of floats with units of mol/volume) - Km for each RNA associated
    with RNases
    - disable_ribosome_capacity_fitting (bool) - if True, ribosome expression
    is not fit
    - disable_rnapoly_capacity_fitting (bool) - if True, RNA polymerase
    expression is not fit

    Requires
    --------
    - MAX_FITTING_ITERATIONS (int) - number of iterations to adjust expression
    before an exception is raised
    - FITNESS_THRESHOLD (float) - acceptable change from one iteration to break
    the fitting loop

    Returns
    --------
    - expression (array of floats) - adjusted expression for each RNA,
    normalized to 1
    - synthProb (array of floats) - synthesis probability for each RNA which
    accounts for expression and degradation rate, normalized to 1
    - avgCellDryMassInit (float with units of mass) - expected initial dry cell mass
    - fitAvgSolubleTargetMolMass (float with units of mass) - the adjusted dry mass
    of the soluble fraction of a cell
    - bulkContainer (np.ndarray object) - Two columns: 'id' for name and 'count'
            for expected counts based on expression of all bulk molecules
    """

    if VERBOSE > 0:
        print(
            f"Fitting RNA synthesis probabilities for condition {conditionKey} ...",
            end="",
        )

    for iteration in range(MAX_FITTING_ITERATIONS):
        if VERBOSE > 1:
            print("Iteration: {}".format(iteration))

        initialExpression = expression.copy()
        expression = setInitialRnaExpression(sim_data, expression, doubling_time)
        bulkContainer = createBulkContainer(sim_data, expression, doubling_time)
        avgCellDryMassInit, fitAvgSolubleTargetMolMass = (
            rescaleMassForSolubleMetabolites(
                sim_data, bulkContainer, concDict, doubling_time
            )
        )

        if not disable_rnapoly_capacity_fitting:
            setRNAPCountsConstrainedByPhysiology(
                sim_data,
                bulkContainer,
                doubling_time,
                avgCellDryMassInit,
                variable_elongation_transcription,
                Km,
            )

        if not disable_ribosome_capacity_fitting:
            setRibosomeCountsConstrainedByPhysiology(
                sim_data, bulkContainer, doubling_time, variable_elongation_translation
            )

        # Normalize expression and write out changes
        expression, synthProb, fit_cistron_expression, cistron_expression_res = (
            fitExpression(
                sim_data, bulkContainer, doubling_time, avgCellDryMassInit, Km
            )
        )

        degreeOfFit = np.sqrt(np.mean(np.square(initialExpression - expression)))

        if VERBOSE > 1:
            print("degree of fit: {}".format(degreeOfFit))
            print(
                f"Average cistron expression residuals: {np.linalg.norm(cistron_expression_res)}"
            )

        if degreeOfFit < FITNESS_THRESHOLD:
            print("! Fitting converged after {} iterations".format(iteration + 1))
            break

    else:
        raise Exception("Fitting did not converge")

    return (
        expression,
        synthProb,
        fit_cistron_expression,
        avgCellDryMassInit,
        fitAvgSolubleTargetMolMass,
        bulkContainer,
        concDict,
    )


def fitCondition(sim_data, spec, condition):
    """
    Takes a given condition and returns the predicted bulk average, bulk deviation,
    protein monomer average, protein monomer deviation, and amino acid supply to
    translation. This relies on calculateBulkDistributions and calculateTranslationSupply.

    Inputs
    ------
    - condition (str) - condition to fit (eg 'CPLX0-7705__active')
    - spec {property (str): property values} - cell specifications for the given condition.
    This function uses the specs "expression", "concDict", "avgCellDryMassInit",
    and "doubling_time"

    Returns
    --------
    - A dictionary {condition (str): spec (dict)} with the updated spec dictionary
    with the following values updated:
            - bulkAverageContainer (np.ndarray object) - Two columns: 'id' for name and 'count'
                    for the mean of the counts of all bulk molecules
            - bulkDeviationContainer (np.ndarray object) - Two columns: 'id' for name and 'count'
                    for the standard deviation of the counts of all bulk molecules
            - proteinMonomerAverageContainer (np.ndarray object) - Two columns: 'id' for name and 'count'
                    for the mean of the counts of all protein monomers
            - proteinMonomerDeviationContainer (np.ndarray object) - Two columns: 'id' for name and 'count'
                    for the standard deviation of the counts of all protein monomers
            - translation_aa_supply (array with units of mol/(mass.time)) - the supply rates
            for each amino acid to translation
    """

    if VERBOSE > 0:
        print("Fitting condition {}".format(condition))

    # Find bulk and protein distributions
    (
        bulkAverageContainer,
        bulkDeviationContainer,
        proteinMonomerAverageContainer,
        proteinMonomerDeviationContainer,
    ) = calculateBulkDistributions(
        sim_data,
        spec["expression"],
        spec["concDict"],
        spec["avgCellDryMassInit"],
        spec["doubling_time"],
    )
    spec["bulkAverageContainer"] = bulkAverageContainer
    spec["bulkDeviationContainer"] = bulkDeviationContainer
    spec["proteinMonomerAverageContainer"] = proteinMonomerAverageContainer
    spec["proteinMonomerDeviationContainer"] = proteinMonomerDeviationContainer

    # Find the supply rates of amino acids to translation given doubling time
    spec["translation_aa_supply"] = calculateTranslationSupply(
        sim_data,
        spec["doubling_time"],
        spec["proteinMonomerAverageContainer"],
        spec["avgCellDryMassInit"],
    )

    return {condition: spec}


def calculateTranslationSupply(
    sim_data, doubling_time, bulkContainer, avgCellDryMassInit
):
    """
    Returns the supply rates of all amino acids to translation given the desired
    doubling time. This creates a limit on the polypeptide elongation process,
    and thus on growth. The amino acid supply rate is found by calculating the
    concentration of amino acids per gram dry cell weight and multiplying by the
    loss to dilution given doubling time.

    Inputs
    ------
    - doubling_time (float with units of time) - measured doubling times given the condition
    - bulkContainer (np.ndarray object) - Two columns: 'id' for name and 'count'
            for count of all bulk molecules
    - avgCellDryMassInit (float with units of mass) - the average initial cell dry mass

    Notes
    -----
    - The supply of amino acids should not be based on a desired doubling time,
    but should come from a more mechanistic basis. This would allow simulations
    of environmental shifts in which the doubling time is unknown.
    """

    aaCounts = sim_data.process.translation.monomer_data[
        "aa_counts"
    ]  # the counts of each amino acid required for each protein
    protein_idx = bulk_name_to_idx(
        sim_data.process.translation.monomer_data["id"], bulkContainer["id"]
    )
    proteinCounts = counts(bulkContainer, protein_idx)  # the counts of all proteins
    nAvogadro = sim_data.constants.n_avogadro

    molAAPerGDCW = units.sum(
        aaCounts * np.tile(proteinCounts.reshape(-1, 1), (1, 21)), axis=0
    ) * ((1 / (units.aa * nAvogadro)) * (1 / avgCellDryMassInit))

    # Calculate required amino acid supply to translation to counter dilution
    translation_aa_supply = molAAPerGDCW * np.log(2) / doubling_time

    return translation_aa_supply


# Sub-fitting functions

# Import pure function infrastructure
from reconstruction.ecoli.parca_updates import (
    SimDataUpdate, ArrayUpdate, CellSpecsUpdate, StageResult,
    apply_sim_data_update, apply_cell_specs_update, apply_stage_result,
    # Input dataclasses for explicit dependencies
    TranslationEfficiencyInputs,
    BalancedTranslationInputs,
    RnaExpressionInputs,
    RnaDegRateInputs,
    ProteinDegRateInputs,
    SmokeModeInputs,
    # Configuration dataclasses for stage functions
    FittingOptions,
)
from reconstruction.ecoli.parca_outputs import (
    # Output dataclasses for typed stage results
    CellSpecsData,
    InitializeOutput,
    InputAdjustmentsOutput,
    BasalSpecsOutput,
    TfConditionSpecsOutput,
    FitConditionOutput,
    PromoterBindingOutput,
    AdjustPromotersOutput,
    SetConditionsOutput,
    FinalAdjustmentsOutput,
    # Mapper functions to convert outputs to StageResult
    apply_initialize_output,
    apply_input_adjustments_output,
    apply_basal_specs_output,
    apply_tf_condition_specs_output,
    apply_fit_condition_output,
    apply_promoter_binding_output,
    apply_adjust_promoters_output,
    apply_set_conditions_output,
    apply_final_adjustments_output,
    # Helper functions
    cell_specs_data_to_dict,
    dict_to_cell_specs_data,
)


# ============================================================================
# Input extraction helpers
# ============================================================================
# These functions extract the required inputs from sim_data for each compute function.


def extract_translation_efficiency_inputs(sim_data) -> TranslationEfficiencyInputs:
    """Extract inputs needed for compute_translation_efficiency_updates."""
    return TranslationEfficiencyInputs(
        translation_efficiencies_adjustments=sim_data.adjustments.translation_efficiencies_adjustments,
        monomer_ids=sim_data.process.translation.monomer_data["id"],
    )


def extract_balanced_translation_inputs(sim_data) -> BalancedTranslationInputs:
    """Extract inputs needed for compute_balanced_translation_updates."""
    return BalancedTranslationInputs(
        monomer_ids=sim_data.process.translation.monomer_data["id"],
        translation_efficiencies=sim_data.process.translation.translation_efficiencies_by_monomer,
        balanced_groups=sim_data.adjustments.balanced_translation_efficiencies,
    )


def extract_rna_expression_inputs(sim_data) -> RnaExpressionInputs:
    """Extract inputs needed for compute_rna_expression_updates."""
    # Pre-compute cistron_id_to_rna_indexes mapping
    cistron_ids = sim_data.process.transcription.cistron_data["id"]
    cistron_id_to_rna_indexes = {
        cid: sim_data.process.transcription.cistron_id_to_rna_indexes(cid)
        for cid in cistron_ids
    }
    return RnaExpressionInputs(
        cistron_ids=cistron_ids,
        rna_ids=sim_data.process.transcription.rna_data["id"],
        basal_expression=sim_data.process.transcription.rna_expression["basal"].copy(),
        rna_expression_adjustments=sim_data.adjustments.rna_expression_adjustments,
        cistron_id_to_rna_indexes=cistron_id_to_rna_indexes,
    )


def extract_rna_deg_rate_inputs(sim_data) -> RnaDegRateInputs:
    """Extract inputs needed for compute_rna_deg_rate_updates."""
    # Pre-compute cistron_id_to_rna_indexes mapping
    cistron_ids = sim_data.process.transcription.cistron_data["id"]
    cistron_id_to_rna_indexes = {
        cid: sim_data.process.transcription.cistron_id_to_rna_indexes(cid)
        for cid in cistron_ids
    }
    return RnaDegRateInputs(
        cistron_ids=cistron_ids,
        rna_ids=sim_data.process.transcription.rna_data["id"],
        rna_deg_rates_adjustments=sim_data.adjustments.rna_deg_rates_adjustments,
        cistron_id_to_rna_indexes=cistron_id_to_rna_indexes,
    )


def extract_protein_deg_rate_inputs(sim_data) -> ProteinDegRateInputs:
    """Extract inputs needed for compute_protein_deg_rate_updates."""
    return ProteinDegRateInputs(
        protein_deg_rates_adjustments=sim_data.adjustments.protein_deg_rates_adjustments,
        monomer_ids=sim_data.process.translation.monomer_data["id"],
    )


def extract_smoke_mode_inputs(sim_data) -> SmokeModeInputs:
    """Extract inputs needed for compute_smoke_mode_updates."""
    return SmokeModeInputs(
        tf_to_active_inactive_conditions=sim_data.tf_to_active_inactive_conditions,
        condition_active_tfs=sim_data.condition_active_tfs,
        condition_inactive_tfs=sim_data.condition_inactive_tfs,
    )


# ============================================================================
# Pure helper functions (return SimDataUpdate instead of mutating sim_data)
# ============================================================================


def compute_translation_efficiency_updates(inputs: TranslationEfficiencyInputs) -> SimDataUpdate:
    """
    Compute translation efficiency adjustments.

    Args:
        inputs: TranslationEfficiencyInputs with:
            - translation_efficiencies_adjustments: Dict[protein_id, multiplier]
            - monomer_ids: Array of monomer IDs

    Returns:
        SimDataUpdate describing the translation efficiency modifications.
    """
    update = SimDataUpdate()

    # Collect all indices and their multipliers
    indices_and_multipliers = []
    for protein, multiplier in inputs.translation_efficiencies_adjustments.items():
        idx = np.where(inputs.monomer_ids == protein)[0]
        for i in idx:
            indices_and_multipliers.append((i, multiplier))

    # Apply each multiplier as a separate array update
    for idx, multiplier in indices_and_multipliers:
        update.arrays[f'process.translation.translation_efficiencies_by_monomer:{idx}'] = ArrayUpdate(
            op='multiply', value=multiplier, indices=idx
        )

    return update


def compute_balanced_translation_updates(inputs: BalancedTranslationInputs) -> SimDataUpdate:
    """
    Compute balanced translation efficiency updates.

    Args:
        inputs: BalancedTranslationInputs with:
            - monomer_ids: Array of monomer IDs
            - translation_efficiencies: Current efficiency values
            - balanced_groups: List of protein groups to balance

    Returns:
        SimDataUpdate describing the translation efficiency modifications.
    """
    update = SimDataUpdate()

    # Build monomer_id (without [c] suffix) to index mapping
    monomer_id_to_index = {
        mid[:-3]: i for i, mid in enumerate(inputs.monomer_ids)
    }

    for group_idx, proteins in enumerate(inputs.balanced_groups):
        protein_indexes = np.array([monomer_id_to_index[m] for m in proteins])
        mean_trl_eff = inputs.translation_efficiencies[protein_indexes].mean()

        # Set all proteins in this group to the mean value
        update.arrays[f'process.translation.translation_efficiencies_by_monomer:group{group_idx}'] = ArrayUpdate(
            op='set', value=mean_trl_eff, indices=protein_indexes
        )

    return update


def compute_rna_expression_updates(inputs: RnaExpressionInputs) -> SimDataUpdate:
    """
    Compute RNA expression adjustments.

    Args:
        inputs: RnaExpressionInputs with:
            - cistron_ids: Array of cistron IDs
            - rna_ids: Array of RNA IDs
            - basal_expression: Current basal expression values
            - rna_expression_adjustments: Dict[mol_id, adjustment_factor]
            - cistron_id_to_rna_indexes: Pre-computed mapping

    Returns:
        SimDataUpdate describing the RNA expression modifications.
    """
    update = SimDataUpdate()

    cistron_ids_set = set(inputs.cistron_ids)
    rna_id_to_index = {
        rna_id[:-3]: i for i, rna_id in enumerate(inputs.rna_ids)
    }

    rna_index_to_adjustment = {}

    for mol_id, adj_factor in inputs.rna_expression_adjustments.items():
        if mol_id in cistron_ids_set:
            rna_indexes = inputs.cistron_id_to_rna_indexes[mol_id]
        elif mol_id in rna_id_to_index:
            rna_indexes = [rna_id_to_index[mol_id]]
        else:
            raise ValueError(
                f"Molecule ID {mol_id} not found in list of cistrons or transcription units."
            )

        for rna_index in rna_indexes:
            rna_index_to_adjustment[rna_index] = max(
                rna_index_to_adjustment.get(rna_index, 0), adj_factor
            )

    # Compute the new expression values
    current_expression = inputs.basal_expression.copy()
    for rna_index, adj_factor in rna_index_to_adjustment.items():
        current_expression[rna_index] *= adj_factor
    current_expression /= current_expression.sum()

    # Store as a dict update (replacing the 'basal' key value)
    update.dicts['process.transcription.rna_expression'] = {'basal': current_expression}

    return update


def compute_rna_deg_rate_updates(inputs: RnaDegRateInputs) -> SimDataUpdate:
    """
    Compute RNA degradation rate adjustments.

    Args:
        inputs: RnaDegRateInputs with:
            - cistron_ids: Array of cistron IDs
            - rna_ids: Array of RNA IDs
            - rna_deg_rates_adjustments: Dict[mol_id, adjustment_factor]
            - cistron_id_to_rna_indexes: Pre-computed mapping

    Returns:
        SimDataUpdate describing the RNA degradation rate modifications.
    """
    update = SimDataUpdate()

    cistron_id_to_index = {cid: i for i, cid in enumerate(inputs.cistron_ids)}
    rna_id_to_index = {rna_id[:-3]: i for i, rna_id in enumerate(inputs.rna_ids)}

    rna_index_to_adjustment = {}
    cistron_index_to_adjustment = {}

    for mol_id, adj_factor in inputs.rna_deg_rates_adjustments.items():
        if mol_id in cistron_id_to_index:
            cistron_index = cistron_id_to_index[mol_id]
            cistron_index_to_adjustment[cistron_index] = adj_factor
            rna_indexes = inputs.cistron_id_to_rna_indexes[mol_id]
        elif mol_id in rna_id_to_index:
            rna_indexes = [rna_id_to_index[mol_id]]
        else:
            raise ValueError(
                f"Molecule ID {mol_id} not found in list of cistrons or transcription units."
            )

        for rna_index in rna_indexes:
            rna_index_to_adjustment[rna_index] = max(
                rna_index_to_adjustment.get(rna_index, 0), adj_factor
            )

    # Add cistron deg_rate updates
    for cistron_index, adj_factor in cistron_index_to_adjustment.items():
        update.arrays[f'process.transcription.cistron_data:cistron{cistron_index}'] = ArrayUpdate(
            op='multiply', value=adj_factor, indices=cistron_index, field='deg_rate'
        )

    # Add RNA deg_rate updates
    for rna_index, adj_factor in rna_index_to_adjustment.items():
        update.arrays[f'process.transcription.rna_data:rna{rna_index}'] = ArrayUpdate(
            op='multiply', value=adj_factor, indices=rna_index, field='deg_rate'
        )

    return update


def compute_protein_deg_rate_updates(inputs: ProteinDegRateInputs) -> SimDataUpdate:
    """
    Compute protein degradation rate adjustments.

    Args:
        inputs: ProteinDegRateInputs with:
            - protein_deg_rates_adjustments: Dict[protein_id, adjustment_factor]
            - monomer_ids: Array of monomer IDs

    Returns:
        SimDataUpdate describing the protein degradation rate modifications.
    """
    update = SimDataUpdate()

    for protein, adj_factor in inputs.protein_deg_rates_adjustments.items():
        idx = np.where(inputs.monomer_ids == protein)[0]
        for i in idx:
            update.arrays[f'process.translation.monomer_data:protein{i}'] = ArrayUpdate(
                op='multiply', value=adj_factor, indices=i, field='deg_rate'
            )

    return update


def compute_smoke_mode_updates(inputs: SmokeModeInputs) -> SimDataUpdate:
    """
    Compute smoke mode filtering updates.

    Args:
        inputs: SmokeModeInputs with:
            - tf_to_active_inactive_conditions: TF condition mappings
            - condition_active_tfs: Active TFs per condition
            - condition_inactive_tfs: Inactive TFs per condition

    Returns:
        SimDataUpdate describing the smoke mode filtering.
    """
    update = SimDataUpdate()

    # Filter TF conditions to only those needed for smoke conditions
    update.attributes['tf_to_active_inactive_conditions'] = {
        k: v for k, v in inputs.tf_to_active_inactive_conditions.items()
        if k in SMOKE_TFS
    }

    # Filter combined conditions to smoke subset
    update.attributes['condition_active_tfs'] = {
        k: [tf for tf in v if tf in SMOKE_TFS]
        for k, v in inputs.condition_active_tfs.items()
        if k in SMOKE_CONDITIONS
    }

    update.attributes['condition_inactive_tfs'] = {
        k: [tf for tf in v if tf in SMOKE_TFS]
        for k, v in inputs.condition_inactive_tfs.items()
        if k in SMOKE_CONDITIONS
    }

    return update


def compute_debug_mode_updates(sim_data) -> SimDataUpdate:
    """
    Compute debug mode filtering updates without mutating sim_data.

    WARNING: Debug mode is BROKEN and will fail at promoter_binding.
    Use smoke mode instead.

    Returns:
        SimDataUpdate describing the debug mode filtering.
    """
    update = SimDataUpdate()

    key = list(sim_data.tf_to_active_inactive_conditions.keys())[0]
    update.attributes['tf_to_active_inactive_conditions'] = {
        key: sim_data.tf_to_active_inactive_conditions[key]
    }

    return update


# ============================================================================
# Pure stage functions (return StageResult instead of mutating)
# ============================================================================


def compute_input_adjustments_typed(sim_data, cell_specs, debug=False, smoke=False, **kwargs) -> InputAdjustmentsOutput:
    """
    Pure function to compute input adjustments, returning typed output.

    Note: This function applies updates incrementally to ensure sequential
    dependencies are respected (e.g., set_balanced_translation_efficiencies
    may depend on values set by setTranslationEfficiencies).

    Args:
        sim_data: The SimulationDataEcoli object (read-only)
        cell_specs: The cell specifications dict (read-only)
        debug: DEPRECATED - use smoke instead
        smoke: If True, fit only minimal TFs for basal + with_aa conditions

    Returns:
        InputAdjustmentsOutput with typed fields
    """
    # Work on a copy to preserve the original sim_data
    sim_data_copy = copy_module.deepcopy(sim_data)

    output = InputAdjustmentsOutput()

    # Handle smoke/debug mode filtering
    if smoke:
        print(
            f"SMOKE MODE: Fitting only {len(SMOKE_TFS)} TFs for {len(SMOKE_CONDITIONS)} conditions"
        )
        smoke_inputs = extract_smoke_mode_inputs(sim_data_copy)
        smoke_update = compute_smoke_mode_updates(smoke_inputs)
        # Extract typed fields from smoke update
        output.tf_to_active_inactive_conditions = smoke_update.attributes.get('tf_to_active_inactive_conditions')
        output.condition_active_tfs = smoke_update.attributes.get('condition_active_tfs')
        output.condition_inactive_tfs = smoke_update.attributes.get('condition_inactive_tfs')
        apply_sim_data_update(sim_data_copy, smoke_update)
        print(f"  TFs: {sorted(output.tf_to_active_inactive_conditions.keys())}")
        print(f"  Combined conditions: {sorted(output.condition_active_tfs.keys())}")
    elif debug:
        print(
            "Warning: Running the Parca in debug mode - not all conditions will be fit.\n"
            "         This mode is BROKEN and will fail at promoter_binding.\n"
            "         Use --smoke instead for fast testing."
        )
        debug_update = compute_debug_mode_updates(sim_data_copy)
        output.tf_to_active_inactive_conditions = debug_update.attributes.get('tf_to_active_inactive_conditions')
        apply_sim_data_update(sim_data_copy, debug_update)

    # 1. Translation efficiencies - extract multipliers
    trans_eff_inputs = extract_translation_efficiency_inputs(sim_data_copy)
    for protein, multiplier in trans_eff_inputs.translation_efficiencies_adjustments.items():
        idx_array = np.where(trans_eff_inputs.monomer_ids == protein)[0]
        for idx in idx_array:
            output.translation_efficiencies_multipliers[int(idx)] = multiplier

    # Apply to copy for next step
    trans_eff_update = compute_translation_efficiency_updates(trans_eff_inputs)
    apply_sim_data_update(sim_data_copy, trans_eff_update)

    # 2. Balanced translation efficiencies (depends on step 1)
    balanced_trans_inputs = extract_balanced_translation_inputs(sim_data_copy)
    # Build monomer_id to index mapping
    monomer_id_to_index = {
        mid[:-3]: i for i, mid in enumerate(balanced_trans_inputs.monomer_ids)
    }
    # Collect all indices that will be balanced
    all_balanced_indices = []
    for proteins in balanced_trans_inputs.balanced_groups:
        protein_indexes = np.array([monomer_id_to_index[m] for m in proteins])
        all_balanced_indices.extend(protein_indexes.tolist())
        mean_trl_eff = balanced_trans_inputs.translation_efficiencies[protein_indexes].mean()

    if all_balanced_indices:
        output.balanced_translation_indices = np.array(all_balanced_indices)
        # The value will vary per group, so we use the update mechanism
        # For the typed output, we store the final balanced state
        balanced_trans_update = compute_balanced_translation_updates(balanced_trans_inputs)
        apply_sim_data_update(sim_data_copy, balanced_trans_update)

    # 3. RNA expression
    rna_expr_inputs = extract_rna_expression_inputs(sim_data_copy)
    rna_expr_update = compute_rna_expression_updates(rna_expr_inputs)
    if 'process.transcription.rna_expression' in rna_expr_update.dicts:
        output.rna_expression_basal = rna_expr_update.dicts['process.transcription.rna_expression'].get('basal')
    apply_sim_data_update(sim_data_copy, rna_expr_update)

    # 4. RNA degradation rates
    rna_deg_inputs = extract_rna_deg_rate_inputs(sim_data_copy)
    cistron_id_to_index = {cid: i for i, cid in enumerate(rna_deg_inputs.cistron_ids)}
    rna_id_to_index = {rna_id[:-3]: i for i, rna_id in enumerate(rna_deg_inputs.rna_ids)}

    for mol_id, adj_factor in rna_deg_inputs.rna_deg_rates_adjustments.items():
        if mol_id in cistron_id_to_index:
            cistron_index = cistron_id_to_index[mol_id]
            output.cistron_deg_rate_multipliers[cistron_index] = adj_factor
            rna_indexes = rna_deg_inputs.cistron_id_to_rna_indexes[mol_id]
        elif mol_id in rna_id_to_index:
            rna_indexes = [rna_id_to_index[mol_id]]
        else:
            continue

        for rna_index in rna_indexes:
            output.rna_deg_rate_multipliers[rna_index] = max(
                output.rna_deg_rate_multipliers.get(rna_index, 0), adj_factor
            )

    rna_deg_update = compute_rna_deg_rate_updates(rna_deg_inputs)
    apply_sim_data_update(sim_data_copy, rna_deg_update)

    # 5. Protein degradation rates
    protein_deg_inputs = extract_protein_deg_rate_inputs(sim_data_copy)
    for protein, adj_factor in protein_deg_inputs.protein_deg_rates_adjustments.items():
        idx_array = np.where(protein_deg_inputs.monomer_ids == protein)[0]
        for idx in idx_array:
            output.protein_deg_rate_multipliers[int(idx)] = adj_factor

    return output


def compute_input_adjustments(sim_data, cell_specs, debug=False, smoke=False, **kwargs) -> StageResult:
    """
    Pure function to compute input adjustments.

    This is a backward-compatible wrapper that calls compute_input_adjustments_typed
    and converts the output to StageResult.

    Args:
        sim_data: The SimulationDataEcoli object (read-only)
        cell_specs: The cell specifications dict (read-only)
        debug: DEPRECATED - use smoke instead
        smoke: If True, fit only minimal TFs for basal + with_aa conditions

    Returns:
        StageResult with sim_data_update and cell_specs_update
    """
    output = compute_input_adjustments_typed(sim_data, cell_specs, debug=debug, smoke=smoke, **kwargs)
    return apply_input_adjustments_output(output)


def compute_basal_specs_typed(
    sim_data,
    cell_specs,
    fitting_options: FittingOptions = None,
    **kwargs,
) -> BasalSpecsOutput:
    """
    Pure function to compute basal cell specifications, returning typed output.

    This builds the basal condition cell specifications and computes
    expression, Km values, and maintenance costs.

    Args:
        sim_data: The SimulationDataEcoli object (will be deep-copied)
        cell_specs: The cell specifications dict (read-only)
        fitting_options: FittingOptions with configuration for fitting behavior

    Returns:
        BasalSpecsOutput with typed fields
    """
    # Use provided options or create defaults from kwargs for backward compatibility
    if fitting_options is None:
        fitting_options = FittingOptions(
            disable_ribosome_capacity_fitting=kwargs.get('disable_ribosome_capacity_fitting', False),
            disable_rnapoly_capacity_fitting=kwargs.get('disable_rnapoly_capacity_fitting', False),
            variable_elongation_transcription=kwargs.get('variable_elongation_transcription', True),
            variable_elongation_translation=kwargs.get('variable_elongation_translation', False),
            cache_dir=kwargs.get('cache_dir'),
        )

    sim_data_copy = copy_module.deepcopy(sim_data)

    # Build basal cell specifications (this returns a new cell_specs dict)
    new_cell_specs = buildBasalCellSpecifications(
        sim_data_copy,
        fitting_options.variable_elongation_transcription,
        fitting_options.variable_elongation_translation,
        fitting_options.disable_ribosome_capacity_fitting,
        fitting_options.disable_rnapoly_capacity_fitting,
    )

    # Set expression based on ppGpp regulation from basal expression
    sim_data_copy.process.transcription.set_ppgpp_expression(sim_data_copy)

    # Compute Km's
    Km = setKmCooperativeEndoRNonLinearRNAdecay(
        sim_data_copy, new_cell_specs["basal"]["bulkContainer"], fitting_options.cache_dir
    )
    n_transcribed_rnas = len(sim_data_copy.process.transcription.rna_data)

    # Maintenance costs (fitMaintenanceCosts sets sim_data.constants.darkATP)
    fitMaintenanceCosts(sim_data_copy, new_cell_specs["basal"]["bulkContainer"])

    # Build basal CellSpecsData
    basal_specs = new_cell_specs['basal']
    basal_cell_specs = CellSpecsData(
        concDict=basal_specs['concDict'],
        expression=basal_specs['expression'],
        doubling_time=basal_specs['doubling_time'],
        synthProb=basal_specs['synthProb'],
        fit_cistron_expression=basal_specs['fit_cistron_expression'],
        avgCellDryMassInit=basal_specs['avgCellDryMassInit'],
        fitAvgSolubleTargetMolMass=basal_specs['fitAvgSolubleTargetMolMass'],
        bulkContainer=basal_specs['bulkContainer'],
        cistron_expression=basal_specs.get('cistron_expression'),
    )

    return BasalSpecsOutput(
        # Mass updates
        avg_cell_dry_mass_init=sim_data_copy.mass.avg_cell_dry_mass_init,
        avg_cell_dry_mass=sim_data_copy.mass.avg_cell_dry_mass,
        avg_cell_water_mass_init=sim_data_copy.mass.avg_cell_water_mass_init,
        fitAvgSolubleTargetMolMass=sim_data_copy.mass.fitAvgSolubleTargetMolMass,
        # Expression
        rna_expression_basal=sim_data_copy.process.transcription.rna_expression['basal'],
        rna_synth_prob_basal=sim_data_copy.process.transcription.rna_synth_prob['basal'],
        fit_cistron_expression_basal=sim_data_copy.process.transcription.fit_cistron_expression['basal'],
        # ppGpp
        exp_ppgpp=sim_data_copy.process.transcription.exp_ppgpp,
        exp_free=sim_data_copy.process.transcription.exp_free,
        # Km
        rna_data_Km_endoRNase=Km[:n_transcribed_rnas],
        mature_rna_data_Km_endoRNase=Km[n_transcribed_rnas:],
        # Maintenance
        darkATP=sim_data_copy.constants.darkATP,
        # Cell specs
        basal_cell_specs=basal_cell_specs,
    )


def compute_basal_specs(
    sim_data,
    cell_specs,
    fitting_options: FittingOptions = None,
    **kwargs,
) -> StageResult:
    """
    Pure function to compute basal cell specifications.

    This is a backward-compatible wrapper that calls compute_basal_specs_typed
    and converts the output to StageResult.

    Args:
        sim_data: The SimulationDataEcoli object (will be deep-copied)
        cell_specs: The cell specifications dict (read-only)
        fitting_options: FittingOptions with configuration for fitting behavior

    Returns:
        StageResult with sim_data_update and cell_specs_update
    """
    output = compute_basal_specs_typed(sim_data, cell_specs, fitting_options=fitting_options, **kwargs)
    return apply_basal_specs_output(output)


def compute_tf_condition_specs_typed(
    sim_data,
    cell_specs,
    fitting_options: FittingOptions = None,
    **kwargs,
) -> TfConditionSpecsOutput:
    """
    Pure function to compute TF condition cell specifications, returning typed output.

    This builds cell specifications for each TF condition (active/inactive)
    and combined conditions.

    Args:
        sim_data: The SimulationDataEcoli object (will be deep-copied)
        cell_specs: The cell specifications dict (will be deep-copied)
        fitting_options: FittingOptions with configuration for fitting behavior

    Returns:
        TfConditionSpecsOutput with typed fields
    """
    # Use provided options or create defaults from kwargs for backward compatibility
    if fitting_options is None:
        fitting_options = FittingOptions(
            cpus=kwargs.get('cpus', 1),
            disable_ribosome_capacity_fitting=kwargs.get('disable_ribosome_capacity_fitting', False),
            disable_rnapoly_capacity_fitting=kwargs.get('disable_rnapoly_capacity_fitting', False),
            variable_elongation_transcription=kwargs.get('variable_elongation_transcription', True),
            variable_elongation_translation=kwargs.get('variable_elongation_translation', False),
        )

    sim_data_copy = copy_module.deepcopy(sim_data)
    cell_specs_copy = copy_module.deepcopy(cell_specs)

    cpus = parallelization.cpus(fitting_options.cpus)

    # Apply updates to cell_specs from buildTfConditionCellSpecifications for each TF condition
    conditions = list(sorted(sim_data_copy.tf_to_active_inactive_conditions))
    args = [
        (
            sim_data_copy,
            tf,
            fitting_options.variable_elongation_transcription,
            fitting_options.variable_elongation_translation,
            fitting_options.disable_ribosome_capacity_fitting,
            fitting_options.disable_rnapoly_capacity_fitting,
        )
        for tf in conditions
    ]
    apply_updates(
        buildTfConditionCellSpecifications, args, conditions, cell_specs_copy, cpus
    )

    # Build combined conditions
    buildCombinedConditionCellSpecifications(
        sim_data_copy,
        cell_specs_copy,
        fitting_options.variable_elongation_transcription,
        fitting_options.variable_elongation_translation,
        fitting_options.disable_ribosome_capacity_fitting,
        fitting_options.disable_rnapoly_capacity_fitting,
    )

    # Build typed output
    output = TfConditionSpecsOutput()

    # Collect expression dicts and condition specs (excluding basal)
    for conditionKey in cell_specs_copy:
        if conditionKey == "basal":
            continue
        if conditionKey not in cell_specs:
            # This is a new condition
            specs = cell_specs_copy[conditionKey]
            output.condition_specs[conditionKey] = CellSpecsData(
                concDict=specs['concDict'],
                expression=specs['expression'],
                doubling_time=specs['doubling_time'],
                synthProb=specs['synthProb'],
                fit_cistron_expression=specs['fit_cistron_expression'],
                avgCellDryMassInit=specs['avgCellDryMassInit'],
                fitAvgSolubleTargetMolMass=specs['fitAvgSolubleTargetMolMass'],
                bulkContainer=specs['bulkContainer'],
                cistron_expression=specs.get('cistron_expression'),
            )

        # Add expression updates
        output.rna_expression[conditionKey] = cell_specs_copy[conditionKey]["expression"]
        output.rna_synth_prob[conditionKey] = cell_specs_copy[conditionKey]["synthProb"]
        output.cistron_expression[conditionKey] = cell_specs_copy[conditionKey]["cistron_expression"]
        output.fit_cistron_expression[conditionKey] = cell_specs_copy[conditionKey]["fit_cistron_expression"]

    return output


def compute_tf_condition_specs(
    sim_data,
    cell_specs,
    fitting_options: FittingOptions = None,
    **kwargs,
) -> StageResult:
    """
    Pure function to compute TF condition cell specifications.

    This is a backward-compatible wrapper that calls compute_tf_condition_specs_typed
    and converts the output to StageResult.

    Args:
        sim_data: The SimulationDataEcoli object (will be deep-copied)
        cell_specs: The cell specifications dict (will be deep-copied)
        fitting_options: FittingOptions with configuration for fitting behavior

    Returns:
        StageResult with sim_data_update and cell_specs_update
    """
    output = compute_tf_condition_specs_typed(sim_data, cell_specs, fitting_options=fitting_options, **kwargs)
    return apply_tf_condition_specs_output(output)


def compute_fit_condition_typed(sim_data, cell_specs, fitting_options: FittingOptions = None, **kwargs) -> FitConditionOutput:
    """
    Pure function to fit conditions, returning typed output.

    This runs fitCondition for each condition to calculate bulk distributions
    and translation supply rates.

    Args:
        sim_data: The SimulationDataEcoli object (will be deep-copied)
        cell_specs: The cell specifications dict (will be deep-copied)
        fitting_options: FittingOptions with configuration for fitting behavior

    Returns:
        FitConditionOutput with typed fields
    """
    # Use provided options or create defaults from kwargs for backward compatibility
    if fitting_options is None:
        fitting_options = FittingOptions(cpus=kwargs.get('cpus', 1))

    sim_data_copy = copy_module.deepcopy(sim_data)
    cell_specs_copy = copy_module.deepcopy(cell_specs)

    cpus = parallelization.cpus(fitting_options.cpus)

    # Apply updates from fitCondition to cell_specs for each fit condition
    conditions = list(sorted(cell_specs_copy))
    args = [(sim_data_copy, cell_specs_copy[condition], condition) for condition in conditions]
    apply_updates(fitCondition, args, conditions, cell_specs_copy, cpus)

    output = FitConditionOutput()

    # Capture cell_specs updates
    for condition in conditions:
        output.condition_specs_updates[condition] = cell_specs_copy[condition]

    # Update translation supply rate
    for condition_label in sorted(cell_specs_copy):
        nutrients = sim_data_copy.conditions[condition_label]["nutrients"]
        if nutrients not in sim_data_copy.translation_supply_rate:
            output.translation_supply_rate[nutrients] = cell_specs_copy[condition_label]["translation_aa_supply"]

    return output


def compute_fit_condition(sim_data, cell_specs, fitting_options: FittingOptions = None, **kwargs) -> StageResult:
    """
    Pure function to fit conditions.

    This is a backward-compatible wrapper that calls compute_fit_condition_typed
    and converts the output to StageResult.

    Args:
        sim_data: The SimulationDataEcoli object (will be deep-copied)
        cell_specs: The cell specifications dict (will be deep-copied)
        fitting_options: FittingOptions with configuration for fitting behavior

    Returns:
        StageResult with sim_data_update and cell_specs_update
    """
    output = compute_fit_condition_typed(sim_data, cell_specs, fitting_options=fitting_options, **kwargs)
    return apply_fit_condition_output(output)


def compute_promoter_binding_typed(sim_data, cell_specs, **kwargs) -> PromoterBindingOutput:
    """
    Pure function to fit promoter bound probability, returning typed output.

    This stage uses convex optimization to fit TF-promoter binding probabilities
    that satisfy physiological constraints while matching experimental data.

    Args:
        sim_data: The SimulationDataEcoli object (will be deep-copied)
        cell_specs: The cell specifications dict (will be deep-copied)

    Returns:
        PromoterBindingOutput with typed fields
    """
    sim_data_copy = copy_module.deepcopy(sim_data)
    cell_specs_copy = copy_module.deepcopy(cell_specs)

    if VERBOSE > 0:
        print("Fitting promoter binding")

    fitPromoterBoundProbability(sim_data_copy, cell_specs_copy)

    return PromoterBindingOutput(
        pPromoterBound=sim_data_copy.pPromoterBound,
        rna_synth_prob=sim_data_copy.process.transcription.rna_synth_prob,
        basal_r_vector=cell_specs_copy['basal']['r_vector'],
        basal_r_columns=cell_specs_copy['basal']['r_columns'],
    )


def compute_promoter_binding(sim_data, cell_specs, **kwargs) -> StageResult:
    """
    Pure function to fit promoter bound probability.

    This is a backward-compatible wrapper that calls compute_promoter_binding_typed
    and converts the output to StageResult.

    Args:
        sim_data: The SimulationDataEcoli object (will be deep-copied)
        cell_specs: The cell specifications dict (will be deep-copied)

    Returns:
        StageResult with sim_data_update and cell_specs_update
    """
    output = compute_promoter_binding_typed(sim_data, cell_specs, **kwargs)
    return apply_promoter_binding_output(output)


def compute_adjust_promoters_typed(sim_data, cell_specs, **kwargs) -> AdjustPromotersOutput:
    """
    Pure function to adjust promoters, returning typed output.

    Fits ligand concentrations and calculates RNAP recruitment.

    Returns:
        AdjustPromotersOutput with typed fields
    """
    sim_data_copy = copy_module.deepcopy(sim_data)
    cell_specs_copy = copy_module.deepcopy(cell_specs)

    fitLigandConcentrations(sim_data_copy, cell_specs_copy)
    calculateRnapRecruitment(sim_data_copy, cell_specs_copy)

    return AdjustPromotersOutput(
        free_to_inactive_total=sim_data_copy.process.equilibrium.free_to_inactive_total,
        rnap_to_bound_prob_from_TFRNAP=sim_data_copy.process.transcription.rnap_to_bound_prob_from_TFRNAP,
        rnap_to_bound_prob_from_basal=sim_data_copy.process.transcription.rnap_to_bound_prob_from_basal,
    )


def compute_adjust_promoters(sim_data, cell_specs, **kwargs) -> StageResult:
    """
    Pure function to adjust promoters.

    This is a backward-compatible wrapper that calls compute_adjust_promoters_typed
    and converts the output to StageResult.

    Returns:
        StageResult with sim_data_update and cell_specs_update
    """
    output = compute_adjust_promoters_typed(sim_data, cell_specs, **kwargs)
    return apply_adjust_promoters_output(output)


def compute_set_conditions_typed(sim_data, cell_specs, **kwargs) -> SetConditionsOutput:
    """
    Pure function to set conditions, returning typed output.

    Rescales mass for soluble metabolites and populates condition-specific
    dictionaries on sim_data.

    Returns:
        SetConditionsOutput with typed fields
    """
    sim_data_copy = copy_module.deepcopy(sim_data)
    cell_specs_copy = copy_module.deepcopy(cell_specs)

    # Initialize the dicts
    rnaSynthProbFraction = {}
    rnapFractionActiveDict = {}
    rnaSynthProbRProtein = {}
    rnaSynthProbRnaPolymerase = {}
    rnaPolymeraseElongationRateDict = {}
    expectedDryMassIncreaseDict = {}
    ribosomeElongationRateDict = {}
    ribosomeFractionActiveDict = {}
    condition_specs_updates = {}

    for condition_label in sorted(cell_specs_copy):
        condition = sim_data_copy.conditions[condition_label]
        nutrients = condition["nutrients"]

        if VERBOSE > 0:
            print("Updating mass in condition {}".format(condition_label))
        spec = cell_specs_copy[condition_label]

        concDict = sim_data_copy.process.metabolism.concentration_updates.concentrations_based_on_nutrients(
            media_id=nutrients
        )
        concDict.update(
            sim_data_copy.mass.getBiomassAsConcentrations(
                sim_data_copy.condition_to_doubling_time[condition_label]
            )
        )

        avgCellDryMassInit, fitAvgSolublePoolMass = rescaleMassForSolubleMetabolites(
            sim_data_copy,
            spec["bulkContainer"],
            concDict,
            sim_data_copy.condition_to_doubling_time[condition_label],
        )

        if VERBOSE > 0:
            print("{} to {}".format(spec["avgCellDryMassInit"], avgCellDryMassInit))

        # Update cell_specs
        condition_specs_updates[condition_label] = {
            'avgCellDryMassInit': avgCellDryMassInit,
            'fitAvgSolublePoolMass': fitAvgSolublePoolMass,
        }

        mRnaSynthProb = sim_data_copy.process.transcription.rna_synth_prob[condition_label][
            sim_data_copy.process.transcription.rna_data["is_mRNA"]
        ].sum()
        tRnaSynthProb = sim_data_copy.process.transcription.rna_synth_prob[condition_label][
            sim_data_copy.process.transcription.rna_data["is_tRNA"]
        ].sum()
        rRnaSynthProb = sim_data_copy.process.transcription.rna_synth_prob[condition_label][
            sim_data_copy.process.transcription.rna_data["is_rRNA"]
        ].sum()

        if len(condition["perturbations"]) == 0:
            if nutrients not in rnaSynthProbFraction:
                rnaSynthProbFraction[nutrients] = {
                    "mRna": mRnaSynthProb,
                    "tRna": tRnaSynthProb,
                    "rRna": rRnaSynthProb,
                }

            if nutrients not in rnaSynthProbRProtein:
                prob = sim_data_copy.process.transcription.rna_synth_prob[condition_label][
                    sim_data_copy.process.transcription.rna_data["includes_ribosomal_protein"]
                ]
                rnaSynthProbRProtein[nutrients] = prob

            if nutrients not in rnaSynthProbRnaPolymerase:
                prob = sim_data_copy.process.transcription.rna_synth_prob[condition_label][
                    sim_data_copy.process.transcription.rna_data["includes_RNAP"]
                ]
                rnaSynthProbRnaPolymerase[nutrients] = prob

            if nutrients not in rnapFractionActiveDict:
                frac = sim_data_copy.growth_rate_parameters.get_fraction_active_rnap(
                    spec["doubling_time"]
                )
                rnapFractionActiveDict[nutrients] = frac

            if nutrients not in rnaPolymeraseElongationRateDict:
                rate = sim_data_copy.growth_rate_parameters.get_rnap_elongation_rate(
                    spec["doubling_time"]
                )
                rnaPolymeraseElongationRateDict[nutrients] = rate

            if nutrients not in expectedDryMassIncreaseDict:
                expectedDryMassIncreaseDict[nutrients] = avgCellDryMassInit

            if nutrients not in ribosomeElongationRateDict:
                rate = sim_data_copy.growth_rate_parameters.get_ribosome_elongation_rate(
                    spec["doubling_time"]
                )
                ribosomeElongationRateDict[nutrients] = rate

            if nutrients not in ribosomeFractionActiveDict:
                frac = sim_data_copy.growth_rate_parameters.get_fraction_active_ribosome(
                    spec["doubling_time"]
                )
                ribosomeFractionActiveDict[nutrients] = frac

    return SetConditionsOutput(
        rnaSynthProbFraction=rnaSynthProbFraction,
        rnapFractionActiveDict=rnapFractionActiveDict,
        rnaSynthProbRProtein=rnaSynthProbRProtein,
        rnaSynthProbRnaPolymerase=rnaSynthProbRnaPolymerase,
        rnaPolymeraseElongationRateDict=rnaPolymeraseElongationRateDict,
        ribosomeElongationRateDict=ribosomeElongationRateDict,
        ribosomeFractionActiveDict=ribosomeFractionActiveDict,
        expectedDryMassIncreaseDict=expectedDryMassIncreaseDict,
        condition_specs_updates=condition_specs_updates,
    )


def compute_set_conditions(sim_data, cell_specs, **kwargs) -> StageResult:
    """
    Pure function to set conditions.

    This is a backward-compatible wrapper that calls compute_set_conditions_typed
    and converts the output to StageResult.

    Returns:
        StageResult with sim_data_update and cell_specs_update
    """
    output = compute_set_conditions_typed(sim_data, cell_specs, **kwargs)
    return apply_set_conditions_output(output)


def compute_final_adjustments_typed(sim_data, cell_specs, **kwargs) -> FinalAdjustmentsOutput:
    """
    Pure function for final adjustments, returning typed output.

    Adjusts expression for RNA attenuation, ppGpp regulation, and sets
    supply/export/uptake constants.

    Returns:
        FinalAdjustmentsOutput with typed fields
    """
    sim_data_copy = copy_module.deepcopy(sim_data)
    cell_specs_copy = copy_module.deepcopy(cell_specs)

    # Adjust expression for RNA attenuation
    sim_data_copy.process.transcription.calculate_attenuation(sim_data_copy, cell_specs_copy)

    # Adjust ppGpp regulated expression
    sim_data_copy.process.transcription.adjust_polymerizing_ppgpp_expression(sim_data_copy)
    sim_data_copy.process.transcription.adjust_ppgpp_expression_for_tfs(sim_data_copy)

    # Set supply constants for amino acids
    average_basal_container = create_bulk_container(sim_data_copy, n_seeds=5)
    average_with_aa_container = create_bulk_container(
        sim_data_copy, condition="with_aa", n_seeds=5
    )
    sim_data_copy.process.metabolism.set_phenomological_supply_constants(sim_data_copy)
    sim_data_copy.process.metabolism.set_mechanistic_supply_constants(
        sim_data_copy, cell_specs_copy, average_basal_container, average_with_aa_container
    )
    sim_data_copy.process.metabolism.set_mechanistic_export_constants(
        sim_data_copy, cell_specs_copy, average_basal_container
    )
    sim_data_copy.process.metabolism.set_mechanistic_uptake_constants(
        sim_data_copy, cell_specs_copy, average_with_aa_container
    )

    # Set ppGpp reaction parameters
    sim_data_copy.process.transcription.set_ppgpp_kinetics_parameters(
        average_basal_container, sim_data_copy.constants
    )

    transcription = sim_data_copy.process.transcription
    metabolism = sim_data_copy.process.metabolism

    return FinalAdjustmentsOutput(
        # Transcription
        attenuation_basal_prob=transcription.attenuation_basal_prob,
        ppgpp_expression=transcription.ppgpp_expression,
        exp_ppgpp=transcription.exp_ppgpp,
        synth_prob_ppgpp=transcription.synth_prob_ppgpp,
        ppgpp_km=transcription.ppgpp_km,
        ppgpp_ki_synthetase=transcription.ppgpp_ki_synthetase,
        ppgpp_ki_hydrolase=transcription.ppgpp_ki_hydrolase,
        # Metabolism
        aa_supply_scaling=metabolism.aa_supply_scaling,
        aa_supply=metabolism.aa_supply,
        aa_export_kcat=metabolism.aa_export_kcat,
        aa_import_kis=metabolism.aa_import_kis,
    )


def compute_final_adjustments(sim_data, cell_specs, **kwargs) -> StageResult:
    """
    Pure function for final adjustments.

    This is a backward-compatible wrapper that calls compute_final_adjustments_typed
    and converts the output to StageResult.

    Returns:
        StageResult with sim_data_update and cell_specs_update
    """
    output = compute_final_adjustments_typed(sim_data, cell_specs, **kwargs)
    return apply_final_adjustments_output(output)


def compute_initialize_typed(sim_data, cell_specs, raw_data=None, **kwargs) -> InitializeOutput:
    """
    Pure function to initialize sim_data, returning typed output.

    This is largely a wrapper around sim_data.initialize() which does
    the bulk of the work.

    Returns:
        InitializeOutput with initialized sim_data
    """
    sim_data_copy = copy_module.deepcopy(sim_data)

    # Initialize sim_data
    sim_data_copy.initialize(
        raw_data=raw_data,
        basal_expression_condition=BASAL_EXPRESSION_CONDITION,
    )

    return InitializeOutput(sim_data=sim_data_copy)


def compute_initialize(sim_data, cell_specs, raw_data=None, **kwargs) -> StageResult:
    """
    Pure function to initialize sim_data.

    This is a backward-compatible wrapper that calls compute_initialize_typed
    and converts the output to StageResult.

    Returns:
        StageResult with sim_data_update and cell_specs_update
    """
    output = compute_initialize_typed(sim_data, cell_specs, raw_data=raw_data, **kwargs)
    return apply_initialize_output(output)


# ============================================================================
# Original mutating helper functions (preserved for backward compatibility)
# ============================================================================


def setTranslationEfficiencies(sim_data):
    """
    This function's goal is to set translation efficiencies for a subset of metabolic proteins.
    It first gathers the index of the proteins it wants to modify, then changes the monomer
    translation efficiencies based on the adjustment that is specified.
    These adjustments were made so that the simulation could run.

    Requires
    --------
    - For each protein that needs to be modified, it takes in an adjustment factor.

    Modifies
    --------
    - This function modifies, for a subset of proteins, their translational efficiencies in sim_data.
    It takes their current efficiency and multiplies them by the factor specified in adjustments.
    """

    for protein in sim_data.adjustments.translation_efficiencies_adjustments:
        idx = np.where(sim_data.process.translation.monomer_data["id"] == protein)[0]
        sim_data.process.translation.translation_efficiencies_by_monomer[idx] *= (
            sim_data.adjustments.translation_efficiencies_adjustments[protein]
        )


def set_balanced_translation_efficiencies(sim_data):
    """
    Sets the translation efficiencies of a group of proteins to be equal to the
    mean value of all proteins within the group.

    Requires
    --------
    - List of proteins that should have balanced translation efficiencies.

    Modifies
    --------
    - Translation efficiencies of proteins within each specified group.
    """
    monomer_id_to_index = {
        monomer["id"][:-3]: i
        for (i, monomer) in enumerate(sim_data.process.translation.monomer_data)
    }

    for proteins in sim_data.adjustments.balanced_translation_efficiencies:
        protein_indexes = np.array([monomer_id_to_index[m] for m in proteins])
        mean_trl_eff = sim_data.process.translation.translation_efficiencies_by_monomer[
            protein_indexes
        ].mean()
        sim_data.process.translation.translation_efficiencies_by_monomer[
            protein_indexes
        ] = mean_trl_eff


def setRNAExpression(sim_data):
    """
    This function's goal is to set expression levels for a subset of RNAs.
    It first gathers the index of the RNA's it wants to modify, then changes
    the expression levels of those RNAs, within sim_data, based on the
    specified adjustment factor. If the specified ID is an RNA cistron, the
    expression levels of all RNA molecules containing the cistron are adjusted.

    Requires
    --------
    - For each RNA that needs to be modified, it takes in an adjustment factor.

    Modifies
    --------
    - This function modifies the basal RNA expression levels set in sim_data,
    for the chosen RNAs. It takes their current basal expression and multiplies
    them by the factor specified in adjustments.
    - After updating the basal expression levels for the given RNAs, the
    function normalizes all the basal expression levels.
    """
    cistron_ids = set(sim_data.process.transcription.cistron_data["id"])
    rna_id_to_index = {
        rna_id[:-3]: i
        for (i, rna_id) in enumerate(sim_data.process.transcription.rna_data["id"])
    }

    rna_index_to_adjustment = {}

    for mol_id, adj_factor in sim_data.adjustments.rna_expression_adjustments.items():
        if mol_id in cistron_ids:
            # Find indexes of all RNAs containing the cistron
            rna_indexes = sim_data.process.transcription.cistron_id_to_rna_indexes(
                mol_id
            )
        elif mol_id in rna_id_to_index:
            rna_indexes = rna_id_to_index[mol_id]
        else:
            raise ValueError(
                f"Molecule ID {mol_id} not found in list of cistrons or transcription units."
            )

        # If multiple adjustments are made for the same RNA, take the maximum
        # adjustment factor
        for rna_index in rna_indexes:
            rna_index_to_adjustment[rna_index] = max(
                rna_index_to_adjustment.get(rna_index, 0), adj_factor
            )

    # Multiply all degradation rates with the specified adjustment factor
    for rna_index, adj_factor in rna_index_to_adjustment.items():
        sim_data.process.transcription.rna_expression["basal"][rna_index] *= adj_factor

    sim_data.process.transcription.rna_expression["basal"] /= (
        sim_data.process.transcription.rna_expression["basal"].sum()
    )


def setRNADegRates(sim_data):
    """
    This function's goal is to adjust the degradation rates for a subset of
    metabolic RNA's. It first gathers the index of the RNA's it wants to modify,
    then changes the degradation rates of those RNAs. If the specified ID is
    that of an RNA cistron, the degradation rates of all RNA molecules
    containing the cistron are adjusted. (Note: since RNA concentrations are
    assumed to be in equilibrium, increasing the degradation rate increases the
    synthesis rates of these RNAs)

    Requires
    --------
    - For each RNA that needs to be modified, it takes in an adjustment factor

    Modifies
    --------
    - This function modifies the RNA degradation rates for the chosen RNAs in
    sim_data. It takes their current degradation rate and multiplies them by the
    factor specified in adjustments.
    """
    cistron_id_to_index = {
        cistron_id: i
        for (i, cistron_id) in enumerate(
            sim_data.process.transcription.cistron_data["id"]
        )
    }
    rna_id_to_index = {
        rna_id[:-3]: i
        for (i, rna_id) in enumerate(sim_data.process.transcription.rna_data["id"])
    }

    rna_index_to_adjustment = {}

    for mol_id, adj_factor in sim_data.adjustments.rna_deg_rates_adjustments.items():
        if mol_id in cistron_id_to_index:
            # Multiply the cistron degradation rate with the specified
            # adjustment factor (Note: these rates are not actually used by the
            # simulation but are still adjusted for bookkeeping purposes)
            cistron_index = cistron_id_to_index[mol_id]
            sim_data.process.transcription.cistron_data.struct_array["deg_rate"][
                cistron_index
            ] *= sim_data.adjustments.rna_deg_rates_adjustments[mol_id]

            # Find indexes of all RNAs containing the cistron
            rna_indexes = sim_data.process.transcription.cistron_id_to_rna_indexes(
                mol_id
            )
        elif mol_id in rna_id_to_index:
            rna_indexes = rna_id_to_index[mol_id]
        else:
            raise ValueError(
                f"Molecule ID {mol_id} not found in list of cistrons or transcription units."
            )

        # If multiple adjustments are made for the same RNA, take the maximum
        # adjustment factor
        for rna_index in rna_indexes:
            rna_index_to_adjustment[rna_index] = max(
                rna_index_to_adjustment.get(rna_index, 0), adj_factor
            )

    # Multiply all degradation rates with the specified adjustment factor
    for rna_index, adj_factor in rna_index_to_adjustment.items():
        sim_data.process.transcription.rna_data.struct_array["deg_rate"][rna_index] *= (
            adj_factor
        )


def setProteinDegRates(sim_data):
    """
    This function's goal is to set the degradation rates for a subset of proteins.
    It first gathers the index of the proteins it wants to modify, then changes the degradation
    rates of those proteins. These adjustments were made so that the simulation could run.

    Requires
    --------
    - For each protein that needs to be modified it take in an adjustment factor.

    Modifies
    --------
    - This function modifies the protein degradation rates for the chosen proteins in sim_data.
    It takes their current degradation rate and multiplies them by the factor specified in adjustments.
    """

    for protein in sim_data.adjustments.protein_deg_rates_adjustments:
        idx = np.where(sim_data.process.translation.monomer_data["id"] == protein)[0]
        sim_data.process.translation.monomer_data.struct_array["deg_rate"][idx] *= (
            sim_data.adjustments.protein_deg_rates_adjustments[protein]
        )


def rescaleMassForSolubleMetabolites(sim_data, bulkMolCntr, concDict, doubling_time):
    """
    Adjust the cell's mass to accomodate target small molecule concentrations.

    Inputs
    ------
    - bulkMolCntr (np.ndarray object) - Two columns: 'id' for name and 'count'
            for count of all bulk molecules
    - concDict (dict) - a dictionary of metabolite ID (string) : concentration (unit'd number, dimensions of concentration) pairs
    - doubling_time (float with units of time) - measured doubling times given the condition

    Requires
    --------
    - Cell mass fraction data at a given doubling time.
    - Average cell density.
    - The conversion factor for transforming from the size of an average cell to the size of a cell
      immediately following division.
    - Avogadro's number.
    - Concentrations of small molecules (including both dry mass components and water).

    Modifies
    --------
    - Adds small molecule counts to bulkMolCntr.

    Returns
    -------
    - newAvgCellDryMassInit, the adjusted dry mass of a cell immediately following division.
    - fitAvgSolubleTargetMolMass, the adjusted dry mass of the soluble fraction of a cell
    """

    avgCellFractionMass = sim_data.mass.get_component_masses(doubling_time)

    non_small_molecule_initial_cell_mass = (
        avgCellFractionMass["proteinMass"]
        + avgCellFractionMass["rnaMass"]
        + avgCellFractionMass["dnaMass"]
    ) / sim_data.mass.avg_cell_to_initial_cell_conversion_factor

    molar_units = units.mol / units.L

    targetMoleculeIds = sorted(concDict)
    targetMoleculeConcentrations = molar_units * np.array(
        [concDict[key].asNumber(molar_units) for key in targetMoleculeIds]
    )  # Have to strip and replace units to obtain the proper array data type

    assert np.all(targetMoleculeConcentrations.asNumber(molar_units) > 0), (
        "Homeostatic dFBA objective requires non-zero (positive) concentrations"
    )

    molecular_weights = sim_data.getter.get_masses(targetMoleculeIds)

    massesToAdd, countsToAdd = masses_and_counts_for_homeostatic_target(
        non_small_molecule_initial_cell_mass,
        targetMoleculeConcentrations,
        molecular_weights,
        sim_data.constants.cell_density,
        sim_data.constants.n_avogadro,
    )

    target_molecule_idx = bulk_name_to_idx(targetMoleculeIds, bulkMolCntr["id"])
    bulkMolCntr["count"][target_molecule_idx] = countsToAdd

    # Increase avgCellDryMassInit to match these numbers & rescale mass fractions
    smallMoleculetargetMoleculesDryMass = units.hstack(
        (
            massesToAdd[: targetMoleculeIds.index("WATER[c]")],
            massesToAdd[targetMoleculeIds.index("WATER[c]") + 1 :],
        )
    )  # remove water since it's not part of the dry mass

    newAvgCellDryMassInit = non_small_molecule_initial_cell_mass + units.sum(
        smallMoleculetargetMoleculesDryMass
    )
    fitAvgSolubleTargetMolMass = (
        units.sum(smallMoleculetargetMoleculesDryMass)
        * sim_data.mass.avg_cell_to_initial_cell_conversion_factor
    )

    return newAvgCellDryMassInit, fitAvgSolubleTargetMolMass


def setInitialRnaExpression(sim_data, expression, doubling_time):
    """
    Creates a container that with the initial count and ID of each RNA,
    calculated based on the mass fraction, molecular weight, and expression
    distribution of each RNA. For rRNA the counts are set based on mass, while
    for tRNA and mRNA the counts are set based on mass and relative abundance.
    Relies on the math function totalCountFromMassesAndRatios.

    Requires
    --------
    - Needs information from the knowledge base about the mass fraction,
    molecular weight, and distribution of each RNA species.

    Inputs
    ------
    - expression (array of floats) - expression for each RNA, normalized to 1
    - doubling_time (float with units of time) - doubling time for condition

    Returns
    --------
    - expression (array of floats) - contains the adjusted RNA expression,
    normalized to 1

    Notes
    -----
    - Now rnaData["synthProb"] does not match "expression"

    """
    # Load from sim_data
    n_avogadro = sim_data.constants.n_avogadro
    transcription = sim_data.process.transcription
    cistron_data = transcription.cistron_data
    rna_data = transcription.rna_data
    get_average_copy_number = sim_data.process.replication.get_average_copy_number
    rna_mw = rna_data["mw"]
    rna_rRNA_mw = rna_data["rRNA_mw"]
    rna_tRNA_mw = rna_data["tRNA_mw"]
    rna_coord = rna_data["replication_coordinate"]

    # Mask arrays for each RNA type
    is_rRNA = rna_data["is_rRNA"]
    is_tRNA = rna_data["is_tRNA"]
    is_mRNA = rna_data["is_mRNA"]

    # Get list of RNA IDs for each type and tRNA cistron IDs
    all_RNA_ids = rna_data["id"]
    ids_rRNA = all_RNA_ids[is_rRNA]
    ids_mRNA = all_RNA_ids[is_mRNA]
    ids_tRNA = all_RNA_ids[is_tRNA]
    ids_tRNA_cistrons = cistron_data["id"][cistron_data["is_tRNA"]]

    # Get mass fractions of each RNA type for this condition
    initial_rna_mass = (
        sim_data.mass.get_component_masses(doubling_time)["rnaMass"]
        / sim_data.mass.avg_cell_to_initial_cell_conversion_factor
    )
    ppgpp = sim_data.growth_rate_parameters.get_ppGpp_conc(doubling_time)
    rna_fractions = transcription.get_rna_fractions(ppgpp)
    total_mass_rRNA = initial_rna_mass * rna_fractions["rRNA"]
    total_mass_tRNA = initial_rna_mass * rna_fractions["tRNA"]
    total_mass_mRNA = initial_rna_mass * rna_fractions["mRNA"]

    # Get molecular weights of each RNA. For rRNAs/tRNAs, we only account for
    # the masses of the mature rRNAs/tRNAs within each RNA, since these RNAs are
    # almost instantly processed to yield the mature RNAs.
    individual_masses_rRNA = rna_rRNA_mw[is_rRNA] / n_avogadro
    individual_masses_tRNA = rna_tRNA_mw[is_tRNA] / n_avogadro
    individual_masses_mRNA = rna_mw[is_mRNA] / n_avogadro

    # Set rRNA TU expression assuming equal per-copy transcription
    # probabilities. The combined expression levels of each rRNA TU are assumed
    # to be proportional to their expected average copy numbers, which are
    # dependent on the doubling time and the chromosomal position.
    tau = doubling_time.asNumber(units.min)
    coord_rRNA = rna_coord[is_rRNA]
    n_avg_copy_rRNA = get_average_copy_number(tau, coord_rRNA)
    distribution_rRNA = normalize(n_avg_copy_rRNA)

    total_count_rRNA = totalCountFromMassesAndRatios(
        total_mass_rRNA, individual_masses_rRNA, distribution_rRNA
    )
    counts_rRNA = total_count_rRNA * distribution_rRNA

    # Get the total mass of tRNAs that are expressed from rRNA TUs and subtract
    # this mass from the total tRNA mass
    tRNA_masses_in_each_rRNA = rna_tRNA_mw[is_rRNA] / n_avogadro
    total_mass_tRNA_in_rRNAs = units.dot(counts_rRNA, tRNA_masses_in_each_rRNA)
    total_mass_tRNA -= total_mass_tRNA_in_rRNAs

    # Get tRNA cistron distribution (see Dong 1996), while setting values for
    # cistrons that are expressed from rRNAs to zero
    tRNA_distribution = sim_data.mass.get_trna_distribution(doubling_time)
    tRNA_id_to_dist = {
        trna_id: dist
        for (trna_id, dist) in zip(
            tRNA_distribution["id"], tRNA_distribution["molar_ratio_to_16SrRNA"]
        )
    }
    distribution_tRNA_cistrons = np.zeros(len(ids_tRNA_cistrons))
    for i, tRNA_id in enumerate(ids_tRNA_cistrons):
        distribution_tRNA_cistrons[i] = tRNA_id_to_dist[tRNA_id]

    tRNA_expressed_from_rRNA_mask = transcription.cistron_tu_mapping_matrix.dot(
        is_rRNA
    )[cistron_data["is_tRNA"]].astype(bool)
    distribution_tRNA_cistrons[tRNA_expressed_from_rRNA_mask] = 0
    distribution_tRNA_cistrons = normalize(distribution_tRNA_cistrons)

    # Approximate distribution of tRNA-including transcripts from tRNA cistron
    # distribution by using NNLS
    distribution_tRNA_including_transcripts, _ = transcription.fit_trna_expression(
        distribution_tRNA_cistrons
    )

    # Get distribution of tRNA-including transcripts that are not rRNAs
    is_hybrid = rna_data["is_rRNA"][rna_data["includes_tRNA"]]
    distribution_tRNA = distribution_tRNA_including_transcripts[~is_hybrid]
    distribution_tRNA = normalize(distribution_tRNA)

    # Assign tRNA counts based on this distribution
    total_count_tRNA = totalCountFromMassesAndRatios(
        total_mass_tRNA, individual_masses_tRNA, distribution_tRNA
    )
    counts_tRNA = total_count_tRNA * distribution_tRNA

    # Assign mRNA counts based on mass and relative abundances (microarrays)
    distribution_mRNA = normalize(expression[is_mRNA])
    total_count_mRNA = totalCountFromMassesAndRatios(
        total_mass_mRNA, individual_masses_mRNA, distribution_mRNA
    )
    counts_mRNA = total_count_mRNA * distribution_mRNA

    # Set expression counts in container
    rRNA_idx = bulk_name_to_idx(ids_rRNA, all_RNA_ids)
    tRNA_idx = bulk_name_to_idx(ids_tRNA, all_RNA_ids)
    mRNA_idx = bulk_name_to_idx(ids_mRNA, all_RNA_ids)
    rna_expression_container = np.zeros(len(all_RNA_ids), dtype=np.float64)
    rna_expression_container[rRNA_idx] = counts_rRNA
    rna_expression_container[tRNA_idx] = counts_tRNA
    rna_expression_container[mRNA_idx] = counts_mRNA

    expression = normalize(rna_expression_container)

    return expression


def totalCountIdDistributionProtein(sim_data, expression, doubling_time):
    """
    Calculates the total counts of proteins from the relative expression of RNA,
    individual protein mass, and total protein mass. Relies on the math functions
    netLossRateFromDilutionAndDegradationProtein, proteinDistributionFrommRNA,
    totalCountFromMassesAndRatios.

    Inputs
    ------
    - expression (array of floats) - relative frequency distribution of RNA expression
    - doubling_time (float with units of time) - measured doubling time given the condition

    Returns
    --------
    - total_count_protein (float) - total number of proteins
    - ids_protein (array of str) - name of each protein with location tag
    - distribution_protein (array of floats) - distribution for each protein,
    normalized to 1
    """
    ids_protein = sim_data.process.translation.monomer_data["id"]
    total_mass_protein = (
        sim_data.mass.get_component_masses(doubling_time)["proteinMass"]
        / sim_data.mass.avg_cell_to_initial_cell_conversion_factor
    )
    individual_masses_protein = (
        sim_data.process.translation.monomer_data["mw"] / sim_data.constants.n_avogadro
    )

    mRNA_cistron_expression = (
        sim_data.process.transcription.cistron_tu_mapping_matrix.dot(expression)[
            sim_data.process.transcription.cistron_data["is_mRNA"]
        ]
    )
    distribution_transcripts_by_protein = normalize(
        sim_data.relation.monomer_to_mRNA_cistron_mapping().dot(mRNA_cistron_expression)
    )

    translation_efficiencies_by_protein = normalize(
        sim_data.process.translation.translation_efficiencies_by_monomer
    )
    degradationRates = sim_data.process.translation.monomer_data["deg_rate"]

    # Find the net protein loss
    netLossRate_protein = netLossRateFromDilutionAndDegradationProtein(
        doubling_time, degradationRates
    )

    # Find the protein distribution
    distribution_protein = proteinDistributionFrommRNA(
        distribution_transcripts_by_protein,
        translation_efficiencies_by_protein,
        netLossRate_protein,
    )

    # Find total protein counts
    total_count_protein = totalCountFromMassesAndRatios(
        total_mass_protein, individual_masses_protein, distribution_protein
    )

    return total_count_protein, ids_protein, distribution_protein


def totalCountIdDistributionRNA(sim_data, expression, doubling_time):
    """
    Calculates the total counts of RNA from their relative expression,
    individual mass, and total RNA mass. Relies on the math function
    totalCountFromMassesAndRatios.

    Inputs
    ------
    - expression (array of floats) - relative frequency distribution of RNA
            expression
    - doubling_time (float with units of time) - measured doubling time given
            the condition

    Returns
    --------
    - total_count_RNA (float) - total number of RNAs
    - ids_rnas (array of str) - name of each RNA with location tag
    - distribution_RNA (array of floats) - distribution for each RNA,
            normalized to 1
    """
    transcription = sim_data.process.transcription
    ids_rnas = transcription.rna_data["id"]
    total_mass_RNA = (
        sim_data.mass.get_component_masses(doubling_time)["rnaMass"]
        / sim_data.mass.avg_cell_to_initial_cell_conversion_factor
    )
    mws = transcription.rna_data["mw"]
    # Use only the rRNA/tRNA mass for rRNA/tRNA transcription units
    is_rRNA = transcription.rna_data["is_rRNA"]
    is_tRNA = transcription.rna_data["is_tRNA"]
    mws[is_rRNA] = transcription.rna_data["rRNA_mw"][is_rRNA]
    mws[is_tRNA] = transcription.rna_data["tRNA_mw"][is_tRNA]
    individual_masses_RNA = mws / sim_data.constants.n_avogadro

    distribution_RNA = normalize(expression)

    total_count_RNA = totalCountFromMassesAndRatios(
        total_mass_RNA, individual_masses_RNA, distribution_RNA
    )

    return total_count_RNA, ids_rnas, distribution_RNA


def createBulkContainer(sim_data, expression, doubling_time):
    """
    Creates a container that tracks the counts of all bulk molecules. Relies on
    totalCountIdDistributionRNA and totalCountIdDistributionProtein to set the
    counts and IDs of all RNAs and proteins.

    Inputs
    ------
    - expression (array of floats) - relative frequency distribution of RNA expression
    - doubling_time (float with units of time) - measured doubling time given the condition

    Returns
    -------
    - bulkContainer (np.ndarray object) - Two columns: 'id' for name and 'count'
            for count of all bulk molecules
    """

    total_count_RNA, ids_rnas, distribution_RNA = totalCountIdDistributionRNA(
        sim_data, expression, doubling_time
    )
    total_count_protein, ids_protein, distribution_protein = (
        totalCountIdDistributionProtein(sim_data, expression, doubling_time)
    )

    ids_molecules = sim_data.internal_state.bulk_molecules.bulk_data["id"]

    # Construct bulk container
    bulkContainer = np.array(
        [mol_data for mol_data in zip(ids_molecules, np.zeros(len(ids_molecules)))],
        dtype=[("id", ids_molecules.dtype), ("count", np.float64)],
    )

    # Assign RNA counts based on mass and expression distribution
    counts_RNA = total_count_RNA * distribution_RNA
    rna_idx = bulk_name_to_idx(ids_rnas, bulkContainer["id"])
    bulkContainer["count"][rna_idx] = counts_RNA

    # Assign protein counts based on mass and mRNA counts
    counts_protein = total_count_protein * distribution_protein
    protein_idx = bulk_name_to_idx(ids_protein, bulkContainer["id"])
    bulkContainer["count"][protein_idx] = counts_protein

    return bulkContainer


def setRibosomeCountsConstrainedByPhysiology(
    sim_data, bulkContainer, doubling_time, variable_elongation_translation
):
    """
    Set counts of ribosomal protein subunits based on three constraints:
    (1) Expected protein distribution doubles in one cell cycle
    (2) Measured rRNA mass fractions
    (3) Expected ribosomal protein subunit counts based on RNA expression data

    Inputs
    ------
    bulkContainer (np.ndarray object) - Two columns: 'id' for name and 'count'
            for count of all bulk molecules
    doubling_time (float with units of time) - doubling time given the condition
    variable_elongation_translation (bool) - whether there is variable elongation for translation

    Modifies
    --------
    - counts of ribosomal protein subunits in bulkContainer
    """

    active_fraction = sim_data.growth_rate_parameters.get_fraction_active_ribosome(
        doubling_time
    )

    # Get IDs and stoichiometry of ribosome subunits
    ribosome_30S_subunits = sim_data.process.complexation.get_monomers(
        sim_data.molecule_ids.s30_full_complex
    )["subunitIds"]
    ribosome_50S_subunits = sim_data.process.complexation.get_monomers(
        sim_data.molecule_ids.s50_full_complex
    )["subunitIds"]
    ribosome_30S_stoich = sim_data.process.complexation.get_monomers(
        sim_data.molecule_ids.s30_full_complex
    )["subunitStoich"]
    ribosome_50S_stoich = sim_data.process.complexation.get_monomers(
        sim_data.molecule_ids.s50_full_complex
    )["subunitStoich"]

    # Remove rRNA subunits from each array
    monomer_ids = set(sim_data.process.translation.monomer_data["id"])

    def remove_rRNA(subunit_ids, subunit_stoich):
        is_protein = np.array(
            [(subunit_id in monomer_ids) for subunit_id in subunit_ids]
        )
        return (subunit_ids[is_protein], subunit_stoich[is_protein])

    ribosome_30S_subunits, ribosome_30S_stoich = remove_rRNA(
        ribosome_30S_subunits, ribosome_30S_stoich
    )
    ribosome_50S_subunits, ribosome_50S_stoich = remove_rRNA(
        ribosome_50S_subunits, ribosome_50S_stoich
    )

    # -- CONSTRAINT 1: Expected protein distribution doubling -- #
    ## Calculate minimium number of 30S and 50S subunits required in order to double our expected
    ## protein distribution in one cell cycle
    proteinLengths = units.sum(
        sim_data.process.translation.monomer_data["aa_counts"], axis=1
    )
    proteinDegradationRates = sim_data.process.translation.monomer_data["deg_rate"]
    protein_idx = bulk_name_to_idx(
        sim_data.process.translation.monomer_data["id"], bulkContainer["id"]
    )
    proteinCounts = counts(bulkContainer, protein_idx)

    netLossRate_protein = netLossRateFromDilutionAndDegradationProtein(
        doubling_time,
        proteinDegradationRates,
    )

    elongation_rates = sim_data.process.translation.make_elongation_rates(
        None,
        sim_data.growth_rate_parameters.get_ribosome_elongation_rate(
            doubling_time
        ).asNumber(units.aa / units.s),
        1,
        variable_elongation_translation,
    )

    nRibosomesNeeded = np.ceil(
        calculateMinPolymerizingEnzymeByProductDistribution(
            proteinLengths, elongation_rates, netLossRate_protein, proteinCounts
        ).asNumber(units.aa / units.s)
        / active_fraction
    )

    # Minimum number of ribosomes needed
    constraint1_ribosome30SCounts = nRibosomesNeeded * ribosome_30S_stoich

    constraint1_ribosome50SCounts = nRibosomesNeeded * ribosome_50S_stoich

    # -- CONSTRAINT 2: Measured rRNA mass fraction -- #
    # Get rRNA counts
    rna_data = sim_data.process.transcription.rna_data
    rrna_idx = bulk_name_to_idx(
        rna_data["id"][rna_data["is_rRNA"]], bulkContainer["id"]
    )
    rRNA_tu_counts = counts(bulkContainer, rrna_idx)
    rRNA_cistron_counts = (
        sim_data.process.transcription.rRNA_cistron_tu_mapping_matrix.dot(
            rRNA_tu_counts
        )
    )
    rRNA_cistron_indexes = np.where(
        sim_data.process.transcription.cistron_data["is_rRNA"]
    )[0]
    rRNA_23S_counts = rRNA_cistron_counts[
        sim_data.process.transcription.cistron_data["is_23S_rRNA"][rRNA_cistron_indexes]
    ]
    rRNA_16S_counts = rRNA_cistron_counts[
        sim_data.process.transcription.cistron_data["is_16S_rRNA"][rRNA_cistron_indexes]
    ]
    rRNA_5S_counts = rRNA_cistron_counts[
        sim_data.process.transcription.cistron_data["is_5S_rRNA"][rRNA_cistron_indexes]
    ]

    ## 16S rRNA is in the 30S subunit
    massFracPredicted_30SCount = rRNA_16S_counts.sum()
    ## 23S and 5S rRNA are in the 50S subunit
    massFracPredicted_50SCount = min(rRNA_23S_counts.sum(), rRNA_5S_counts.sum())

    constraint2_ribosome30SCounts = massFracPredicted_30SCount * ribosome_30S_stoich
    constraint2_ribosome50SCounts = massFracPredicted_50SCount * ribosome_50S_stoich

    # -- CONSTRAINT 3: Expected ribosomal subunit counts based expression
    ## Calculate fundamental ribosomal subunit count distribution based on RNA expression data
    ## Already calculated and stored in bulkContainer
    ribosome_30S_idx = bulk_name_to_idx(ribosome_30S_subunits, bulkContainer["id"])
    ribosome30SCounts = counts(bulkContainer, ribosome_30S_idx)
    ribosome_50S_idx = bulk_name_to_idx(ribosome_50S_subunits, bulkContainer["id"])
    ribosome50SCounts = counts(bulkContainer, ribosome_50S_idx)

    # -- SET RIBOSOME FUNDAMENTAL SUBUNIT COUNTS TO MAXIMUM CONSTRAINT -- #
    constraint_names = np.array(
        [
            "Insufficient to double protein counts",
            "Too small for mass fraction",
            "Current level OK",
        ]
    )
    rib30lims = np.array(
        [
            nRibosomesNeeded,
            massFracPredicted_30SCount,
            (ribosome30SCounts / ribosome_30S_stoich).min(),
        ]
    )
    rib50lims = np.array(
        [
            nRibosomesNeeded,
            massFracPredicted_50SCount,
            (ribosome50SCounts / ribosome_50S_stoich).min(),
        ]
    )
    if VERBOSE > 1:
        print(
            "30S limit: {}".format(
                constraint_names[np.where(rib30lims.max() == rib30lims)[0]][-1]
            )
        )
        print(
            "30S actual count: {}".format(
                (ribosome30SCounts / ribosome_30S_stoich).min()
            )
        )
        print(
            "30S count set to: {}".format(
                rib30lims[np.where(rib30lims.max() == rib30lims)[0]][-1]
            )
        )
        print(
            "50S limit: {}".format(
                constraint_names[np.where(rib50lims.max() == rib50lims)[0]][-1]
            )
        )
        print(
            "50S actual count: {}".format(
                (ribosome50SCounts / ribosome_50S_stoich).min()
            )
        )
        print(
            "50S count set to: {}".format(
                rib50lims[np.where(rib50lims.max() == rib50lims)[0]][-1]
            )
        )

    bulkContainer["count"][ribosome_30S_idx] = np.fmax(
        np.fmax(ribosome30SCounts, constraint1_ribosome30SCounts),
        constraint2_ribosome30SCounts,
    )

    bulkContainer["count"][ribosome_50S_idx] = np.fmax(
        np.fmax(ribosome50SCounts, constraint1_ribosome50SCounts),
        constraint2_ribosome50SCounts,
    )


def setRNAPCountsConstrainedByPhysiology(
    sim_data,
    bulkContainer,
    doubling_time,
    avgCellDryMassInit,
    variable_elongation_transcription,
    Km=None,
):
    """
    Set counts of RNA polymerase based on two constraints:
    (1) Number of RNAP subunits required to maintain steady state of mRNAs
    (2) Expected RNAP subunit counts based on (mRNA) distribution recorded in
            bulkContainer

    Inputs
    ------
    - bulkContainer (np.ndarray object) - Two columns: 'id' for name and 'count'
            for count of all bulk molecules
    - doubling_time (float with units of time) - doubling time given the condition
    - avgCellDryMassInit (float with units of mass) - expected initial dry cell mass
    - Km (array of floats with units of mol/volume) - Km for each RNA associated
    with RNases

    Modifies
    --------
    - bulkContainer (np.ndarray object) - the counts of RNA polymerase
            subunits are set according to Constraint 1

    Notes
    -----
    - Constraint 2 is not being used -- see final line of this function.
    """

    # -- CONSTRAINT 1: Expected RNA distribution doubling -- #
    rnaLengths = units.sum(
        sim_data.process.transcription.rna_data["counts_ACGU"], axis=1
    )

    rnaLossRate = None
    rna_idx = bulk_name_to_idx(
        sim_data.process.transcription.rna_data["id"], bulkContainer["id"]
    )

    if Km is None:
        # RNA loss rate is in units of counts/time, and computed by summing the
        # contributions of degradation and dilution.
        rnaLossRate = netLossRateFromDilutionAndDegradationRNALinear(
            doubling_time,
            sim_data.process.transcription.rna_data["deg_rate"],
            counts(bulkContainer, rna_idx),
        )
    else:
        # Get constants to compute countsToMolar factor
        cellDensity = sim_data.constants.cell_density
        cellVolume = (
            avgCellDryMassInit / cellDensity / sim_data.mass.cell_dry_mass_fraction
        )
        countsToMolar = 1 / (sim_data.constants.n_avogadro * cellVolume)

        # Gompute input arguments for netLossRateFromDilutionAndDegradationRNA()
        rnaConc = countsToMolar * counts(bulkContainer, rna_idx)
        endoRNase_idx = bulk_name_to_idx(
            sim_data.process.rna_decay.endoRNase_ids, bulkContainer["id"]
        )
        endoRNaseConc = countsToMolar * counts(bulkContainer, endoRNase_idx)
        kcatEndoRNase = sim_data.process.rna_decay.kcats
        totalEndoRnaseCapacity = units.sum(endoRNaseConc * kcatEndoRNase)

        # RNA loss rate is in units of counts/time, and computed by accounting
        # for the competitive inhibition of RNase by other RNA targets.
        rnaLossRate = netLossRateFromDilutionAndDegradationRNA(
            doubling_time,
            (1 / countsToMolar) * totalEndoRnaseCapacity,
            Km,
            rnaConc,
            countsToMolar,
        )

    # Compute number of RNA polymerases required to maintain steady state of mRNA
    elongation_rates = sim_data.process.transcription.make_elongation_rates(
        None,
        sim_data.growth_rate_parameters.get_rnap_elongation_rate(
            doubling_time
        ).asNumber(units.nt / units.s),
        1,
        variable_elongation_transcription,
    )

    nActiveRnapNeeded = calculateMinPolymerizingEnzymeByProductDistributionRNA(
        rnaLengths, elongation_rates, rnaLossRate
    ).asNumber(units.nt / units.s)

    nRnapsNeeded = np.ceil(
        nActiveRnapNeeded
        / sim_data.growth_rate_parameters.get_fraction_active_rnap(doubling_time)
    )

    # Convert nRnapsNeeded to the number of RNA polymerase subunits required
    rnapIds = sim_data.process.complexation.get_monomers(
        sim_data.molecule_ids.full_RNAP
    )["subunitIds"]
    rnapStoich = sim_data.process.complexation.get_monomers(
        sim_data.molecule_ids.full_RNAP
    )["subunitStoich"]
    minRnapSubunitCounts = nRnapsNeeded * rnapStoich

    # -- CONSTRAINT 2: Expected RNAP subunit counts based on distribution -- #
    rnap_idx = bulk_name_to_idx(rnapIds, bulkContainer["id"])
    rnapCounts = counts(bulkContainer, rnap_idx)

    ## -- SET RNAP COUNTS TO MAXIMUM CONSTRAINTS -- #
    constraint_names = np.array(
        ["Current level OK", "Insufficient to double RNA distribution"]
    )
    rnapLims = np.array(
        [(rnapCounts / rnapStoich).min(), (minRnapSubunitCounts / rnapStoich).min()]
    )
    if VERBOSE > 1:
        print(
            "rnap limit: {}".format(
                constraint_names[np.where(rnapLims.max() == rnapLims)[0]][0]
            )
        )
        print("rnap actual count: {}".format((rnapCounts / rnapStoich).min()))
        print(
            "rnap counts set to: {}".format(
                rnapLims[np.where(rnapLims.max() == rnapLims)[0]][0]
            )
        )

    if np.any(minRnapSubunitCounts < 0):
        raise ValueError("RNAP protein counts must be positive.")

    bulkContainer["count"][rnap_idx] = minRnapSubunitCounts


def fitExpression(sim_data, bulkContainer, doubling_time, avgCellDryMassInit, Km=None):
    """
    Determines expression and synthesis probabilities for RNA molecules to fit
    protein levels and RNA degradation rates. Assumes a steady state analysis
    where the RNA synthesis probability will be the same as the degradation rate.
    If no Km is given, then RNA degradation is assumed to be linear otherwise
    degradation is calculated based on saturation with RNases.

    Inputs
    ------
    - bulkContainer (np.ndarray object) - Two columns: 'id' for name and 'count'
            for expected count based on expression of all bulk molecules
    - doubling_time (float with units of time) - doubling time
    - avgCellDryMassInit (float with units of mass) - expected initial dry cell mass
    - Km (array of floats with units of mol/volume) - Km for each RNA associated
    with RNases

    Modifies
    --------
    - bulkContainer counts of RNA and proteins

    Returns
    --------
    - expression (array of floats) - adjusted expression for each RNA,
    normalized to 1
    - synth_prob (array of floats) - synthesis probability for each RNA which
    accounts for expression and degradation rate, normalized to 1
    - fit_cistron_expression (array of floats) - target expression levels of
    each cistron (gene) used to calculate RNA expression levels
    - cistron_expression_res (array of floats) - the residuals of the NNLS
    problem solved to calculate RNA expression levels

    Notes
    -----
    - TODO - sets bulkContainer counts and returns values - change to only return values
    """
    # Load required parameters
    transcription = sim_data.process.transcription
    translation = sim_data.process.translation
    translation_efficiencies_by_protein = normalize(
        translation.translation_efficiencies_by_monomer
    )
    degradation_rates_protein = translation.monomer_data["deg_rate"]
    net_loss_rate_protein = netLossRateFromDilutionAndDegradationProtein(
        doubling_time, degradation_rates_protein
    )
    avg_cell_fraction_mass = sim_data.mass.get_component_masses(doubling_time)
    total_mass_RNA = (
        avg_cell_fraction_mass["rnaMass"]
        / sim_data.mass.avg_cell_to_initial_cell_conversion_factor
    )
    cistron_tu_mapping_matrix = transcription.cistron_tu_mapping_matrix

    # Calculate current expression fraction of mRNA transcription units
    rna_idx = bulk_name_to_idx(transcription.rna_data["id"], bulkContainer["id"])
    RNA_counts = counts(bulkContainer, rna_idx)
    rna_expression_container = normalize(RNA_counts)

    mRNA_tu_expression_frac = np.sum(
        rna_expression_container[transcription.rna_data["is_mRNA"]]
    )

    # Calculate current expression levels of each cistron given the RNA
    # expression levels
    fit_cistron_expression = normalize(cistron_tu_mapping_matrix.dot(RNA_counts))
    mRNA_cistron_expression_frac = fit_cistron_expression[
        transcription.cistron_data["is_mRNA"]
    ].sum()

    # Calculate required mRNA expression from monomer counts
    protein_idx = bulk_name_to_idx(translation.monomer_data["id"], bulkContainer["id"])
    counts_protein = counts(bulkContainer, protein_idx)
    mRNA_cistron_distribution_per_protein = mRNADistributionFromProtein(
        normalize(counts_protein),
        translation_efficiencies_by_protein,
        net_loss_rate_protein,
    )

    mRNA_cistron_distribution = normalize(
        sim_data.relation.monomer_to_mRNA_cistron_mapping().T.dot(
            mRNA_cistron_distribution_per_protein
        )
    )

    # Replace mRNA cistron expression with values calculated from monomer counts
    fit_cistron_expression[transcription.cistron_data["is_mRNA"]] = (
        mRNA_cistron_expression_frac * mRNA_cistron_distribution
    )

    # Use least squares to calculate expression of transcription units required
    # to generate the given cistron expression levels and the residuals for
    # the expression of each cistron
    fit_tu_expression, cistron_expression_res = transcription.fit_rna_expression(
        fit_cistron_expression
    )
    fit_mRNA_tu_expression = fit_tu_expression[transcription.rna_data["is_mRNA"]]

    rna_expression_container[transcription.rna_data["is_mRNA"]] = (
        mRNA_tu_expression_frac * normalize(fit_mRNA_tu_expression)
    )
    expression = normalize(rna_expression_container)

    # Set number of RNAs based on expression we just set
    mws = transcription.rna_data["mw"]

    # Use only the rRNA/tRNA mass for rRNA/tRNA transcription units
    is_rRNA = transcription.rna_data["is_rRNA"]
    is_tRNA = transcription.rna_data["is_tRNA"]
    mws[is_rRNA] = transcription.rna_data["rRNA_mw"][is_rRNA]
    mws[is_tRNA] = transcription.rna_data["tRNA_mw"][is_tRNA]

    n_rnas = totalCountFromMassesAndRatios(
        total_mass_RNA, mws / sim_data.constants.n_avogadro, expression
    )
    bulkContainer["count"][rna_idx] = n_rnas * expression

    if Km is None:
        rnaLossRate = netLossRateFromDilutionAndDegradationRNALinear(
            doubling_time,
            transcription.rna_data["deg_rate"],
            counts(bulkContainer, rna_idx),
        )
    else:
        # Get constants to compute countsToMolar factor
        cellDensity = sim_data.constants.cell_density
        dryMassFraction = sim_data.mass.cell_dry_mass_fraction
        cellVolume = avgCellDryMassInit / cellDensity / dryMassFraction
        countsToMolar = 1 / (sim_data.constants.n_avogadro * cellVolume)

        endoRNase_idx = bulk_name_to_idx(
            sim_data.process.rna_decay.endoRNase_ids, bulkContainer["id"]
        )
        endoRNaseConc = countsToMolar * counts(bulkContainer, endoRNase_idx)
        kcatEndoRNase = sim_data.process.rna_decay.kcats
        totalEndoRnaseCapacity = units.sum(endoRNaseConc * kcatEndoRNase)

        rnaLossRate = netLossRateFromDilutionAndDegradationRNA(
            doubling_time,
            (1 / countsToMolar) * totalEndoRnaseCapacity,
            Km,
            countsToMolar * counts(bulkContainer, rna_idx),
            countsToMolar,
        )

    synth_prob = normalize(rnaLossRate.asNumber(1 / units.min))

    return expression, synth_prob, fit_cistron_expression, cistron_expression_res


def fitMaintenanceCosts(sim_data, bulkContainer):
    """
    Fits the growth-associated maintenance (GAM) cost associated with metabolism.

    The energetic costs associated with growth have been estimated utilizing flux-balance analysis
    and are used with FBA to obtain accurate growth predictions.  In the whole-cell model, some of
    these costs are explicitly associated with the energetic costs of translation, a biomass
    assembly process.  Consequently we must estimate the amount of energy utilized by translation
    per unit of biomass (i.e. dry mass) produced, and subtract that quantity from reported GAM to
    acquire the modified GAM that we use in the metabolic submodel.

    Requires
    --------
    - amino acid counts associated with protein monomers
    - average initial dry mass
    - energetic (GTP) cost of translation (per amino acid polymerized)
    - observed growth-associated maintenance (GAM)
    In dimensions of ATP or ATP equivalents consumed per biomass

    Modifies
    --------
    - the "dark" ATP, i.e. the modified GAM

    Notes
    -----
    As more non-metabolic submodels account for energetic costs, this function should be extended
    to subtract those costs off the observed GAM.

    There also exists, in contrast, non-growth-associated-maintenance (NGAM), which is relative to
    total biomass rather than the biomass accumulation rate.  As the name would imply, this
    accounts for the energetic costs of maintaining the existing biomass.  It is also accounted for
    in the metabolic submodel.

    TODO (John): Rewrite as a true function.
    """

    aaCounts = sim_data.process.translation.monomer_data["aa_counts"]
    protein_idx = bulk_name_to_idx(
        sim_data.process.translation.monomer_data["id"], bulkContainer["id"]
    )
    proteinCounts = counts(bulkContainer, protein_idx)
    nAvogadro = sim_data.constants.n_avogadro
    avgCellDryMassInit = sim_data.mass.avg_cell_dry_mass_init
    gtpPerTranslation = sim_data.constants.gtp_per_translation
    atp_per_charge = (
        2  # ATP -> AMP is explicitly used in charging reactions so can remove from GAM
    )

    # GTPs used for translation (recycled, not incorporated into biomass)
    aaMmolPerGDCW = units.sum(
        aaCounts * np.tile(proteinCounts.reshape(-1, 1), (1, 21)), axis=0
    ) * ((1 / (units.aa * nAvogadro)) * (1 / avgCellDryMassInit))

    aasUsedOverCellCycle = units.sum(aaMmolPerGDCW)
    explicit_mmol_maintenance_per_gdcw = (
        atp_per_charge + gtpPerTranslation
    ) * aasUsedOverCellCycle

    darkATP = (  # This has everything we can't account for
        sim_data.constants.growth_associated_maintenance
        - explicit_mmol_maintenance_per_gdcw
    )

    # We do not want to create energy with growth by having a negative darkATP
    # value. GAM measurements have some error so it's possible explicit
    # accounting could be more accurate or the GAM value used is too low which
    # would lead to a negative value. Easy fix is setting darkATP = 0 if this
    # error is raised.
    if darkATP.asNumber() < 0:
        raise ValueError(
            "GAM has been adjusted too low. Explicit energy accounting should not exceed GAM."
            " Consider setting darkATP to 0 if energy corrections are accurate."
        )

    sim_data.constants.darkATP = darkATP


def calculateBulkDistributions(
    sim_data, expression, concDict, avgCellDryMassInit, doubling_time
):
    """
    Finds a distribution of copy numbers for macromolecules. While RNA and protein
    expression can be approximated using well-described statistical	distributions,
    complexes require absolute copy numbers. To get these distributions, this
    function instantiates many cells with a reduced set of molecules, forms complexes,
    and iterates through equilibrium and two-component system processes until
    metabolite counts reach a steady-state. It then computes the resulting
    statistical distributions.

    Requires
    --------
    - N_SEEDS (int) - the number of instantiated cells

    Inputs
    ------
    - expression (array of floats) - expression for each RNA, normalized to 1
    - concDict {metabolite (str): concentration (float with units of mol/volume)} -
    dictionary for concentrations of each metabolite with location tag
    - avgCellDryMassInit (float with units of mass) - initial dry cell mass
    - doubling_time (float with units of time) - doubling time for condition

    Returns
    --------
    - bulkAverageContainer (np.ndarray object) - Two columns: 'id' for name and 'count'
            for the mean of the counts of all bulk molecules
    - bulkDeviationContainer (np.ndarray object) - Two columns: 'id' for name and 'count'
            for the standard deviation of the counts of all bulk molecules
    - proteinMonomerAverageContainer (np.ndarray object) - Two columns: 'id' for name and 'count'
            for the mean of the counts of all protein monomers
    - proteinMonomerDeviationContainer (np.ndarray object) - Two columns: 'id' for name and 'count'
            for the standard deviation of the counts of all protein monomers
    """

    # Ids
    totalCount_RNA, ids_rnas, distribution_RNA = totalCountIdDistributionRNA(
        sim_data, expression, doubling_time
    )
    totalCount_protein, ids_protein, distribution_protein = (
        totalCountIdDistributionProtein(sim_data, expression, doubling_time)
    )
    ids_complex = sim_data.process.complexation.molecule_names
    ids_equilibrium = sim_data.process.equilibrium.molecule_names
    ids_twoComponentSystem = sim_data.process.two_component_system.molecule_names
    ids_metabolites = sorted(concDict)
    conc_metabolites = (units.mol / units.L) * np.array(
        [concDict[key].asNumber(units.mol / units.L) for key in ids_metabolites]
    )
    allMoleculesIDs = sorted(
        set(ids_rnas)
        | set(ids_protein)
        | set(ids_complex)
        | set(ids_equilibrium)
        | set(ids_twoComponentSystem)
        | set(ids_metabolites)
    )

    # Data for complexation
    complexationStoichMatrix = sim_data.process.complexation.stoich_matrix().astype(
        np.int64, order="F"
    )
    # Data for equilibrium binding
    # equilibriumDerivatives = sim_data.process.equilibrium.derivatives
    # equilibriumDerivativesJacobian = sim_data.process.equilibrium.derivativesJacobian

    # Data for metabolites
    cellDensity = sim_data.constants.cell_density
    cellVolume = avgCellDryMassInit / cellDensity / sim_data.mass.cell_dry_mass_fraction

    # Construct bulk container

    # We want to know something about the distribution of the copy numbers of
    # macromolecules in the cell.  While RNA and protein expression can be
    # approximated using well-described statistical distributions, we need
    # absolute copy numbers to form complexes.  To get a distribution, we must
    # instantiate many cells, form complexes, and finally compute the
    # statistics we will use in the fitting operations.

    bulk_ids = sim_data.internal_state.bulk_molecules.bulk_data.struct_array["id"]
    bulkContainer = np.array(
        [mol_data for mol_data in zip(bulk_ids, np.zeros(len(bulk_ids)))],
        dtype=[("id", bulk_ids.dtype), ("count", int)],
    )

    rna_idx = bulk_name_to_idx(ids_rnas, bulkContainer["id"])
    protein_idx = bulk_name_to_idx(ids_protein, bulkContainer["id"])
    complexation_molecules_idx = bulk_name_to_idx(ids_complex, bulkContainer["id"])
    equilibrium_molecules_idx = bulk_name_to_idx(ids_equilibrium, bulkContainer["id"])
    two_component_system_molecules_idx = bulk_name_to_idx(
        ids_twoComponentSystem, bulkContainer["id"]
    )
    metabolites_idx = bulk_name_to_idx(ids_metabolites, bulkContainer["id"])
    all_molecules_idx = bulk_name_to_idx(allMoleculesIDs, bulkContainer["id"])

    allMoleculeCounts = np.empty((N_SEEDS, len(allMoleculesIDs)), np.int64)
    proteinMonomerCounts = np.empty((N_SEEDS, len(ids_protein)), np.int64)

    if VERBOSE > 1:
        print("Bulk distribution seed:")

    # Instantiate cells to find average copy numbers of macromolecules
    for seed in range(N_SEEDS):
        if VERBOSE > 1:
            print("seed = {}".format(seed))

        bulkContainer["count"][all_molecules_idx] = 0

        bulkContainer["count"][rna_idx] = totalCount_RNA * distribution_RNA

        bulkContainer["count"][protein_idx] = totalCount_protein * distribution_protein

        proteinMonomerCounts[seed, :] = counts(bulkContainer, protein_idx)
        complexationMoleculeCounts = counts(bulkContainer, complexation_molecules_idx)

        # Form complexes
        time_step = 2**31  # don't stop until all complexes are formed.
        complexation_rates = sim_data.process.complexation.rates
        system = StochasticSystem(complexationStoichMatrix.T, random_seed=seed)
        complexation_result = system.evolve(
            time_step, complexationMoleculeCounts, complexation_rates
        )

        updatedCompMoleculeCounts = complexation_result["outcome"]
        bulkContainer["count"][complexation_molecules_idx] = updatedCompMoleculeCounts

        metDiffs = np.inf * np.ones_like(counts(bulkContainer, metabolites_idx))
        nIters = 0

        # Iterate processes until metabolites converge to a steady-state
        while np.linalg.norm(metDiffs, np.inf) > 1:
            random_state = np.random.RandomState(seed)
            metCounts = conc_metabolites * cellVolume * sim_data.constants.n_avogadro
            metCounts.normalize()
            metCounts.checkNoUnit()
            bulkContainer["count"][metabolites_idx] = metCounts.asNumber().round()

            # Find reaction fluxes from equilibrium process
            # Do not use jit to avoid compiling time (especially when running
            # in parallel since sim_data needs to be pickled and reconstructed
            # each time)
            rxnFluxes, _ = sim_data.process.equilibrium.fluxes_and_molecules_to_SS(
                bulkContainer["count"][equilibrium_molecules_idx],
                cellVolume.asNumber(units.L),
                sim_data.constants.n_avogadro.asNumber(1 / units.mol),
                random_state,
                jit=False,
            )
            bulkContainer["count"][equilibrium_molecules_idx] += np.dot(
                sim_data.process.equilibrium.stoich_matrix().astype(np.int64),
                rxnFluxes.astype(np.int64),
            )
            assert np.all(bulkContainer["count"][equilibrium_molecules_idx] >= 0)

            # Find changes from two component system
            _, moleculeCountChanges = (
                sim_data.process.two_component_system.molecules_to_ss(
                    bulkContainer["count"][two_component_system_molecules_idx],
                    cellVolume.asNumber(units.L),
                    sim_data.constants.n_avogadro.asNumber(1 / units.mmol),
                )
            )

            bulkContainer["count"][two_component_system_molecules_idx] += (
                moleculeCountChanges.astype(np.int64)
            )

            metDiffs = (
                bulkContainer["count"][metabolites_idx] - metCounts.asNumber().round()
            )

            nIters += 1
            if nIters > 100:
                raise Exception("Equilibrium reactions are not converging!")

        allMoleculeCounts[seed, :] = counts(bulkContainer, all_molecules_idx)

    # Update counts in bulk objects container
    bulkAverageContainer = np.array(
        [mol_data for mol_data in zip(bulk_ids, np.zeros(len(bulk_ids)))],
        dtype=[("id", bulk_ids.dtype), ("count", np.float64)],
    )
    bulkDeviationContainer = np.array(
        [mol_data for mol_data in zip(bulk_ids, np.zeros(len(bulk_ids)))],
        dtype=[("id", bulk_ids.dtype), ("count", np.float64)],
    )
    monomer_ids = sim_data.process.translation.monomer_data["id"]
    proteinMonomerAverageContainer = np.array(
        [mol_data for mol_data in zip(monomer_ids, np.zeros(len(monomer_ids)))],
        dtype=[("id", monomer_ids.dtype), ("count", np.float64)],
    )
    proteinMonomerDeviationContainer = np.array(
        [mol_data for mol_data in zip(monomer_ids, np.zeros(len(monomer_ids)))],
        dtype=[("id", monomer_ids.dtype), ("count", np.float64)],
    )

    bulkAverageContainer["count"][all_molecules_idx] = allMoleculeCounts.mean(0)
    bulkDeviationContainer["count"][all_molecules_idx] = allMoleculeCounts.std(0)
    proteinMonomerAverageContainer["count"] = proteinMonomerCounts.mean(0)
    proteinMonomerDeviationContainer["count"] = proteinMonomerCounts.std(0)

    return (
        bulkAverageContainer,
        bulkDeviationContainer,
        proteinMonomerAverageContainer,
        proteinMonomerDeviationContainer,
    )


# Math functions


def totalCountFromMassesAndRatios(totalMass, individualMasses, distribution):
    """
    Function to determine the expected total counts for a group of molecules
    in order to achieve a total mass with a given distribution of individual
    molecules.

    Math:
            Total mass = dot(mass, count)

            Fraction of i:
            f = count / Total counts

            Substituting:
            Total mass = dot(mass, f * Total counts)
            Total mass = Total counts * dot(mass, f)

            Total counts = Total mass / dot(mass, f)

    Requires
    --------
    - totalMass (float with mass units): total mass of the group of molecules
    - individualMasses (array of floats with mass units): mass for individual
    molecules in the group
    - distribution (array of floats): distribution of individual molecules,
    normalized to 1

    Returns
    --------
    - counts (float): total counts (does not need to be a whole number)
    """

    assert np.allclose(np.sum(distribution), 1)
    counts = 1 / units.dot(individualMasses, distribution) * totalMass
    return units.strip_empty_units(counts)


def proteinDistributionFrommRNA(
    distribution_mRNA, translation_efficiencies, netLossRate
):
    """
    dP_i / dt = k * M_i * e_i - P_i * Loss_i

    At steady state:
    P_i = k * M_i * e_i / Loss_i

    Fraction of mRNA for ith gene is defined as:
    f_i = M_i / M_total

    Substituting in:
    P_i = k * f_i * e_i * M_total / Loss_i

    Normalizing P_i by summing over all i cancels out k and M_total
    assuming constant translation rate.

    Inputs
    ------
    - distribution_mRNA (array of floats) - distribution for each mRNA,
    normalized to 1
    - translation_efficiencies (array of floats) - translational efficiency for each mRNA,
    normalized to 1
    - netLossRate (array of floats with units of 1/time) - rate of loss for each protein

    Returns
    -------
    - array of floats for the distribution of each protein, normalized to 1
    """

    assert np.allclose(np.sum(distribution_mRNA), 1)
    assert np.allclose(np.sum(translation_efficiencies), 1)
    distributionUnnormed = (
        1 / netLossRate * distribution_mRNA * translation_efficiencies
    )
    distributionNormed = distributionUnnormed / units.sum(distributionUnnormed)
    distributionNormed.normalize()
    distributionNormed.checkNoUnit()

    return distributionNormed.asNumber()


def mRNADistributionFromProtein(
    distribution_protein, translation_efficiencies, netLossRate
):
    """
    dP_i / dt = k * M_i * e_i - P_i * Loss_i

    At steady state:
    M_i = Loss_i * P_i / (k * e_i)

    Fraction of protein for ith gene is defined as:
    f_i = P_i / P_total

    Substituting in:
    M_i = Loss_i * f_i * P_total / (k * e_i)

    Normalizing M_i by summing over all i cancels out k and P_total
    assuming a constant translation rate.

    Inputs
    ------
    - distribution_protein (array of floats) - distribution for each protein,
    normalized to 1
    - translation_efficiencies (array of floats) - translational efficiency for each mRNA,
    normalized to 1
    - netLossRate (array of floats with units of 1/time) - rate of loss for each protein

    Returns
    -------
    - array of floats for the distribution of each mRNA, normalized to 1
    """

    assert np.allclose(np.sum(distribution_protein), 1)
    distributionUnnormed = netLossRate * distribution_protein / translation_efficiencies
    distributionNormed = distributionUnnormed / units.sum(distributionUnnormed)
    distributionNormed.normalize()
    distributionNormed.checkNoUnit()

    return distributionNormed.asNumber()


def calculateMinPolymerizingEnzymeByProductDistribution(
    productLengths, elongationRates, netLossRate, productCounts
):
    """
    Compute the number of ribosomes required to maintain steady state.

    dP/dt = production rate - loss rate
    dP/dt = e_r * (1/L) * R - (k_loss * P)

    At steady state: dP/dt = 0
    R = sum over i ((L_i / e_r) * k_loss_i * P_i)

    Multiplying both sides by volume gives an equation in terms of counts.

    P = protein concentration
    e_r = polypeptide elongation rate per ribosome
    L = protein length
    R = ribosome concentration
    k_loss = net protein loss rate
    i = ith protein

    Inputs
    ------
    - productLengths (array of ints with units of amino_acids) - L, protein lengths
    - elongationRates (array of ints with units of amino_acid/time) e_r, polypeptide elongation rate
    - netLossRate (array of floats with units of 1/time) - k_loss, protein loss rate
    - productCounts (array of floats) - P, protein counts

    Returns
    --------
    - float with dimensionless units for the number of ribosomes required to
    maintain steady state
    """

    nPolymerizingEnzymeNeeded = units.sum(
        productLengths / elongationRates * netLossRate * productCounts
    )
    return nPolymerizingEnzymeNeeded


def calculateMinPolymerizingEnzymeByProductDistributionRNA(
    productLengths, elongationRates, netLossRate
):
    """
    Compute the number of RNA polymerases required to maintain steady state of mRNA.

    dR/dt = production rate - loss rate
    dR/dt = e_r * (1/L) * RNAp - k_loss

    At steady state: dR/dt = 0
    RNAp = sum over i ((L_i / e_r) * k_loss_i)

    Multiplying both sides by volume gives an equation in terms of counts.

    R = mRNA transcript concentration
    e_r = transcript elongation rate per RNAp
    L = transcript length
    RNAp = RNAp concentration
    k_loss = net transcript loss rate (unit: concentration / time)
    i = ith transcript

    Inputs
    ------
    - productLengths (array of ints with units of nucleotides) - L, transcript lengths
    - elongationRates (array of ints with units of nucleotide/time) - e_r, transcript elongation rate
    - netLossRate (array of floats with units of 1/time) - k_loss, transcript loss rate

    Returns
    -------
    - float with dimensionless units for the number of RNA polymerases required to
    maintain steady state
    """

    nPolymerizingEnzymeNeeded = units.sum(
        productLengths / elongationRates * netLossRate
    )
    return nPolymerizingEnzymeNeeded


def netLossRateFromDilutionAndDegradationProtein(doublingTime, degradationRates):
    """
    Compute total loss rate (summed contributions of degradation and dilution).

    Inputs
    ------
    - doublingTime (float with units of time) - doubling time of the cell
    - degradationRates (array of floats with units of 1/time) - protein degradation rate

    Returns
    --------
    - array of floats with units of 1/time for the total loss rate for each protein
    """

    return np.log(2) / doublingTime + degradationRates


def netLossRateFromDilutionAndDegradationRNA(
    doublingTime, totalEndoRnaseCountsCapacity, Km, rnaConc, countsToMolar
):
    """
    Compute total loss rate (summed impact of degradation and dilution).
    Returns the loss rate in units of (counts/time) in preparation for use in
    the steady state analysis in fitExpression() and
    setRNAPCountsConstrainedByPhysiology()
    (see calculateMinPolymerizingEnzymeByProductDistributionRNA()).

    Derived from steady state analysis of Michaelis-Menten enzyme kinetics with
    competitive inhibition: for a given RNA, all other RNAs compete for RNase.

    V_i = k_cat * [ES_i]
    v_i = k_cat * [E]0 * ([S_i]/Km_i) / (1 + sum over j genes([S_j] / Km_j))

    Inputs
    ------
    - doublingTime (float with units of time) - doubling time of the cell
    - totalEndoRnaseCountsCapacity (float with units of 1/time) total kinetic
    capacity of all RNases in the cell
    - Km (array of floats with units of mol/volume) - Michaelis-Menten constant
    for each RNA
    - rnaConc (array of floats with units of mol/volume) - concentration for each RNA
    - countsToMolar (float with units of mol/volume) - conversion between counts and molar

    Returns
    --------
    - array of floats with units of 1/time for the total loss rate for each RNA
    """

    fracSaturated = rnaConc / Km / (1 + units.sum(rnaConc / Km))
    rnaCounts = (1 / countsToMolar) * rnaConc

    return (np.log(2) / doublingTime) * rnaCounts + (
        totalEndoRnaseCountsCapacity * fracSaturated
    )


def netLossRateFromDilutionAndDegradationRNALinear(
    doublingTime, degradationRates, rnaCounts
):
    """
    Compute total loss rate (summed contributions of degradation and dilution).
    Returns the loss rate in units of (counts/time) in preparation for use in
    the steady state analysis in fitExpression() and
    setRNAPCountsConstrainedByPhysiology()
    (see calculateMinPolymerizingEnzymeByProductDistributionRNA()).

    Requires
    --------
    - doublingTime (float with units of time) - doubling time of the cell
    - degradationRates (array of floats with units of 1/time) - degradation rate
    for each RNA
    - rnaCounts (array of floats) - counts for each RNA

    Returns
    --------
    - array of floats with units of 1/time for the total loss rate for each RNA
    """

    return (np.log(2) / doublingTime + degradationRates) * rnaCounts


def expressionFromConditionAndFoldChange(transcription, condPerturbations, tfFCs):
    """
    Adjusts expression of RNA based on fold changes from basal for a given
    condition. Since fold changes are reported for individual RNA cistrons, the
    changes are applied to the basal expression levels of each cistron and the
    resulting vector is mapped back to RNA expression through nonnegative least
    squares. For genotype perturbations, the expression of all RNAs that include
    the given cistron are set to the given value.

    Inputs
    ------
    - transcription: Instance of the Transcription class from
            reconstruction.ecoli.dataclasses.process.transcription
    - condPerturbations {cistron ID (str): fold change (float)} -
            dictionary of fold changes for cistrons based on the given condition
    - tfFCs {cistron ID (str): fold change (float)} -
            dictionary of fold changes for cistrons based on transcription factors
            in the given condition

    Returns
    --------
    - expression (array of floats) - adjusted expression for each RNA,
    normalized to 1

    Notes
    -----
    - TODO (Travis) - Might not properly handle if an RNA is adjusted from both a
    perturbation and a transcription factor, currently RNA self regulation is not
    included in tfFCs
    """
    cistron_ids = transcription.cistron_data["id"]
    cistron_expression = transcription.fit_cistron_expression["basal"].copy()

    # Gather indices and fold changes for each cistron that will be adjusted
    cistron_id_to_index = {cistron_id: i for (i, cistron_id) in enumerate(cistron_ids)}
    cistron_indexes = []
    cistron_fcs = []

    # Compile indexes and fold changes of each cistron
    for cistron_id, fc_value in tfFCs.items():
        if cistron_id in condPerturbations:
            continue
        cistron_indexes.append(cistron_id_to_index[cistron_id])
        cistron_fcs.append(fc_value)

    def apply_fcs_to_expression(expression, indexes, fcs):
        """
        Applys the fold-change values to an expression vector while keeping the
        sum of expression values at one.

        Args:
                expression (np.ndarray of floats): Original expression vector of
                        cistrons or RNAs
                indexes (List of floats): Indexes of cistrons/RNAs that the
                        fold-changes should be applied to
                fcs (List of floats): Fold-changes of cistron/RNA expression
        """
        fcs = [fc for (idx, fc) in sorted(zip(indexes, fcs), key=lambda pair: pair[0])]
        indexes = [
            idx for (idx, fc) in sorted(zip(indexes, fcs), key=lambda pair: pair[0])
        ]

        # Adjust expression based on fold change and normalize
        indexes_bool = np.zeros(len(expression), dtype=bool)
        indexes_bool[indexes] = 1
        fcs = np.array(fcs)
        scaleTheRestBy = (1.0 - (expression[indexes] * fcs).sum()) / (
            1.0 - (expression[indexes]).sum()
        )
        expression[indexes_bool] *= fcs
        expression[~indexes_bool] *= scaleTheRestBy

        return expression

    cistron_expression = apply_fcs_to_expression(
        cistron_expression, cistron_indexes, cistron_fcs
    )

    # Use NNLS to map new cistron expression to RNA expression
    expression, _ = transcription.fit_rna_expression(cistron_expression)
    expression = normalize(expression)

    # Apply genotype perturbations to all RNAs that contain each cistron
    rna_indexes = []
    rna_fcs = []
    cistron_perturbation_indexes = []
    cistron_perturbation_values = []

    for cistron_id, perturbation_value in condPerturbations.items():
        rna_indexes_with_cistron = transcription.cistron_id_to_rna_indexes(cistron_id)
        rna_indexes.extend(rna_indexes_with_cistron)
        rna_fcs.extend([perturbation_value] * len(rna_indexes_with_cistron))
        cistron_perturbation_indexes.append(cistron_id_to_index[cistron_id])
        cistron_perturbation_values.append(perturbation_value)

    expression = apply_fcs_to_expression(expression, rna_indexes, rna_fcs)
    # Also apply perturbations to cistrons for bookkeeping purposes
    cistron_expression = apply_fcs_to_expression(
        cistron_expression, cistron_perturbation_indexes, cistron_perturbation_values
    )

    return expression, cistron_expression


def crc32(*arrays: np.ndarray, initial: int = 0) -> int:
    """Return a CRC32 checksum of the given ndarrays."""

    def crc_next(initial: int, array: np.ndarray) -> int:
        shape = str(array.shape).encode()
        values = array.tobytes()
        return binascii.crc32(values, binascii.crc32(shape, initial))

    return functools.reduce(crc_next, arrays, initial)


def setKmCooperativeEndoRNonLinearRNAdecay(sim_data, bulkContainer, cache_dir):
    """
    Fits the affinities (Michaelis-Menten constants) for RNAs binding to
    endoRNAses.

    EndoRNAses perform the first step of RNA decay by cleaving a whole RNA
    somewhere inside its extent.  This results in RNA fragments, which are then
    digested into monomers by exoRNAses. To model endoRNAse activity, we need to
    determine an affinity (Michaelis-Menten constant) for each RNA that is
    consistent with experimentally observed half-lives.  The Michaelis-Menten
    constants must be determined simultaneously, as the RNAs must compete for
    the active site of the endoRNAse. (See the RnaDegradation Process class for
    more information about the dynamical model.) The parameters are estimated
    using a root solver (scipy.optimize.fsolve).  (See the
    sim_data.process.rna_decay.kmLossFunction method for more information about
    the optimization problem.)

    Requires
    --------
    - cell density, dry mass fraction, and average initial dry mass
            Used to calculate the cell volume, which in turn is used to calculate
            concentrations.
    - observed RNA degradation rates (half-lives)
    - endoRNAse counts
    - endoRNAse catalytic rate constants
    - RNA counts
    - boolean options that enable sensitivity analyses (see Notes below)

    Modifies
    --------
    - Michaelis-Menten constants for first-order decay (initially set to zeros)
    - Several optimization-related values
            Sensitivity analyses (optional, see Notes below)
            Terminal values for optimization-related functions

    Returns
    -------
    - enoRNAse Km values, in units of M

    Notes
    -----
    If certain options are set, a sensitivity analysis will be performed using a
    range of metaparameters. Outputs will be cached and utilized instead of
    running the optimization if possible.
    The function that generates the optimization functions is defined under
    sim_data but has no dependency on sim_data, and therefore could be moved
    here or elsewhere. (TODO)

    TODO (John): Refactor as a pure function.
    TODO (John): Why is this function called 'cooperative'?  It seems to instead
            assume and model competitive binding.
    TODO (John): Determine what part (if any) of the 'linear' parameter fitting
            should be retained.
    """

    def arrays_differ(a: np.ndarray, b: np.ndarray) -> bool:
        return a.shape != b.shape or not np.allclose(a, b, equal_nan=True)

    cellDensity = sim_data.constants.cell_density
    cellVolume = (
        sim_data.mass.avg_cell_dry_mass_init
        / cellDensity
        / sim_data.mass.cell_dry_mass_fraction
    )
    countsToMolar = 1 / (sim_data.constants.n_avogadro * cellVolume)

    degradable_rna_ids = np.concatenate(
        (
            sim_data.process.transcription.rna_data["id"],
            sim_data.process.transcription.mature_rna_data["id"],
        )
    )
    degradation_rates = (1 / units.s) * np.concatenate(
        (
            sim_data.process.transcription.rna_data["deg_rate"].asNumber(1 / units.s),
            sim_data.process.transcription.mature_rna_data["deg_rate"].asNumber(
                1 / units.s
            ),
        )
    )
    endoRNase_idx = bulk_name_to_idx(
        sim_data.process.rna_decay.endoRNase_ids, bulkContainer["id"]
    )
    endoRNaseConc = countsToMolar * counts(bulkContainer, endoRNase_idx)
    kcatEndoRNase = sim_data.process.rna_decay.kcats
    totalEndoRnaseCapacity = units.sum(endoRNaseConc * kcatEndoRNase)

    endoRnaseRnaIds = sim_data.molecule_groups.endoRNase_rnas
    isEndoRnase = np.array([(x in endoRnaseRnaIds) for x in degradable_rna_ids])

    degradable_rna_idx = bulk_name_to_idx(degradable_rna_ids, bulkContainer["id"])
    rna_counts = counts(bulkContainer, degradable_rna_idx)
    rna_conc = countsToMolar * rna_counts
    Km_counts = ((1 / degradation_rates * totalEndoRnaseCapacity) - rna_conc).asNumber()
    sim_data.process.rna_decay.Km_first_order_decay = Km_counts

    # Residuals can be written as follows: Res = f(Km) = 0, then Km = g(Km)
    # Compute derivative g(Km) in counts:
    KmQuadratic = 1 / np.power((1 / countsToMolar * Km_counts).asNumber(), 2)
    denominator = np.power(
        np.sum(rna_counts / (1 / countsToMolar * Km_counts).asNumber()), 2
    )
    numerator = (1 / countsToMolar * totalEndoRnaseCapacity).asNumber() * (
        denominator - (rna_counts / (1 / countsToMolar * Km_counts).asNumber())
    )
    gDerivative = np.abs(KmQuadratic * (1 - (numerator / denominator)))
    if VERBOSE:
        print("Max derivative (counts) = %f" % max(gDerivative))

    # Compute derivative g(Km) in concentrations:
    KmQuadratic = 1 / np.power(Km_counts, 2)
    denominator = np.power(np.sum(rna_conc.asNumber() / Km_counts), 2)
    numerator = totalEndoRnaseCapacity.asNumber() * (
        denominator - (rna_conc.asNumber() / Km_counts)
    )
    gDerivative = np.abs(KmQuadratic * (1 - (numerator / denominator)))
    if VERBOSE:
        print("Max derivative (concentration) = %f" % max(gDerivative))

    # Sensitivity analysis: alpha (regularization term)
    Alphas = []
    if sim_data.constants.sensitivity_analysis_alpha:
        Alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]

    total_endo_rnase_capacity_mol_l_s = totalEndoRnaseCapacity.asNumber(
        units.mol / units.L / units.s
    )
    rna_conc_mol_l = rna_conc.asNumber(units.mol / units.L)
    degradation_rates_s = degradation_rates.asNumber(1 / units.s)

    for alpha in Alphas:
        if VERBOSE:
            print("Alpha = %f" % alpha)

        loss, loss_jac, res, res_aux = sim_data.process.rna_decay.km_loss_function(
            total_endo_rnase_capacity_mol_l_s,
            rna_conc_mol_l,
            degradation_rates_s,
            isEndoRnase,
            alpha,
        )
        Km_cooperative_model = np.exp(
            scipy.optimize.minimize(loss, np.log(Km_counts), jac=loss_jac).x
        )
        sim_data.process.rna_decay.sensitivity_analysis_alpha_residual[alpha] = np.sum(
            np.abs(res_aux(Km_cooperative_model))
        )

    alpha = 0.5

    # Sensitivity analysis: kcatEndoRNase
    kcatEndo = []
    if sim_data.constants.sensitivity_analysis_kcat_endo:
        kcatEndo = [0.0001, 0.001, 0.01, 0.1, 1, 10]

    for kcat in kcatEndo:
        if VERBOSE:
            print("Kcat = %f" % kcat)

        totalEndoRNcap = units.sum(endoRNaseConc * kcat)
        loss, loss_jac, res, res_aux = sim_data.process.rna_decay.km_loss_function(
            totalEndoRNcap.asNumber(units.mol / units.L),
            rna_conc_mol_l,
            degradation_rates_s,
            isEndoRnase,
            alpha,
        )
        km_counts_ini = (
            (totalEndoRNcap / degradation_rates.asNumber()) - rna_conc
        ).asNumber()
        Km_cooperative_model = np.exp(
            scipy.optimize.minimize(loss, np.log(km_counts_ini), jac=loss_jac).x
        )
        sim_data.process.rna_decay.sensitivity_analysis_kcat[kcat] = (
            Km_cooperative_model
        )
        sim_data.process.rna_decay.sensitivity_analysis_kcat_res_ini[kcat] = np.sum(
            np.abs(res_aux(km_counts_ini))
        )
        sim_data.process.rna_decay.sensitivity_analysis_kcat_res_opt[kcat] = np.sum(
            np.abs(res_aux(Km_cooperative_model))
        )

    # Loss function, and derivative
    loss, loss_jac, res, res_aux = sim_data.process.rna_decay.km_loss_function(
        total_endo_rnase_capacity_mol_l_s,
        rna_conc_mol_l,
        degradation_rates_s,
        isEndoRnase,
        alpha,
    )

    # The checksum in the filename picks independent caches for distinct cases
    # such as different Parca options or Parca code in different git branches.
    # `make clean` will delete the cache files.
    needToUpdate = ""
    checksum = crc32(Km_counts, isEndoRnase, np.array(alpha))
    km_filepath = os.path.join(cache_dir, f"parca-km-{checksum}.cPickle")

    if os.path.exists(km_filepath):
        with open(km_filepath, "rb") as f:
            Km_cache = pickle.load(f)

        # KmCooperativeModel fits a set of Km values to give the expected degradation rates.
        # It takes 1.5 - 3 minutes to recompute.
        # R_aux calculates the difference of the degradation rate based on these
        # Km values and the expected rate so this sum seems like a good test of
        # whether the cache fits current input data, but cross-check additional
        # inputs to avoid Issue #996.
        Km_cooperative_model = Km_cache["Km_cooperative_model"]
        if (
            Km_counts.shape != Km_cooperative_model.shape
            or np.sum(np.abs(res_aux(Km_cooperative_model))) > 1e-15
            or arrays_differ(
                Km_cache["total_endo_rnase_capacity_mol_l_s"],
                total_endo_rnase_capacity_mol_l_s,
            )
            or arrays_differ(Km_cache["rna_conc_mol_l"], rna_conc_mol_l)
            or arrays_differ(Km_cache["degradation_rates_s"], degradation_rates_s)
        ):
            needToUpdate = "recompute"
    else:
        needToUpdate = "compute"

    if needToUpdate:
        if VERBOSE:
            print(f"Running non-linear optimization to {needToUpdate} {km_filepath}")
        sol = scipy.optimize.minimize(loss, np.log(Km_counts), jac=loss_jac, tol=1e-8)
        Km_cooperative_model = np.exp(sol.x)
        Km_cache = dict(
            Km_cooperative_model=Km_cooperative_model,
            total_endo_rnase_capacity_mol_l_s=total_endo_rnase_capacity_mol_l_s,
            rna_conc_mol_l=rna_conc_mol_l,
            degradation_rates_s=degradation_rates_s,
        )

        with open(km_filepath, "wb") as f:
            pickle.dump(Km_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        if VERBOSE:
            print(
                "Not running non-linear optimization--using cached result {}".format(
                    km_filepath
                )
            )

    # Calculate log Km for loss functions
    log_Km_cooperative_model = np.log(Km_cooperative_model)
    log_Km_counts = np.log(Km_counts)

    if VERBOSE > 1:
        print("Loss function (Km inital) = %f" % np.sum(np.abs(loss(log_Km_counts))))
        print(
            "Loss function (optimized Km) = %f"
            % np.sum(np.abs(loss(log_Km_cooperative_model)))
        )
        print("Residuals (Km initial) = %f" % np.sum(np.abs(res(Km_counts))))
        print("Residuals optimized = %f" % np.sum(np.abs(res(Km_cooperative_model))))
        print(
            "EndoR residuals (Km initial) = %f"
            % np.sum(np.abs(isEndoRnase * res(Km_counts)))
        )
        print(
            "EndoR residuals optimized = %f"
            % np.sum(np.abs(isEndoRnase * res(Km_cooperative_model)))
        )
        print(
            "Residuals (scaled by Kdeg * RNAcounts) Km initial = %f"
            % np.sum(np.abs(res_aux(Km_counts)))
        )
        print(
            "Residuals (scaled by Kdeg * RNAcounts) optimized = %f"
            % np.sum(np.abs(res_aux(Km_cooperative_model)))
        )

    # Save statistics KM optimization
    sim_data.process.rna_decay.stats_fit["LossKm"] = np.sum(np.abs(loss(log_Km_counts)))
    sim_data.process.rna_decay.stats_fit["LossKmOpt"] = np.sum(
        np.abs(loss(log_Km_cooperative_model))
    )
    sim_data.process.rna_decay.stats_fit["ResKm"] = np.sum(np.abs(res(Km_counts)))
    sim_data.process.rna_decay.stats_fit["ResKmOpt"] = np.sum(
        np.abs(res(Km_cooperative_model))
    )
    sim_data.process.rna_decay.stats_fit["ResEndoRNKm"] = np.sum(
        np.abs(isEndoRnase * res(Km_counts))
    )
    sim_data.process.rna_decay.stats_fit["ResEndoRNKmOpt"] = np.sum(
        np.abs(isEndoRnase * res(Km_cooperative_model))
    )
    sim_data.process.rna_decay.stats_fit["ResScaledKm"] = np.sum(
        np.abs(res_aux(Km_counts))
    )
    sim_data.process.rna_decay.stats_fit["ResScaledKmOpt"] = np.sum(
        np.abs(res_aux(Km_cooperative_model))
    )

    return units.mol / units.L * Km_cooperative_model
