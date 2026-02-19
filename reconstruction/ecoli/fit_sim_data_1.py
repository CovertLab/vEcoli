"""
The parca, aka parameter calculator.

Orchestrates the 9-stage ParCa pipeline.  Each stage is implemented as a
pure-function module under ``reconstruction.ecoli.parca.stage_*`` with
explicit Input/Output dataclasses defined in ``reconstruction.ecoli.parca._types``.
"""

import functools
import os
import pickle
import time

from reconstruction.ecoli.simulation_data import SimulationDataEcoli
from reconstruction.ecoli.parca.stage_02_input_adjustments import (
    extract_input as _extract_input_adjustments,
    compute_input_adjustments as _compute_input_adjustments,
    merge_output as _merge_input_adjustments,
)
from reconstruction.ecoli.parca.stage_03_basal_specs import (
    extract_input as _extract_basal_specs,
    compute_basal_specs as _compute_basal_specs,
    merge_output as _merge_basal_specs,
)
from reconstruction.ecoli.parca.stage_04_tf_condition_specs import (
    extract_input as _extract_tf_condition_specs,
    compute_tf_condition_specs as _compute_tf_condition_specs,
    merge_output as _merge_tf_condition_specs,
)
from reconstruction.ecoli.parca.stage_05_fit_condition import (
    extract_input as _extract_fit_condition,
    compute_fit_condition as _compute_fit_condition,
    merge_output as _merge_fit_condition,
)
from reconstruction.ecoli.parca.stage_06_promoter_binding import (
    extract_input as _extract_promoter_binding,
    compute_promoter_binding as _compute_promoter_binding,
    merge_output as _merge_promoter_binding,
)
from reconstruction.ecoli.parca.stage_07_adjust_promoters import (
    extract_input as _extract_adjust_promoters,
    compute_adjust_promoters as _compute_adjust_promoters,
    merge_output as _merge_adjust_promoters,
)
from reconstruction.ecoli.parca.stage_08_set_conditions import (
    extract_input as _extract_set_conditions,
    compute_set_conditions as _compute_set_conditions,
    merge_output as _merge_set_conditions,
)
from reconstruction.ecoli.parca.stage_09_final_adjustments import (
    extract_input as _extract_final_adjustments,
    compute_final_adjustments as _compute_final_adjustments,
    merge_output as _merge_final_adjustments,
)

# Backward-compatible re-exports used by test_fit_sim_data_1.py and analysis scripts.
from reconstruction.ecoli.parca._math import (  # noqa: F401
    totalCountFromMassesAndRatios,
    proteinDistributionFrommRNA,
    mRNADistributionFromProtein,
    calculateMinPolymerizingEnzymeByProductDistribution,
    netLossRateFromDilutionAndDegradationProtein,
)

BASAL_EXPRESSION_CONDITION = "M9 Glucose minus AAs"

functions_run = []


def fitSimData_1(raw_data, **kwargs):
    """
    Fits parameters necessary for the simulation based on the knowledge base

    Inputs:
            raw_data (KnowledgeBaseEcoli) - knowledge base consisting of the
                    necessary raw data
            cpus (int) - number of processes to use (if > 1, use multiprocessing)
            debug (bool) - if True, fit only one arbitrarily-chosen transcription
                    factor in order to speed up a debug cycle (should not be used for
                    an actual simulation)
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

    """

    sim_data = SimulationDataEcoli()
    cell_specs = {}

    # Functions to modify sim_data and/or cell_specs
    # Functions defined below should be wrapped by @save_state to allow saving
    # and loading sim_data and cell_specs to skip certain functions while doing
    # development for faster testing and iteration of later functions that
    # might not need earlier functions to be rerun each time.
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
            print(f"Saved data for {func_name}")

        # Record which functions have been run to know if the loaded function has run
        functions_run.append(func_name)

        return sim_data, cell_specs

    return wrapper


@save_state
def initialize(sim_data, cell_specs, raw_data=None, **kwargs):
    sim_data.initialize(
        raw_data=raw_data,
        basal_expression_condition=BASAL_EXPRESSION_CONDITION,
    )

    return sim_data, cell_specs


@save_state
def input_adjustments(sim_data, cell_specs, **kwargs):
    inp = _extract_input_adjustments(sim_data, cell_specs, **kwargs)
    out = _compute_input_adjustments(inp)
    _merge_input_adjustments(sim_data, cell_specs, out)
    return sim_data, cell_specs


@save_state
def basal_specs(sim_data, cell_specs, **kwargs):
    inp = _extract_basal_specs(sim_data, cell_specs, **kwargs)
    out = _compute_basal_specs(inp)
    _merge_basal_specs(sim_data, cell_specs, out)

    return sim_data, cell_specs


@save_state
def tf_condition_specs(sim_data, cell_specs, **kwargs):
    inp = _extract_tf_condition_specs(sim_data, cell_specs, **kwargs)
    out = _compute_tf_condition_specs(inp)
    _merge_tf_condition_specs(sim_data, cell_specs, out)
    return sim_data, cell_specs


@save_state
def fit_condition(sim_data, cell_specs, **kwargs):
    inp = _extract_fit_condition(sim_data, cell_specs, **kwargs)
    out = _compute_fit_condition(inp)
    _merge_fit_condition(sim_data, cell_specs, out)
    return sim_data, cell_specs


@save_state
def promoter_binding(sim_data, cell_specs, **kwargs):
    inp = _extract_promoter_binding(sim_data, cell_specs, **kwargs)
    out = _compute_promoter_binding(inp)
    _merge_promoter_binding(sim_data, cell_specs, out)
    return sim_data, cell_specs


@save_state
def adjust_promoters(sim_data, cell_specs, **kwargs):
    inp = _extract_adjust_promoters(sim_data, cell_specs, **kwargs)
    out = _compute_adjust_promoters(inp)
    _merge_adjust_promoters(sim_data, cell_specs, out)
    return sim_data, cell_specs


@save_state
def set_conditions(sim_data, cell_specs, **kwargs):
    inp = _extract_set_conditions(sim_data, cell_specs, **kwargs)
    out = _compute_set_conditions(inp)
    _merge_set_conditions(sim_data, cell_specs, out)
    return sim_data, cell_specs


@save_state
def final_adjustments(sim_data, cell_specs, **kwargs):
    inp = _extract_final_adjustments(sim_data, cell_specs, **kwargs)
    out = _compute_final_adjustments(inp)
    _merge_final_adjustments(sim_data, cell_specs, out)
    return sim_data, cell_specs
