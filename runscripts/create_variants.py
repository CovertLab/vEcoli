import argparse
import copy
import importlib
import itertools
import json
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from configs import CONFIG_DIR_PATH
from ecoli.experiments.ecoli_master_sim import SimConfig

if TYPE_CHECKING:
    from reconstruction.ecoli.simulation_data import SimulationDataEcoli


def parse_variants(
    variant_config: dict[str, str | dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Parse parameters for a variant specified under ``variants`` key of config.

    Args:
        variant_config: Dictionary of the form::

            {
                # Define one or more parameters with any names EXCEPT `op`
                'param_name': {
                    # Each parameter defines only ONE of the following keys
                    # A list of parameter values
                    'value': [...]
                    # Numpy function that returns array of parameter values
                    # Example: np.linspace
                    'linspace': {
                        # Kwargs
                        'start': ...,
                        'stop': ...,
                        'num' (optional): ...
                    }
                    # Dictionary of parameters with same rules as this one
                    'nested': {...}
                },
                # When more than one parameter is defined, an 'op' key
                # MUST define how to combine them. The three options are:
                # 'zip': Zip parameters (must have same length)
                # 'prod': Cartesian product of parameters
                # 'add': Concatenate parameter lists into single parameter
                #        named {param_name_1}__{param_name_2}__...
                'param_2': {...},
                'op': 'zip'
            }

    Returns:
        List of parameter dictionaries generated from variant config

    """
    # Extract operation if more than one parameter
    operation = None
    if len(variant_config) > 1:
        assert "op" in variant_config, (
            "Variant has more than 1 parameter but no op key defined."
        )
        operation = variant_config.pop("op")
    elif "op" in variant_config:
        raise TypeError(
            "Variant only has a single parameter and should not define op key."
        )

    # Perform pre-processing of parameters
    parsed = {}
    for param_name, param_conf in variant_config.items():
        param_conf = cast(dict[str, Any], param_conf)
        if len(param_conf) > 1:
            raise TypeError(f"{param_name} should only have 1 type.")
        param_type = list(param_conf.keys())[0]
        param_vals = param_conf[param_type]
        if param_type == "value":
            if not isinstance(param_vals, list):
                raise TypeError(f"{param_name} should have a list value.")
            parsed[param_name] = param_vals
        elif param_type == "nested":
            param_vals = cast(dict[str, str | dict[str, Any]], param_vals)
            parsed[param_name] = parse_variants(param_vals)
        else:
            try:
                np_func = getattr(np, param_type)
            except AttributeError as e:
                raise TypeError(f"{param_name} is unknown type {param_type}.") from e
            parsed[param_name] = np_func(**param_vals)

    # Apply parameter operations
    if operation == "prod":
        param_tuples = itertools.product(*(parsed[k] for k in parsed))
        param_dicts = [
            {name: val for name, val in zip(parsed.keys(), param_tuple)}
            for param_tuple in param_tuples
        ]
    elif operation == "zip":
        n_combos = -1
        for name, val in parsed.items():
            if n_combos == -1:
                n_combos = len(val)
            if len(val) != n_combos:
                raise RuntimeError(
                    f"At least 1 other parameter has a "
                    f"different # of values than {name}."
                )
        param_dicts = [
            {name: val[i] for name, val in parsed.items()} for i in range(n_combos)
        ]
    elif operation == "add":
        combined_param_name = "__".join(parsed)
        param_dicts = []
        for val in parsed.values():
            param_dicts.extend({combined_param_name: i} for i in val)
    elif operation is None:
        param_name = list(parsed.keys())[0]
        param_vals = parsed[param_name]
        param_dicts = [{param_name: param_val} for param_val in param_vals]
    else:
        raise RuntimeError(f"Unknown operation {operation} in {variant_config}")

    return param_dicts


def apply_and_save_variants(
    sim_data: "SimulationDataEcoli",
    param_dicts: list[dict[str, Any]],
    variant_name: str,
    outdir: str,
    skip_baseline: bool,
):
    """
    Applies variant function to ``sim_data`` with each parameter dictionary
    in ``param_dicts``. Saves each variant as ``{i}.cPickle``
    in ``outdir``, where ``i`` is the index of the parameter dictionary in
    ``param_dicts`` used to create that variant. Also saves ``metadata.json``
    in ``outdir`` that maps each ``{i}`` to the parameter
    dictionary used to create it.

    Args:
        sim_data: Simulation data object to modify
        param_dicts: Return value of :py:func:`~.parse_variants`
        variant_name: Name of variant function file in ``ecoli/variants`` folder
        outdir: Path to folder where variant ``sim_data`` pickles are saved
        skip_baseline: Whether to save metadata for baseline sim_data
    """
    variant_mod = importlib.import_module(f"ecoli.variants.{variant_name}")
    variant_metadata: dict[int, str | dict[str, Any]] = {}
    if not skip_baseline:
        variant_metadata[0] = "baseline"
    for i, params in enumerate(param_dicts):
        sim_data_copy = copy.deepcopy(sim_data)
        variant_metadata[i + 1] = params
        variant_sim_data = variant_mod.apply_variant(sim_data_copy, params)
        outpath = os.path.join(outdir, f"{i + 1}.cPickle")
        with open(outpath, "wb") as f:
            pickle.dump(variant_sim_data, f)
    with open(os.path.join(outdir, "metadata.json"), "w") as f:
        json.dump({variant_name: variant_metadata}, f)


def test_parse_variants():
    """
    Test variant parameter parsing.
    """
    variant_config = {
        "a": {"value": [1, 2]},
        "b": {"value": ["one", "two"]},
        "c": {"nested": {"d": {"value": [3, 4]}, "e": {"value": [5, 6]}, "op": "zip"}},
        "op": "prod",
    }
    parsed_params = parse_variants(variant_config)
    assert parsed_params == [
        {"a": 1, "b": "one", "c": {"d": 3, "e": 5}},
        {"a": 1, "b": "one", "c": {"d": 4, "e": 6}},
        {"a": 1, "b": "two", "c": {"d": 3, "e": 5}},
        {"a": 1, "b": "two", "c": {"d": 4, "e": 6}},
        {"a": 2, "b": "one", "c": {"d": 3, "e": 5}},
        {"a": 2, "b": "one", "c": {"d": 4, "e": 6}},
        {"a": 2, "b": "two", "c": {"d": 3, "e": 5}},
        {"a": 2, "b": "two", "c": {"d": 4, "e": 6}},
    ]


class SimData:
    """
    Mock sim_data class for testing.
    """

    pass


def test_create_variants():
    """
    Test modification and saving of variant sim_data.
    """
    try:
        os.makedirs("test_create_variants/kb", exist_ok=True)
        # Create mock sim_data pickle
        with open("test_create_variants/kb/simData.cPickle", "wb") as f:
            pickle.dump(SimData(), f)
        repo_dir = os.path.dirname(os.path.dirname(__file__))
        # Test script and config system
        os.environ["PYTHONPATH"] = repo_dir
        subprocess.run(
            [
                "python",
                "runscripts/create_variants.py",
                "--config",
                "configs/test_variant.json",
                "--kb",
                "test_create_variants/kb",
                "-o",
                "test_create_variants/out",
            ],
            check=True,
            env=os.environ,
        )
        # Check that metadata aligns with variant sim_data attrs
        with open("test_create_variants/out/metadata.json") as f:
            variant_metadata = json.load(f)
        assert "variant_test" in variant_metadata
        variant_metadata = variant_metadata["variant_test"]
        out_path = Path("test_create_variants/out")
        var_paths = out_path.glob("*.cPickle")
        for var_path in var_paths:
            # Skip baseline
            if var_path.stem == "0":
                continue
            with open(var_path, "rb") as f:
                variant_sim_data = pickle.load(f)
            variant_params = variant_metadata[var_path.stem]
            assert variant_sim_data.a == variant_params["a"]
            assert variant_sim_data.b == variant_params["b"]
            assert variant_sim_data.d == variant_params["c"]["d"]
            assert variant_sim_data.e == variant_params["c"]["e"]
    finally:
        shutil.rmtree("test_create_variants", ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    default_config = os.path.join(CONFIG_DIR_PATH, "default.json")
    parser.add_argument(
        "--config",
        action="store",
        default=default_config,
        help=(
            "Path to configuration file for the simulation. "
            "All key-value pairs in this file will be applied on top "
            f"of the options defined in {default_config}. To configure "
            "variants, the config must include the `variant` key. Under the "
            "`variant` key should be a single key with the name of the "
            "variant module under `ecoli.variant` (for example, `variant_1` "
            "if imported as `ecoli.variant.variant_1` or `folder_1.variant_1` "
            "if imported as `ecoli.variant.folder_1.variant_1`). See "
            "`ecoli.variants.template` for variant template. Under the "
            "variant module name should be a parameter dictionary as "
            "described in the docstring for `parse_variants`."
        ),
    )
    parser.add_argument(
        "--kb", action="store", type=str, help="Path to kb folder generated by ParCa."
    )
    parser.add_argument(
        "--outdir",
        "-o",
        action="store",
        type=str,
        help="Path to folder where variant sim_data and metadata are written.",
    )
    args = parser.parse_args()
    with open(default_config, "r") as f:
        config = json.load(f)
    if args.config is not None:
        with open(os.path.join(args.config), "r") as f:
            SimConfig.merge_config_dicts(config, json.load(f))
    for k, v in vars(args).items():
        if v is not None:
            config[k] = v

    print("Loading sim_data...")
    with open(os.path.join(config["kb"], "simData.cPickle"), "rb") as f:
        sim_data = pickle.load(f)
    config_outdir = os.path.abspath(config["outdir"])
    os.makedirs(config_outdir, exist_ok=True)
    if config["skip_baseline"]:
        print("Skipping baseline sim_data...")
    else:
        print("Saving baseline sim_data...")
        with open(os.path.join(config_outdir, "0.cPickle"), "wb") as f:
            pickle.dump(sim_data, f)
    variant_config = config.get("variants", {})
    if len(variant_config) > 1:
        raise RuntimeError(
            "Only one variant name allowed. Variants can "
            "be manually composed in Python by having one "
            "variant function internally call another."
        )
    elif len(variant_config) == 1:
        variant_name = list(variant_config.keys())[0]
        variant_params = variant_config[variant_name]
        print("Parsing variants...")
        parsed_params = parse_variants(variant_params)
        print("Applying variants and saving variant sim_data...")
        apply_and_save_variants(
            sim_data,
            parsed_params,
            variant_name,
            config_outdir,
            config["skip_baseline"],
        )
    else:
        with open(os.path.join(config_outdir, "metadata.json"), "w") as f:
            json.dump({None: {0: "baseline"}}, f)
    print("Done.")


if __name__ == "__main__":
    main()
