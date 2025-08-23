import os
import argparse
import json


# First, do minimal argument parsing just to get the CPU count
def parse_cpu_arg():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cpus", "-n", default=1, type=int)
    parser.add_argument("--config")
    # Only arguments necessary to determine CPU count
    args, _ = parser.parse_known_args()
    if args.config is None:
        return args.cpus
    with open(args.config, "r") as f:
        config = json.load(f)
    return config["analysis_options"].get("cpus", args.cpus)


# Set Polars thread count before any imports might load it
cpu_count = parse_cpu_arg()
if cpu_count is not None:
    os.environ["POLARS_MAX_THREADS"] = str(cpu_count)
    print(f"Setting POLARS_MAX_THREADS={cpu_count}")

import importlib  # noqa: E402
import warnings  # noqa: E402
from urllib import parse  # noqa: E402
from typing import Any  # noqa: E402

from fsspec import url_to_fs  # noqa: E402

from configs import CONFIG_DIR_PATH  # noqa: E402
from ecoli.experiments.ecoli_master_sim import SimConfig  # noqa: E402
from ecoli.library.parquet_emitter import (  # noqa: E402
    dataset_sql,
    create_duckdb_conn,
    open_output_file,
)

FILTERS = {
    "experiment_id": str,
    "variant": int,
    "lineage_seed": int,
    "generation": int,
    "agent_id": str,
}
"""Mapping of data filters to data type."""

ANALYSIS_TYPES = {
    "multiexperiment": [],
    "multivariant": ["experiment_id"],
    "multiseed": ["experiment_id", "variant"],
    "multigeneration": ["experiment_id", "variant", "lineage_seed"],
    "multidaughter": ["experiment_id", "variant", "lineage_seed", "generation"],
    "single": ["experiment_id", "variant", "lineage_seed", "generation", "agent_id"],
    "parca": [],
}
"""Mapping of all possible analysis types to the combination of identifiers that
must be unique for each subset of the data given to that analysis type as input."""


def parse_variant_data_dir(
    experiment_id: list[str], variant_data_dir: list[str]
) -> tuple[dict[str, dict[int, Any]], dict[str, dict[int, str]], dict[str, str]]:
    """
    For each experiment ID and corresponding variant sim data directory,
    load the variant metadata JSON and parse the variant sim data file
    names to construct mappings from experiments to variants to variant
    metadata and variant sim_data paths.

    Args:
        experiment_id: List of experiment IDs
        variant_data_dir: List of directories containing output from
            create_variants.py, one for each experiment ID, in order

    Returns:
        Tuple containing three dictionaries::

            (
                {experiment_id: {variant_id: variant_metadata, ...}, ...},
                {experiment_id: {variant_id: variant_sim_data_path, ...}, ...}
                {experiment_id: variant_name, ...}
            )
    """
    variant_metadata = {}
    sim_data_dict = {}
    variant_names = {}
    for e_id, v_data_dir in zip(experiment_id, variant_data_dir):
        with open_output_file(os.path.join(v_data_dir, "metadata.json")) as f:
            v_metadata = json.load(f)
            variant_name = list(v_metadata.keys())[0]
            variant_names[e_id] = variant_name
            variant_metadata[e_id] = {
                int(k): v for k, v in v_metadata[variant_name].items()
            }
        if not (v_data_dir.startswith("gs://") or v_data_dir.startswith("gcs://")):
            v_data_dir = os.path.abspath(v_data_dir)
        fs, data_dir = url_to_fs(v_data_dir)
        sim_data_dict[e_id] = {
            int(os.path.basename(os.path.splitext(data_path)[0])): str(data_path)
            for data_path in fs.find(data_dir)
            if os.path.splitext(data_path)[1] == ".cPickle"
        }
    return variant_metadata, sim_data_dict, variant_names


def make_sim_data_dict(exp_id: str, variants: list[int], sim_data_path: list[str]):
    if len(variants) == 0:
        raise ValueError(
            "Must specify variant or variant_range if not using variant_data_dir"
        )
    if len(sim_data_path) != len(variants):
        raise ValueError(
            "Must specify sim_data_path for each variant if not using variant_data_dir"
        )
    return {exp_id: dict(zip(variants, sim_data_path))}


def main():
    parser = argparse.ArgumentParser()
    default_config = os.path.join(CONFIG_DIR_PATH, "default.json")
    parser.add_argument(
        "--config",
        default=default_config,
        help=(
            "Path to configuration file for the simulation. "
            "All key-value pairs in this file will be applied on top "
            f"of the options defined in {default_config}."
        ),
    )
    for data_filter, data_type in FILTERS.items():
        parser.add_argument(
            f"--{data_filter}",
            nargs="*",
            type=data_type,
            help=f"Limit data to one or more {data_filter}(s).",
        )
        if data_type is not str:
            parser.add_argument(
                f"--{data_filter}_range",
                nargs=2,
                metavar=("START", "END"),
                type=data_type,
                help=f"Limit data to range of {data_filter}s not incl. END.",
            )
    parser.add_argument(
        "--sim_data_path",
        nargs="*",
        help="Path to the sim_data pickle(s) to use. If multiple variants given"
        " via --variant or --variant_range, must provide same number"
        " of paths here in same order. Alternatively, see --variant_data_dir.",
    )
    parser.add_argument(
        "--validation_data_path",
        nargs="*",
        help="Path to the validation_data pickle(s) to use.",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        help="Directory that all analysis output is saved to."
        " MUST be a local path, not a Cloud Storage bucket URI.",
    )
    parser.add_argument(
        "--cpus",
        "-n",
        type=int,
        help="Number of CPUs to use for DuckDB.",
    )
    parser.add_argument(
        "--variant_metadata_path",
        help="Path to JSON file with variant metadata from create_variants.py."
        " Required with --sim_data_path. Otherwise, see --variant_data_dir.",
    )
    parser.add_argument(
        "--variant_data_dir",
        nargs="*",
        help="Path(s) to one or more directories containing variant sim data"
        " and metadata from create_variants.py. Supersedes --sim_data_path and"
        " --variant_metadata_path. If >1 experiment IDs, this is required and"
        " must have the same length and order as the given experiment IDs.",
    )
    parser.add_argument(
        "--analysis_types",
        "-t",
        nargs="*",
        choices=list(ANALYSIS_TYPES.keys()),
        help="Type(s) of analysis scripts to run. By default, every script under"
        " analysis_options in the config JSON is run. For example, say that"
        " 2 experiment IDs are given with --experiment_id, 2 variants with"
        " --variant, 2 seeds with --lineage_seed, 2 generations with --generation,"
        " and 2 agent IDs with --agent_id. The multiexperiment scripts in the config"
        " JSON will each run once with all data matching this filter. The multivariant"
        " scripts will each run twice, first with filtered data for one experiment ID,"
        " then with filtered data for the other. The multiseed scripts will each run"
        " 4 times (2 exp IDs * 2 variants), the multigeneration scripts 8 times (4"
        " * 2 seeds), the multidaughter scripts 16 times (8 * 2 generations), and"
        " the single scripts 32 times (16 * 2 agent IDs). If you only want to run"
        " the single and multivariant scripts, specify -t single multivariant.",
    )
    config_file = os.path.join(CONFIG_DIR_PATH, "default.json")
    args = parser.parse_args()
    with open(config_file, "r") as f:
        config = json.load(f)
    if args.config is not None:
        config_file = args.config
        with open(os.path.join(args.config), "r") as f:
            SimConfig.merge_config_dicts(config, json.load(f))
    if "out_uri" not in config["emitter_arg"]:
        out_uri = os.path.abspath(config["emitter_arg"]["out_dir"])
        gcs_bucket = False
    else:
        out_uri = config["emitter_arg"]["out_uri"]
        assert (
            parse.urlparse(out_uri).scheme == "gcs"
            or parse.urlparse(out_uri).scheme == "gs"
        )
        gcs_bucket = True
    config = config["analysis_options"]
    for k, v in vars(args).items():
        if v is not None:
            config[k] = v

    # Set up DuckDB filters for data
    duckdb_filter = []
    last_analysis_level = -1
    filter_types = list(FILTERS.keys())
    for current_analysis_level, (
        data_filter,
        data_type,
    ) in enumerate(FILTERS.items()):
        if config.get(f"{data_filter}_range", None) is not None:
            if config[data_filter] is not None:
                warnings.warn(
                    f"Provided both range and value(s) for {data_filter}. "
                    "Range takes precedence."
                )
            config[data_filter] = list(
                range(
                    config[f"{data_filter}_range"][0], config[f"{data_filter}_range"][1]
                )
            )
        if config.get(data_filter, None) is not None:
            if last_analysis_level != current_analysis_level - 1:
                skipped_filters = filter_types[
                    last_analysis_level + 1 : current_analysis_level
                ]
                warnings.warn(
                    f"Filtering by {data_filter} when last filter "
                    f"specified was {filter_types[last_analysis_level]}. "
                    "Will load all applicable data for the skipped "
                    f"filters: {skipped_filters}."
                )
            if len(config[data_filter]) > 1:
                if data_type is str:
                    filter_values = "', '".join(
                        parse.quote_plus(str(i)) for i in config[data_filter]
                    )
                    duckdb_filter.append(f"{data_filter} IN ('{filter_values}')")
                else:
                    filter_values = ", ".join(str(i) for i in config[data_filter])
                    duckdb_filter.append(f"{data_filter} IN ({filter_values})")
            else:
                if data_type is str:
                    quoted_val = parse.quote_plus(str(config[data_filter][0]))
                    duckdb_filter.append(f"{data_filter} = '{quoted_val}'")
                else:
                    duckdb_filter.append(f"{data_filter} = {config[data_filter][0]}")
            last_analysis_level = current_analysis_level
    duckdb_filter = " AND ".join(duckdb_filter)

    # Load variant metadata
    if "experiment_id" not in config:
        raise KeyError("Must provide at least one experiment ID with experiment_id")
    if len(config["experiment_id"]) > 1:
        assert "variant_data_dir" in config, (
            "Must provide --variant_data_dir for each experiment ID."
        )
        assert len(config["variant_data_dir"]) == len(config["experiment_id"]), (
            "Must provide --variant_data_dir for each experiment ID."
        )
    if "variant_data_dir" in config:
        if "variant_metadata_path" in config:
            warnings.warn(
                "Ignoring --variant_metadata_path in favor of --variant_data_dir"
            )
        if "sim_data_path" in config:
            warnings.warn("Ignoring --sim_data_path in favor of --variant_data_dir")
        variant_metadata, sim_data_dict, variant_names = parse_variant_data_dir(
            config["experiment_id"], config["variant_data_dir"]
        )
    elif "variant_metadata_path" in config:
        with open(config["variant_metadata_path"], "r") as f:
            variant_metadata = json.load(f)
            variant_name = list(variant_metadata.keys())[0]
            variant_metadata = {
                config["experiment_id"][0]: {
                    int(k): v for k, v in variant_metadata[variant_name].items()
                }
            }
            variant_names = {config["experiment_id"][0]: variant_name}
        sim_data_dict = make_sim_data_dict(
            config["experiment_id"][0],
            config.get("variant", []),
            config.get("sim_data_path", []),
        )
    else:
        warnings.warn(
            "No variant metadata provided. Using empty variant metadata/names dictionaries."
        )
        variant_metadata = {config["experiment_id"][0]: {}}
        variant_names = {config["experiment_id"][0]: None}
        sim_data_dict = make_sim_data_dict(
            config["experiment_id"][0],
            config.get("variant", []),
            config.get("sim_data_path", []),
        )

    # Save copy of config JSON with parameters for plots
    os.makedirs(config["outdir"], exist_ok=True)
    metadata_path = os.path.join(os.path.abspath(config["outdir"]), "metadata.json")
    if os.path.exists(metadata_path):
        raise FileExistsError(
            f"{metadata_path} already exists, indicating an analysis has "
            f"been run with output directory {config['outdir']}. Please "
            "delete/move it or specify a different output directory."
        )
    with open(metadata_path, "w") as f:
        json.dump(config, f)

    # Establish DuckDB connection
    conn = create_duckdb_conn(out_uri, gcs_bucket, config.get("cpus"))
    history_sql, config_sql, success_sql = dataset_sql(out_uri, config["experiment_id"])
    # If no explicit analysis type given, run all types in config JSON
    if "analysis_types" not in config:
        config["analysis_types"] = [
            analysis_type for analysis_type in ANALYSIS_TYPES if analysis_type in config
        ]
    for analysis_type in config["analysis_types"]:
        if analysis_type not in config:
            raise KeyError(
                f"Specified {analysis_type} analysis type"
                " but none provided in analysis_options."
            )
        elif len(config[analysis_type]) == 0:
            print(f"Skipping {analysis_type} analysis - none provided in config.")
            continue
        # Compile collection of history and config SQL queries for each cell
        # subset identified for current analysis type
        id_cols = ANALYSIS_TYPES[analysis_type]
        query_strings = {}
        # Figure out what Hive partition in main output directory
        # to store outputs for analyses run on this cell subset
        if len(id_cols) > 0:
            joined_cols = ", ".join(id_cols)
            data_ids = conn.sql(
                f"SELECT DISTINCT ON({joined_cols}) {joined_cols}"
                f" FROM ({config_sql}) WHERE {duckdb_filter}"
            ).fetchall()
            for data_id in data_ids:
                data_filters = []
                curr_outdir = os.path.abspath(config["outdir"])
                for col, col_val in zip(id_cols, data_id):
                    curr_outdir = os.path.join(curr_outdir, f"{col}={col_val}")
                    # Quote string Hive partition values for DuckDB query
                    if FILTERS[col] is str:
                        col_val = f"'{col_val}'"
                    data_filters.append(f"{col}={col_val}")
                os.makedirs(curr_outdir, exist_ok=True)
                data_filters = " AND ".join(data_filters)
                query_strings[data_filters] = (
                    f"SELECT * FROM ({history_sql}) WHERE {data_filters}",
                    f"SELECT * FROM ({config_sql}) WHERE {data_filters}",
                    f"SELECT * FROM ({success_sql}) WHERE {data_filters}",
                    curr_outdir,
                )
        else:
            curr_outdir = os.path.abspath(config["outdir"])
            os.makedirs(curr_outdir, exist_ok=True)
            query_strings[duckdb_filter] = (
                f"SELECT * FROM ({history_sql}) WHERE {duckdb_filter}",
                f"SELECT * FROM ({config_sql}) WHERE {duckdb_filter}",
                f"SELECT * FROM ({success_sql}) WHERE {duckdb_filter}",
                os.path.abspath(config["outdir"]),
            )
        for analysis_name in config[analysis_type]:
            analysis_mod = importlib.import_module(
                f"ecoli.analysis.{analysis_type}.{analysis_name}"
            )
            for data_filters, (
                history_q,
                config_q,
                success_q,
                curr_outdir,
            ) in query_strings.items():
                print(f"Running {analysis_type} {analysis_name} with {data_filters}.")
                analysis_mod.plot(
                    config[analysis_type][analysis_name],
                    conn,
                    history_q,
                    config_q,
                    success_q,
                    sim_data_dict,
                    config.get("validation_data_path", []),
                    curr_outdir,
                    variant_metadata,
                    variant_names,
                )


if __name__ == "__main__":
    main()
