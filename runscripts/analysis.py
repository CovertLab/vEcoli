import argparse
import importlib
import json
import os
import warnings
from urllib import parse
from typing import Any

import duckdb
from fsspec import filesystem
import pyarrow as pa
from pyarrow import fs

from ecoli.composites.ecoli_configs import CONFIG_DIR_PATH
from ecoli.experiments.ecoli_master_sim import SimConfig
from ecoli.library.parquet_emitter import get_dataset_sql, open_output_file

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
) -> tuple[dict[str, dict[int, Any]], dict[str, dict[int, str]], list[str]]:
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
        Tuple containing two nested dictionaries and a list::

            (
                {experiment_id: {variant_id: variant_metadata, ...}, ...},
                {experiment_id: {variant_id: variant_sim_data_path, ...}, ...}
                [variant_name_for_experiment_id, ...]
            )
    """
    variant_metadata = {}
    sim_data_dict = {}
    variant_names = []
    for e_id, v_data_dir in zip(experiment_id, variant_data_dir):
        with open_output_file(os.path.join(v_data_dir, "metadata.json")) as f:
            v_metadata = json.load(f)
            variant_name = list(v_metadata.keys())[0]
            variant_names.append(variant_name)
            variant_metadata[e_id] = {
                int(k): v for k, v in v_metadata[variant_name].items()
            }
        if not (v_data_dir.startswith("gs://") or v_data_dir.startswith("gcs://")):
            v_data_dir = os.path.abspath(v_data_dir)
        filesystem, data_dir = fs.FileSystem.from_uri(v_data_dir)
        sim_data_dict[e_id] = {
            int(os.path.basename(os.path.splitext(i.path)[0])): str(i.path)
            for i in filesystem.get_file_info(fs.FileSelector(data_dir, recursive=True))
            if os.path.splitext(i.path)[1] == ".cPickle"
        }
    return variant_metadata, sim_data_dict, variant_names


def create_duckdb_conn(out_uri, gcs_bucket, n_cpus=None):
    conn = duckdb.connect()
    out_path = out_uri
    if gcs_bucket:
        conn.register_filesystem(filesystem("gcs"))
    # Temp directory so DuckDB can spill to disk when data larger than RAM
    conn.execute(f"SET temp_directory = '{out_path}'")
    # Turning this off reduces RAM usage
    conn.execute("SET preserve_insertion_order = false")
    # Cache Parquet metadata so only needs to be scanned once
    conn.execute("SET enable_object_cache = true")
    # Set number of threads for DuckDB
    if n_cpus is not None:
        conn.execute(f"SET threads = {n_cpus}")
    return conn


def main():
    parser = argparse.ArgumentParser()
    default_config = os.path.join(CONFIG_DIR_PATH, "default.json")
    parser.add_argument(
        "--config",
        "-c",
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
        " via --variant or --variant-range, must provide same number"
        " of paths here in same order. Alternatively, see --variant-data-dir.",
    )
    parser.add_argument(
        "--validation_data_path",
        nargs="*",
        help="Path to the validation_data pickle(s) to use.",
    )
    parser.add_argument(
        "--outdir", "-o", help="Directory that all analysis output is saved to."
    )
    parser.add_argument(
        "--n_cpus", "-n", help="Number of CPUs to use for DuckDB and PyArrow."
    )
    parser.add_argument(
        "--variant_metadata_path",
        help="Path to JSON file with variant metadata from create_variants.py."
        " Required with --sim-data-path. Otherwise, see --variant-data-dir.",
    )
    parser.add_argument(
        "--variant_data_dir",
        nargs="*",
        help="Path(s) to one or more directories containing variant sim data"
        " and metadata from create_variants.py. Supersedes --sim-data-path and"
        " --variant-metadata-path. If >1 experiment IDs, this is required and"
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
    if "out_uri" not in config["emitter"]:
        out_uri = os.path.abspath(config["emitter"]["out_dir"])
        gcs_bucket = True
    else:
        out_uri = config["emitter"]["out_uri"]
        assert (
            parse.urlparse(out_uri).scheme == "gcs"
            or parse.urlparse(out_uri).scheme == "gs"
        )
        gcs_bucket = True
    config = config["analysis_options"]
    for k, v in vars(args).items():
        if v is not None:
            config[k] = v

    # Set number of threads for PyArrow
    if "n_cpus" in config:
        pa.set_cpu_count(config["n_cpus"])

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
    if len(config["experiment_id"]) > 1:
        assert (
            "variant_data_dir" in config
        ), "Must provide --variant-data-dir for each experiment ID."
        assert len(config["variant_data_dir"]) == len(
            config["experiment_id"]
        ), "Must provide --variant-data-dir for each experiment ID."
    if "variant_data_dir" in config:
        if "variant_metadata_path" in config:
            warnings.warn(
                "Ignoring --variant-metadata-path in favor of" " --variant-data-dir"
            )
        if "sim_data_path" in config:
            warnings.warn("Ignoring --sim-data-path in favor of" " --variant-data-dir")
        variant_metadata, sim_data_dict, variant_names = parse_variant_data_dir(
            config["experiment_id"], config["variant_data_dir"]
        )
    else:
        with open(config["variant_metadata_path"], "r") as f:
            variant_metadata = json.load(f)
            variant_name = list(variant_metadata.keys())[0]
            variant_metadata = {
                config["experiment_id"][0]: {
                    int(k): v for k, v in variant_metadata[variant_name].items()
                }
            }
        sim_data_dict = {
            config["experiment_id"][0]: dict(
                zip(config["variant"], config["sim_data_path"])
            )
        }
        variant_names = [variant_name]

    # Establish DuckDB connection
    conn = create_duckdb_conn(out_uri, gcs_bucket, config.get("n_cpus"))
    history_sql, config_sql = get_dataset_sql(out_uri)
    # SQL template for retrieving unique column combinations for cell subsets
    id_template = (
        "SELECT DISTINCT ON({cols}) {cols}"
        f" FROM ({config_sql}) WHERE {duckdb_filter}"
    )
    # If no explicit analysis type given, run all types in config JSON
    if config["analysis_types"] is None:
        config["analysis_types"] = [
            analysis_type for analysis_type in ANALYSIS_TYPES if analysis_type in config
        ]
    for analysis_type in config["analysis_types"]:
        if analysis_type not in config:
            raise KeyError(
                f"Specified {analysis_type} analysis type"
                " but none provided in analysis_options."
            )
        # Compile collection of history and config SQL queries for each cell
        # subset identified for current analysis type
        cols = ANALYSIS_TYPES[analysis_type]
        query_strings = {}
        if len(cols) > 0:
            data_ids = conn.sql(id_template.format(cols=cols)).fetchall()
            for data_id in data_ids:
                data_filters = " AND ".join(
                    [f"{col} = {v}" for col, v in zip(cols, data_id)]
                )
                query_strings[data_filters] = (
                    f"SELECT * FROM ({history_sql}) WHERE {data_filters}",
                    f"SELECT * FROM ({config_sql}) WHERE {data_filters}",
                )
        else:
            query_strings[data_filters] = (
                f"SELECT * FROM ({history_sql}) WHERE {duckdb_filter}",
                f"SELECT * FROM ({config_sql}) WHERE {duckdb_filter}",
            )
        for analysis_name in config[analysis_type]:
            analysis_mod = importlib.import_module(
                f"ecoli.analysis.{analysis_type}.{analysis_name}"
            )
            for data_filters, (history_q, config_q) in query_strings.items():
                print(f"Running {analysis_type} {analysis_name} with {data_filters}.")
                analysis_mod.plot(
                    config[analysis_type][analysis_name],
                    conn,
                    history_q,
                    config_q,
                    sim_data_dict,
                    config["validation_data_path"],
                    config["outdir"],
                    variant_metadata,
                    variant_names,
                )

    # Save copy of config JSON with parameters for plots
    with open(os.path.join(config["outdir"], "metadata.json"), "w") as f:
        json.dump(config, f)


if __name__ == "__main__":
    main()
