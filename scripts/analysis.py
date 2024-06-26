import argparse
import importlib
import json
import os
import warnings
from urllib import parse

import duckdb
from fsspec import filesystem
import pyarrow as pa

from ecoli.composites.ecoli_configs import CONFIG_DIR_PATH
from ecoli.experiments.ecoli_master_sim import SimConfig
from ecoli.library.parquet_emitter import get_dataset_sql

FILTERS = {
    "experiment_id": (str, "multiexperiment", "multivariant"),
    "variant": (int, "multivariant", "multiseed"),
    "lineage_seed": (int, "multiseed", "multigeneration"),
    "generation": (int, "multigeneration", "multidaughter"),
    "agent_id": (str, "multidaughter", "single"),
}
"""Mapping of data filters to data type, analysis type if more than
one value is given for filter, and analysis type for one value."""


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
    for data_filter, (data_type, _, _) in FILTERS.items():
        parser.add_argument(
            f"--{data_filter}",
            nargs="*",
            type=data_type,
            help=f"Limit data to one or more {data_filter}(s).",
        )
        if data_type is not str:
            parser.add_argument(
                f"--{data_filter}-range",
                nargs=2,
                metavar=("START", "END"),
                type=data_type,
                help=f"Limit data to range of {data_filter}s not incl. END.",
            )
    parser.add_argument(
        "--sim_data_path",
        "--sim-data-path",
        nargs="*",
        help="Path to the sim_data pickle(s) to use.",
    )
    parser.add_argument(
        "--validation_data_path",
        "--validation-data-path",
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
        "--variant-metadata-path",
        "--variant_metadata_path",
        help="Path to JSON file with variant metadata from create_variants.py.",
    )
    config_file = os.path.join(CONFIG_DIR_PATH, "default.json")
    args = parser.parse_args()
    with open(config_file, "r") as f:
        config = json.load(f)
    if args.config is not None:
        config_file = args.config
        with open(os.path.join(args.config), "r") as f:
            SimConfig.merge_config_dicts(config, json.load(f))
    if "out_uri" not in config["emitter"]["config"]:
        out_uri = os.path.abspath(config["emitter"]["config"]["out_dir"])
        gcs_bucket = True
    else:
        out_uri = config["emitter"]["config"]["out_uri"]
        assert (parse.urlparse(out_uri).scheme == "gcs" or
                parse.urlparse(out_uri).scheme == "gs")
        gcs_bucket = True
    config = config["analysis_options"]
    for k, v in vars(args).items():
        if v is not None:
            config[k] = v

    # Set number of threads for PyArrow
    if "n_cpus" in config:
        pa.set_cpu_count(config["n_cpus"])

    # Set up DuckDB filters for data
    # If no filters were provided, assume analyzing ParCa output
    analysis_type = "parca"
    duckdb_filter = []
    last_analysis_level = -1
    filter_types = list(FILTERS.keys())
    for current_analysis_level, (
        data_filter,
        (data_type, analysis_many, analysis_one),
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
                analysis_type = analysis_many
                if data_type is str:
                    filter_values = "', '".join(
                        parse.quote_plus(str(i)) for i in config[data_filter]
                    )
                    duckdb_filter.append(f"{data_filter} IN ('{filter_values}')")
                else:
                    filter_values = ", ".join(str(i) for i in config[data_filter])
                    duckdb_filter.append(f"{data_filter} IN ({filter_values})")
            else:
                analysis_type = analysis_one
                if data_type is str:
                    quoted_val = parse.quote_plus(str(config[data_filter][0]))
                    duckdb_filter.append(f"{data_filter} = '{quoted_val}'")
                else:
                    duckdb_filter.append(f"{data_filter} = {config[data_filter][0]}")
            last_analysis_level = current_analysis_level

    # Load variant metadata
    with open(config["variant_metadata_path"], "r") as f:
        variant_metadata = json.load(f)
        variant_name = list(variant_metadata.keys())[0]
        variant_metadata = {
            int(k): v for k, v in variant_metadata[variant_name].items()
        }

    # Run the analyses listed under the most specific filter
    analysis_options = config[analysis_type]
    analysis_modules = {}
    for analysis_name in analysis_options:
        analysis_modules[analysis_name] = importlib.import_module(
            f"ecoli.analysis.{analysis_type}.{analysis_name}"
        )
    for analysis_name, analysis_mod in analysis_modules.items():
        # Establish a fresh in-memory DuckDB for every analysis
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
        if "n_cpus" in config:
            conn.execute(f"SET threads = {config['n_cpus']}")
        if analysis_type == "parca":
            config_sql = ""
            history_sql = ""
        else:
            # Get SQL to read Parquet files from output directory with filters
            history_sql, config_sql = get_dataset_sql(out_path)
            duckdb_filter = " AND ".join(duckdb_filter)
            config_sql = f"{config_sql} WHERE {duckdb_filter}"
            history_sql = f"{history_sql} WHERE {duckdb_filter}"
        analysis_mod.plot(
            analysis_options[analysis_name],
            conn,
            history_sql,
            config_sql,
            dict(zip(config["variant"], config["sim_data_path"])),
            config["validation_data_path"],
            config["outdir"],
            variant_metadata,
            variant_name,
        )

    # Save copy of config JSON with parameters for plots
    with open(os.path.join(config["outdir"], "metadata.json"), "w") as f:
        json.dump(config, f)


if __name__ == "__main__":
    main()
