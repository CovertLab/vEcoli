import argparse
import importlib
import json
import os
import warnings

import duckdb
from fsspec import filesystem

from ecoli.composites.ecoli_configs import CONFIG_DIR_PATH
from ecoli.experiments.ecoli_master_sim import SimConfig
from ecoli.library.parquet_emitter import register_sim_views

FILTERS = {
    "experiment_id": (str, "multi_experiment", "multi_variant"),
    "variant": (int, "multi_variant", "multi_seed"),
    "lineage_seed": (int, "multi_seed", "multi_generation"),
    "generation": (int, "multi_generation", "multi_daughter"),
    "agent_id": (str, "multi_daughter", "single"),
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
    config_file = os.path.join(CONFIG_DIR_PATH, "default.json")
    args = parser.parse_args()
    with open(config_file, "r") as f:
        config = json.load(f)
    if args.config is not None:
        config_file = args.config
        with open(os.path.join(args.config), "r") as f:
            SimConfig.merge_config_dicts(config, json.load(f))
    out_dir = config["emitter"]["config"].get("out_dir", None)
    out_uri = config["emitter"]["config"].get("out_uri", None)
    config = config["analysis_options"]
    for k, v in vars(args).items():
        if v is not None:
            config[k] = v

    # Set up Polars filters for data
    analysis_type = None
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
                    filter_values = "', '".join(config[data_filter])
                    duckdb_filter.append(f"{data_filter} IN ('{filter_values}')")
                else:
                    filter_values = ", ".join(config[data_filter])
                    duckdb_filter.append(f"{data_filter} IN ({filter_values})")
            else:
                analysis_type = analysis_one
                if data_type is str:
                    duckdb_filter.append(f"{data_filter} = '{config[data_filter][0]}'")
                else:
                    duckdb_filter.append(f"{data_filter} = {config[data_filter][0]}")
            last_analysis_level = current_analysis_level

    # Run the analyses listed under the most specific filter
    analysis_options = config[analysis_type]
    analysis_modules = {}
    for analysis_name in analysis_options:
        analysis_modules[analysis_name] = importlib.import_module(f"ecoli.analysis.{analysis_name}")
    for analysis_name, analysis_mod in analysis_modules.items():
        # Establish a fresh in-memory DuckDB for every analysis
        conn = duckdb.connect()
        out_path = out_dir
        if out_path is None:
            out_path = out_uri
            duckdb.register_filesystem(filesystem("gcs"))
        conn.execute(f"SET temp_directory = '{out_path}'")
        conn.execute("SET preserve_insertion_order = false")
        # If no filters were provided, assume analyzing ParCa output
        if analysis_type is None:
            analysis_type = "parca"
        else:
            # Register filtered views of Parquet files from output directory
            # or URI specified in config
            register_sim_views(conn, out_path)
            duckdb_filter = " AND ".join(duckdb_filter)
            conn.register("configuration", conn.sql(
                f"SELECT * FROM unfiltered_configuration WHERE {duckdb_filter}"))
            conn.register("history", conn.sql(
                f"SELECT * FROM unfiltered_history WHERE {duckdb_filter}"))
        analysis_mod.plot(
            analysis_options[analysis_name],
            conn,
            config["sim_data_path"],
            config["validation_data_path"],
            config["outdir"],
        )

    # Save copy of config JSON with parameters for plots
    with open(os.path.join(config["outdir"], "metadata.json"), "w") as f:
        json.dump(config, f)


if __name__ == "__main__":
    main()
