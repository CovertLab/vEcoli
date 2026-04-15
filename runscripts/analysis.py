import os
import argparse
import json
import sys
from collections import defaultdict


# First, do minimal argument parsing just to get the CPU count
def parse_cpu_arg():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cpus", "-n", type=int)
    parser.add_argument("--config")
    # Only arguments necessary to determine CPU count
    args, _ = parser.parse_known_args()
    if args.cpus is not None:
        return args.cpus
    if args.config is None:
        return 1
    with open(args.config, "r") as f:
        config = json.load(f)
    return config["analysis_options"].get("cpus", 1)


# Set Polars thread count before any imports might load it
cpu_count = parse_cpu_arg()
if cpu_count is not None:
    os.environ["POLARS_MAX_THREADS"] = str(cpu_count)
    print(f"Setting POLARS_MAX_THREADS={cpu_count}")

import importlib  # noqa: E402
import warnings  # noqa: E402
from urllib import parse  # noqa: E402
from typing import Any  # noqa: E402

from fsspec import open as fsspec_open, url_to_fs  # noqa: E402

from configs import CONFIG_DIR_PATH  # noqa: E402
from ecoli.experiments.ecoli_master_sim import SimConfig  # noqa: E402
from ecoli.library.parquet_emitter import (  # noqa: E402
    dataset_sql,
    create_duckdb_conn,
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
        with fsspec_open(os.path.join(v_data_dir, "metadata.json"), "r") as f:
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


def build_duckdb_filter(config: dict) -> str:
    """
    Build a DuckDB WHERE clause from config filters.

    Args:
        config: Configuration dictionary with filter values

    Returns:
        DuckDB WHERE clause string
    """
    duckdb_filter = []
    last_analysis_level = -1
    filter_types = list(FILTERS.keys())

    for current_analysis_level, (data_filter, data_type) in enumerate(FILTERS.items()):
        # Handle range filters
        if config.get(f"{data_filter}_range", None) is not None:
            if config.get(data_filter) is not None:
                warnings.warn(
                    f"Provided both range and value(s) for {data_filter}. "
                    "Range takes precedence."
                )
            config[data_filter] = list(
                range(
                    config[f"{data_filter}_range"][0], config[f"{data_filter}_range"][1]
                )
            )

        # Build filter clause if filter is specified
        if config.get(data_filter, None) is not None:
            # Warn about skipped filters
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

            # Build the filter clause
            if len(config[data_filter]) > 1:
                if data_type is str:
                    filter_values = "', '".join(str(i) for i in config[data_filter])
                    duckdb_filter.append(f"{data_filter} IN ('{filter_values}')")
                else:
                    filter_values = ", ".join(str(i) for i in config[data_filter])
                    duckdb_filter.append(f"{data_filter} IN ({filter_values})")
            else:
                if data_type is str:
                    quoted_val = str(config[data_filter][0])
                    duckdb_filter.append(f"{data_filter} = '{quoted_val}'")
                else:
                    duckdb_filter.append(f"{data_filter} = {config[data_filter][0]}")

            last_analysis_level = current_analysis_level

    return " AND ".join(duckdb_filter)


def load_variant_metadata(
    config: dict,
) -> tuple[dict[str, dict[int, Any]], dict[str, dict[int, str]], dict[str, str]]:
    """
    Load variant metadata from configured sources.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (variant_metadata, sim_data_dict, variant_names)

    Raises:
        KeyError: If experiment_id not in config
        AssertionError: If multiple experiment IDs without proper variant_data_dir
    """
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
        with fsspec_open(config["variant_metadata_path"], "r") as f:
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
        variant_names = {config["experiment_id"][0]: ""}
        sim_data_dict = make_sim_data_dict(
            config["experiment_id"][0],
            config.get("variant", []),
            config.get("sim_data_path", []),
        )

    return variant_metadata, sim_data_dict, variant_names


def filter_variant_dicts(
    variant_set: set[tuple[str, int]],
    variant_metadata: dict[str, dict[int, Any]],
    sim_data_dict: dict[str, dict[int, str]],
    variant_names: dict[str, str],
) -> tuple[dict[str, dict[int, Any]], dict[str, dict[int, str]], dict[str, str]]:
    """
    Filter variant dictionaries to only include variants in the given set.

    Args:
        variant_set: Set of (experiment_id, variant_id) tuples to keep
        variant_metadata: Full variant metadata dictionary
        sim_data_dict: Full sim_data dictionary
        variant_names: Variant names dictionary

    Returns:
        Tuple of (filtered_variant_metadata, filtered_sim_data_dict, filtered_variant_names)
    """
    filtered_variant_metadata: dict[str, dict[int, Any]] = {}
    filtered_sim_data_dict: dict[str, dict[int, str]] = {}
    filtered_variant_names: dict[str, str] = {}

    for exp_id, var_id in variant_set:
        if exp_id not in filtered_variant_metadata:
            filtered_variant_metadata[exp_id] = {}
            filtered_sim_data_dict[exp_id] = {}
            filtered_variant_names[exp_id] = variant_names.get(exp_id, "")

        if exp_id in variant_metadata and var_id in variant_metadata[exp_id]:
            filtered_variant_metadata[exp_id][var_id] = variant_metadata[exp_id][var_id]

        if exp_id in sim_data_dict and var_id in sim_data_dict[exp_id]:
            filtered_sim_data_dict[exp_id][var_id] = sim_data_dict[exp_id][var_id]

    return filtered_variant_metadata, filtered_sim_data_dict, filtered_variant_names


def build_query_strings(
    analysis_type: str,
    duckdb_filter: str,
    config_sql: str,
    history_sql: str,
    success_sql: str,
    outdir: str,
    conn,
) -> dict[str, tuple[str, str, str, str, set]]:
    """
    Build query strings for a given analysis type.

    Args:
        analysis_type: Type of analysis (e.g., "multivariant", "single")
        duckdb_filter: DuckDB WHERE clause
        config_sql: SQL query for config data
        history_sql: SQL query for history data
        success_sql: SQL query for success data
        outdir: Output directory path
        conn: DuckDB connection

    Returns:
        Dictionary mapping filter strings to tuples of
        (history_query, config_query, success_query, output_dir, variant_set)
    """
    id_cols = ANALYSIS_TYPES[analysis_type]
    query_strings = {}

    if len(id_cols) > 0:
        # Query for distinct id_cols and also get variants for filtering
        cols_to_select = list(id_cols)
        if "experiment_id" not in cols_to_select:
            cols_to_select.append("experiment_id")
        if "variant" not in cols_to_select:
            cols_to_select.append("variant")
        select_cols = ", ".join(cols_to_select)

        data_ids_with_variants = conn.sql(
            f"SELECT DISTINCT {select_cols} FROM ({config_sql}) WHERE {duckdb_filter}"
        ).fetchall()

        # Group by the id_cols to collect variants for each subset
        id_to_variants = defaultdict(set)
        for row in data_ids_with_variants:
            data_id = row[: len(id_cols)]
            exp_id_idx = cols_to_select.index("experiment_id")
            var_id_idx = cols_to_select.index("variant")
            exp_id = row[exp_id_idx]
            var_id = row[var_id_idx]
            id_to_variants[data_id].add((exp_id, var_id))

        for data_id, variant_set in id_to_variants.items():
            data_filters = []
            curr_outdir = os.path.abspath(outdir)
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
                variant_set,
            )
    else:
        curr_outdir = os.path.abspath(outdir)
        os.makedirs(curr_outdir, exist_ok=True)
        # For analysis types with no id_cols, query all variants matching the filter
        all_variants = conn.sql(
            f"SELECT DISTINCT experiment_id, variant"
            f" FROM ({config_sql}) WHERE {duckdb_filter}"
        ).fetchall()
        variant_set = set(all_variants)
        query_strings[duckdb_filter] = (
            f"SELECT * FROM ({history_sql}) WHERE {duckdb_filter}",
            f"SELECT * FROM ({config_sql}) WHERE {duckdb_filter}",
            f"SELECT * FROM ({success_sql}) WHERE {duckdb_filter}",
            os.path.abspath(outdir),
            variant_set,
        )

    return query_strings


def run_analysis_loop(
    config: dict,
    conn,
    history_sql: str,
    config_sql: str,
    success_sql: str,
    duckdb_filter: str,
    variant_metadata: dict,
    sim_data_dict: dict,
    variant_names: dict,
) -> tuple[dict[str, int], int | None]:
    """
    Run the main analysis loop for all configured analysis types.

    Args:
        config: Configuration dictionary with analysis_types and options
        conn: DuckDB connection
        history_sql: SQL query for history data
        config_sql: SQL query for config data
        success_sql: SQL query for success data
        duckdb_filter: DuckDB WHERE clause for filtering data
        variant_metadata: Variant metadata dictionary
        sim_data_dict: Sim data dictionary
        variant_names: Variant names dictionary

    Returns:
        Dictionary with statistics about analyses run:
        {"total_runs": N, "skipped": M, "errors": K}
    """
    stats = {"total_runs": 0, "skipped": 0, "errors": 0}
    last_exit_code: int | None = None

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
            stats["skipped"] += 1
            continue

        # Build query strings for this analysis type
        query_strings = build_query_strings(
            analysis_type,
            duckdb_filter,
            config_sql,
            history_sql,
            success_sql,
            config["outdir"],
            conn,
        )

        for analysis_name in config[analysis_type]:
            try:
                analysis_mod = importlib.import_module(
                    f"ecoli.analysis.{analysis_type}.{analysis_name}"
                )

                for data_filters, (
                    history_q,
                    config_q,
                    success_q,
                    curr_outdir,
                    variant_set,
                ) in query_strings.items():
                    print(
                        f"Running {analysis_type} {analysis_name} with {data_filters}."
                    )

                    # Filter variant_metadata and sim_data_dict to only include
                    # variants that match the current filters
                    (
                        filtered_variant_metadata,
                        filtered_sim_data_dict,
                        filtered_variant_names,
                    ) = filter_variant_dicts(
                        variant_set, variant_metadata, sim_data_dict, variant_names
                    )

                    # Create analysis-specific output directory
                    analysis_outdir = os.path.join(
                        curr_outdir, f"analysis={analysis_name}"
                    )
                    if os.path.exists(analysis_outdir):
                        raise FileExistsError(
                            f"{analysis_outdir} already exists, indicating this "
                            "analysis has been run. Please delete/move it or "
                            "specify a different output directory."
                        )
                    os.makedirs(analysis_outdir)

                    analysis_mod.plot(
                        config[analysis_type][analysis_name],
                        conn,
                        history_q,
                        config_q,
                        success_q,
                        filtered_sim_data_dict,
                        config.get("validation_data_path", []),
                        analysis_outdir,
                        filtered_variant_metadata,
                        filtered_variant_names,
                    )

                    # Write metadata.json for this analysis
                    analysis_metadata = {
                        "analysis_type": analysis_type,
                        "analysis_name": analysis_name,
                        "data_filters": data_filters,
                        "config": config[analysis_type][analysis_name],
                    }
                    with open(os.path.join(analysis_outdir, "metadata.json"), "w") as f:
                        json.dump(analysis_metadata, f)

                    stats["total_runs"] += 1
            except Exception as e:
                print(f"Error running {analysis_type} {analysis_name}: {e}")
                stats["errors"] += 1
                rc = None
                # Try to extract a return/exit code from common exception attributes
                if hasattr(e, "returncode"):
                    rc = getattr(e, "returncode")
                elif hasattr(e, "errno"):
                    rc = getattr(e, "errno")
                elif isinstance(e, SystemExit):
                    rc = e.code
                try:
                    if isinstance(rc, int) and rc != 0:
                        last_exit_code = rc
                except Exception:
                    pass

    return stats, last_exit_code


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
    parser.add_argument(
        "--analysis_name",
        nargs="*",
        help=(
            "Limit to specific analysis script name(s) within the selected "
            "analysis type(s)."
        ),
    )
    config_file = os.path.join(CONFIG_DIR_PATH, "default.json")
    args = parser.parse_args()
    with open(config_file, "r") as f:
        config = json.load(f)
    if args.config is not None:
        config_file = args.config
        with fsspec_open(args.config, "r") as f:
            SimConfig.merge_config_dicts(config, json.load(f))
    if "out_uri" not in config["emitter_arg"]:
        out_uri = os.path.abspath(config["emitter_arg"]["out_dir"])
        object_store = ""
    else:
        out_uri = config["emitter_arg"]["out_uri"]
        object_store = parse.urlparse(out_uri).scheme
        assert object_store in ("gcs", "gs", "s3"), (
            f"Unsupported URI scheme {object_store} in out_uri. Must be one of gcs, gs, or s3."
        )
    config = config["analysis_options"]
    for k, v in vars(args).items():
        if v is not None:
            config[k] = v

    analysis_names = config.get("analysis_name")
    if analysis_names:
        analysis_types = config.get("analysis_types")
        if analysis_types is None:
            analysis_types = [
                analysis_type
                for analysis_type in ANALYSIS_TYPES
                if analysis_type in config
            ]
        missing = set(analysis_names)
        selected_types: list[str] = []
        for analysis_type in analysis_types:
            if analysis_type not in config or not isinstance(
                config[analysis_type], dict
            ):
                continue
            analyses = config[analysis_type]
            filtered = {
                name: analyses[name] for name in analysis_names if name in analyses
            }
            if filtered:
                selected_types.append(analysis_type)
                missing.difference_update(filtered.keys())
            config[analysis_type] = filtered
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise KeyError(f"No analyses found for name(s): {missing_list}")
        config["analysis_types"] = selected_types

    # Set up DuckDB filters for data
    duckdb_filter = build_duckdb_filter(config)

    # Load variant metadata
    variant_metadata, sim_data_dict, variant_names = load_variant_metadata(config)

    # Create output directory
    os.makedirs(config["outdir"], exist_ok=True)

    # Establish DuckDB connection
    conn = create_duckdb_conn(config["outdir"], object_store, config.get("cpus"))
    history_sql, config_sql, success_sql = dataset_sql(out_uri, config["experiment_id"])

    # Run the analysis loop
    stats, last_exit_code = run_analysis_loop(
        config,
        conn,
        history_sql,
        config_sql,
        success_sql,
        duckdb_filter,
        variant_metadata,
        sim_data_dict,
        variant_names,
    )

    print(
        f"\nAnalysis complete: {stats['total_runs']} runs, "
        f"{stats['skipped']} skipped, {stats['errors']} errors"
    )

    # Propagate a non-zero exit code if any errors occurred. Prefer the
    # last observed non-zero exit code from exceptions; otherwise return 1.
    if stats.get("errors", 0) > 0:
        if last_exit_code is not None:
            return int(last_exit_code)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
