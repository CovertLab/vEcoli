import argparse
import copy
import hashlib
import json
import os
import pathlib
import random
import shutil
import subprocess
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional
from urllib import parse

# Try to import fsspec, but make it optional
try:
    from fsspec import url_to_fs
    from fsspec.spec import AbstractFileSystem

    FSSPEC_AVAILABLE = True
except ImportError:
    FSSPEC_AVAILABLE = False


@dataclass
class ClusterConfig:
    """Resolved configuration for running on a SLURM HPC cluster."""

    name: str
    config_dict: dict
    build_options: dict[str, Any]
    nextflow_options: dict[str, Any]
    nf_config_overrides: dict[str, Any]
    nf_setup: Optional[str] = None

    def require_container_image(self) -> str:
        image = self.config_dict.get("container_image")
        if image is None:
            raise RuntimeError(
                f"Must supply name for container image when using the {self.name} cluster."
            )
        return image

    @property
    def build_image(self) -> bool:
        return self.config_dict.get("build_image", False)

    @property
    def jenkins(self) -> bool:
        return self.config_dict.get("jenkins", False)

    def apply_config_overrides(self, nf_config: str) -> str:
        return _apply_nf_config_overrides(nf_config, self.nf_config_overrides)

    @staticmethod
    def _coerce_positive_int(value: Any, field_name: str) -> int:
        try:
            value_int = int(value)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"{field_name} must be an integer greater than zero"
            ) from err
        if value_int <= 0:
            raise ValueError(f"{field_name} must be an integer greater than zero")
        return value_int


def hyperqueue_snippets(outdir: str) -> tuple[str, str]:
    """Return init and exit shell snippets for HyperQueue if enabled."""
    hq_server_dir = os.path.join(outdir, ".hq-server")
    journal_path = os.path.join(hq_server_dir, "journal")
    init = f"""
# Set the directory which HyperQueue will use 
export HQ_SERVER_DIR={hq_server_dir}
mkdir -p ${{HQ_SERVER_DIR}}

# Start the server in the background (&) and wait until it has started
hq server start --journal {journal_path} &
until hq job list &>/dev/null ; do sleep 1 ; done

"""
    exit_cmd = "hq job wait all; hq worker stop all; hq server stop"
    return init, exit_cmd


_JAVA_SETUP = """
export JAVA_HOME=$HOME/.local/bin/java-22
export PATH=$JAVA_HOME/bin:$HOME/.local/bin:$PATH
"""

CLUSTER_PRESETS: dict[str, dict[str, Any]] = {
    "sherlock": {
        "build_image": {
            "time": "00:30:00",
            "cpus-per-task": 2,
            "mem": "8GB",
            "partition": "owners,normal",
            # Save stdout and stderr to same file so can stream
            # all build output to command line by reading one file
            "output": "{outdir}/build_image_{experiment_id}.out",
            "error": "{outdir}/build_image_{experiment_id}.out",
        },
        "nf_config_overrides": {
            # Restrict to newer CPU generations
            # See https://github.com/CovertLab/vEcoli/pull/331
            "CLUSTER_OPTIONS": {
                "prefer": '"CPU_GEN:GEN|CPU_GEN:SPR"',
                "constraint": '"CPU_GEN:RME|CPU_GEN:MLN|CPU_GEN:BGM|CPU_GEN:SIE|CPU_GEN:GEN|CPU_GEN:SPR"',
            },
            "QUEUE": "owners,normal",
        },
        "nextflow": {
            "time": "7-00:00:00",
            "cpus-per-task": 1,
            "mem": "4GB",
            # Run nextflow on lab partition for guaranteed
            # resources, no preemption, longer time limit
            "partition": "mcovert",
            "output": "{outdir}/nf_{experiment_id}.out",
            "error": "{outdir}/nf_{experiment_id}.err",
        },
    },
    "carina": {
        "build_image": {
            "time": "00:30:00",
            "cpus-per-task": 2,
            "mem": "8GB",
            "partition": "normal",
            # Save stdout and stderr to same file so can stream
            # all build output to command line by reading one file
            "output": "{outdir}/build_image_{experiment_id}.out",
            "error": "{outdir}/build_image_{experiment_id}.out",
        },
        "nf_config_overrides": {
            "CLUSTER_OPTIONS": {},
            "QUEUE": "normal",
        },
        "nextflow": {
            "time": "2-00:00:00",
            "cpus-per-task": 1,
            "mem": "4GB",
            # Consider running on long partition for 7 day limit
            # Only 1 long job allowed so cannot if running Jenkins
            "partition": "normal",
            "output": "{outdir}/nf_{experiment_id}.out",
            "error": "{outdir}/nf_{experiment_id}.err",
        },
    },
    "ccam": {
        # Environment variables to load and substitute in options below
        "env_vars": {
            "partition": "SLURM_PARTITION",
            "qos": "SLURM_QOS",
            "nodelist": "SLURM_NODE_LIST",
            "slurm_log_dir": "SLURM_LOG_BASE_PATH",
        },
        "build_image": {
            "time": "00:30:00",
            "cpus-per-task": 2,
            "mem": "8GB",
            "partition": "{partition}",
            "output": "{slurm_log_dir}/build_image_{experiment_id}.out",
            "error": "{slurm_log_dir}/build_image_{experiment_id}.err",
        },
        "nf_config_overrides": {
            "CLUSTER_OPTIONS": {
                # qos was not set correctly in original implementation
                "qos": "{qos}",
                "nodelist": "{nodelist}",
            },
            "QUEUE": "{partition}",
        },
        "nextflow": {
            "time": "7-00:00:00",
            "cpus-per-task": 1,
            "mem": "4GB",
            "partition": "{partition}",
            "output": "{slurm_log_dir}/nf_{experiment_id}.out",
            "error": "{slurm_log_dir}/nf_{experiment_id}.err",
            "qos": "{qos}",
            "mail-type": "ALL",
            "nodelist": "{nodelist}",
        },
        "nf_setup": _JAVA_SETUP,
    },
    "aws_cdk": {
        # Environment variables to load and substitute in options below
        "env_vars": {
            "partition": "SLURM_PARTITION",
            "slurm_log_dir": "SLURM_LOG_BASE_PATH",
        },
        "build_image": {
            "time": "00:30:00",
            "cpus-per-task": 2,
            "mem": "8GB",
            "partition": "{partition}",
            "output": "{slurm_log_dir}/build_image_{experiment_id}.out",
            "error": "{slurm_log_dir}/build_image_{experiment_id}.err",
        },
        "nf_config_overrides": {
            "QUEUE": "{partition}",
        },
        "nextflow": {
            "time": "7-00:00:00",
            "cpus-per-task": 1,
            "mem": "4GB",
            # This partition was hard-coded in the original implementation
            "partition": "jobs-queue",
            "output": "{slurm_log_dir}/nf_{experiment_id}.out",
            "error": "{slurm_log_dir}/nf_{experiment_id}.err",
        },
        "nf_setup": _JAVA_SETUP,
    },
}
"""Default config values for different SLURM clusters.

Each key is a cluster name and each value is a dictionary with the following
structure::

    {
        # Environment variables to load for string substitution in any of
        # the options in the following sections. Note that the variables
        # "outdir" (experiment output directory from emitter_arg --> out_dir)
        # and "experiment_id" are always available for substitution.
        "env_vars": { ... },
        # SLURM options for building container image
        "build_image": { ... },
        # Strings to substitute in Nextflow config file (only QUEUE and
        # CLUSTER_OPTIONS are required, others use defaults from
        # configs/default.json)
        "nf_config_overrides": {
            # SLURM partition to run HyperQueue workers and non-HyperQueue jobs on
            "QUEUE": str,
            # Number of cores to allocate each HyperQueue worker
            "HQ_CORES": int,
            # Number of CPUs to allocate per simulation (max 2)
            "SIM_CPUS": int,
            # Amount of memory to allocate per simulation in GB
            "SIM_MEM": int,
            # Whether to use HyperQueue for simulation job scheduling
            "HYPERQUEUE": bool,
            # Additional cluster options to pass to SLURM
            "CLUSTER_OPTIONS": dict[str, str],
        },
        # SLURM options for Nextflow job
        "nextflow": { ... },
        # Shell commands to setup environment in Nextflow job script
        "nf_setup": str
    }

:meta hide-value:
"""


class _FormatDict(dict):
    def __missing__(self, key):
        """Handle missing keys during string formatting.

        This method is called by :meth:`str.format_map` when a placeholder
        references a key that is not present in the formatting context.
        By raising :class:`KeyError`, it allows callers to detect and track
        missing substitutions for error reporting.
        """
        raise KeyError(key)


_MISSING_SENTINEL = object()


def _format_recursive(
    value,
    formatter: _FormatDict,
    path: list[str],
    on_missing: Callable[[list[str], str], None],
):
    """Recursively apply string formatting to nested configuration values.

    This helper walks over ``value``, which may be a plain value, string, or
    nested mapping, and formats any strings using the provided ``formatter``
    (typically a :class:`_FormatDict` used with ``str.format_map``).

    The current location within the nested structure is tracked via ``path``,
    which is extended as the recursion descends into dictionaries. When a
    placeholder in a string cannot be resolved by ``formatter``, a
    :class:`KeyError` is raised, the missing placeholder name is reported to
    the ``on_missing`` callback together with the corresponding ``path``, and
    the function returns the internal ``_MISSING_SENTINEL`` so that the
    missing entry can be skipped by the caller.

    Non-string, non-mapping values are returned unchanged.
    """
    if value is None:
        return None
    elif isinstance(value, str):
        try:
            return value.format_map(formatter)
        except KeyError as err:
            missing = err.args[0]
            on_missing(path, missing)
            return _MISSING_SENTINEL
    elif isinstance(value, dict):
        dict_result = {}
        for key, sub_value in value.items():
            formatted = _format_recursive(
                sub_value,
                formatter,
                path + [str(key)],
                on_missing,
            )
            if formatted is _MISSING_SENTINEL:
                continue
            dict_result[key] = formatted
        return dict_result
    return value


def _format_template_section(
    section: Any, context: dict[str, Any], cluster_name: str, section_name: str
):
    """Format a configuration or template section using the provided context.

    The ``section`` value (which may be a string, mapping, or nested structure)
    is traversed and any ``str.format``-style placeholders are resolved using
    keys from ``context``. If a placeholder cannot be resolved, the
    corresponding entry is skipped from the resulting structure and a warning
    is emitted indicating the missing placeholder and its location within
    ``cluster_name.section_name``.

    Args:
        section: The configuration or template section to format. May be
            ``None``, in which case ``None`` is returned.
        context: Mapping of placeholder names to their replacement values.
        cluster_name: Name of the cluster, used only for constructing
            human-readable warning messages.
        section_name: Name of the section being formatted, used in warning
            messages and as the root path when reporting missing placeholders.

    Returns:
        The formatted section with all resolvable placeholders substituted and
        entries with missing placeholders omitted, or ``None`` if ``section``
        is ``None``.
    """
    if section is None:
        return None
    formatter = _FormatDict(context)

    def _warn_missing(path_parts: list[str], placeholder: str):
        location = ".".join(path_parts)
        warnings.warn(
            (
                f"Skipping '{location}' in {cluster_name}.{section_name} because "
                f"placeholder '{{{placeholder}}}' is not available."
            ),
            stacklevel=2,
        )

    formatted = _format_recursive(section, formatter, [section_name], _warn_missing)
    return formatted


def _load_cluster_env_values(
    cluster_name: str, env_var_map: dict[str, str]
) -> dict[str, str]:
    """Load environment variable values for a cluster configuration.

    This helper resolves placeholders used in a cluster's configuration by reading
    their corresponding environment variables. For each entry in ``env_var_map``,
    it looks up the environment variable name and, if the value is set and
    non-empty, includes it in the returned mapping. When an environment variable
    is missing or empty, a warning is emitted and any options that depend on the
    associated placeholder will be skipped.

    Args:
        cluster_name: Name of the cluster whose configuration is being prepared.
        env_var_map: Mapping from placeholder names to environment variable names.

    Returns:
        A mapping from placeholder names to resolved environment variable values
        for the given cluster.
    """
    values: dict[str, str] = {}
    for placeholder, env_var in env_var_map.items():
        env_value = os.getenv(env_var)
        if env_value is None or env_value == "":
            warnings.warn(
                (
                    f"Environment variable '{env_var}' is not set for the {cluster_name} cluster; "
                    f"options requiring '{{{placeholder}}}' will be skipped."
                ),
                stacklevel=2,
            )
            continue
        values[placeholder] = env_value
    return values


def _render_slurm_directives(options: dict[str, Any]) -> str:
    """Convert a mapping of SLURM options into `#SBATCH` directive lines.

    Each key in ``options`` is treated as a SLURM option name (without the leading
    ``--``). Values that are ``None`` or the empty string are skipped entirely.
    Boolean values are interpreted as flags: if the value is ``True``, a directive
    of the form ``#SBATCH --<key>`` is emitted; if ``False``, the option is
    omitted. All other values are rendered as ``#SBATCH --<key>=<value>``.

    The returned string contains one directive per line, separated by newlines.
    """
    directives: list[str] = []
    for key, value in options.items():
        if value in (None, ""):
            continue
        if isinstance(value, bool):
            if value:
                directives.append(f"#SBATCH --{key}")
            continue
        directives.append(f"#SBATCH --{key}={value}")
    return "\n".join(directives)


def _serialize_cluster_options(options: dict[str, Any]) -> str:
    """Serialize cluster options into a command-line string.

    Each dictionary item is converted into a ``--key=value`` flag. Keys that do
    not already start with ``"--"`` are automatically prefixed. Options with a
    value of ``None`` or the empty string are skipped. Values containing
    whitespace are wrapped in double quotes so they are treated as a single
    argument by the shell.
    """
    parts: list[str] = []
    for key, value in options.items():
        if value in (None, ""):
            continue
        flag = key if key.startswith("--") else f"--{key}"
        value_str = str(value)
        if any(char.isspace() for char in value_str):
            value_str = f'"{value_str}"'
        parts.append(f"{flag}={value_str}")
    return " ".join(parts)


def _apply_nf_config_overrides(nf_config: str, overrides: dict[str, Any]) -> str:
    """Apply configuration overrides to a Nextflow config string.

    The keys in ``overrides`` are treated as literal placeholders in ``nf_config``
    and replaced with their stringified values. The special key
    ``"CLUSTER_OPTIONS"`` may contain a dictionary of options, which is
    serialized into a command-line style string via ``_serialize_cluster_options``.
    Boolean values are converted to their lowercase string representation
    (``"true"`` or ``"false"``) before substitution to match JSON-style booleans.
    """
    result = nf_config
    for key, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, dict):
            if key != "CLUSTER_OPTIONS":
                raise ValueError(
                    "Only 'CLUSTER_OPTIONS' can be a dictionary in Nextflow config overrides."
                )
            replacement = _serialize_cluster_options(value)
        elif isinstance(value, bool):
            # JSON bool is lowercase
            replacement = str(value).lower()
        else:
            replacement = str(value)
        result = result.replace(key, replacement)
    return result


LIST_KEYS_TO_MERGE = (
    "save_times",
    "add_processes",
    "exclude_processes",
    "processes",
    "engine_process_reports",
    "initial_state_overrides",
)
"""
Special configuration keys that are list values which are concatenated
together when they are found in multiple sources (e.g. default JSON and
user-specified JSON) instead of being directly overriden.
"""

# Resource-only configuration keys that do not affect simulation output.
# These are stripped from the config passed to Nextflow processes so that
# changing them does not invalidate the cache and force re-runs.
# The full config (including these keys) is still saved for reference.
RESOURCE_ONLY_KEYS = {
    # Top-level keys
    "SIM_CPUS",
    "SIM_MEM",
    "SIM_TIME",
    "HYPERQUEUE",
    "HQ_CORES",
}
RESOURCE_ONLY_NESTED_KEYS = {
    # Keys under parca_options
    "parca_options": {"cpus", "memory_gb", "slurm_time_hrs"},
    # Keys under analysis_options
    "analysis_options": {"memory_gb", "slurm_time_hrs", "cpus", "duckdb_threads"},
}


def strip_resource_keys(config: dict) -> dict:
    """
    Create a copy of the config with resource-only keys removed.

    This allows changing resource allocations (memory, CPUs, time limits)
    without invalidating Nextflow's cache, since these keys should not
    affect simulation output.

    Args:
        config: Full configuration dictionary

    Returns:
        Config dictionary with resource-only keys stripped
    """
    stripped = copy.deepcopy(config)

    # Remove top-level resource-only keys
    for key in RESOURCE_ONLY_KEYS:
        stripped.pop(key, None)

    # Remove nested resource-only keys
    for parent_key, child_keys in RESOURCE_ONLY_NESTED_KEYS.items():
        if parent_key in stripped and isinstance(stripped[parent_key], dict):
            for child_key in child_keys:
                stripped[parent_key].pop(child_key, None)

    return stripped


CONFIG_DIR_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
)
NEXTFLOW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nextflow")

# These input channels calculate the values that the analysis jobs defined in
# runscripts/nextflow/analysis.nf consume.
MULTIDAUGHTER_CHANNEL = """
    generationSize = {gen_size}
    simCh
        .map {{ tuple(groupKey(it[2..5], generationSize[it[5]]), it[0], it[1], it[2], it[3], it[4], it[5] ) }}
        .groupTuple(remainder: true)
        .map {{ tuple(it[1][0], it[2][0], it[3][0], it[4][0], it[5][0], it[6][0], it[1].size()) }}
        .set {{ multiDaughterCh }}
"""
MULTIGENERATION_CHANNEL = """
    simCh
        .groupTuple(by: [2, 3, 4], size: {size}, remainder: true)
        .map {{ tuple(it[0][0], it[1][0], it[2], it[3], it[4], it[0].size()) }}
        .set {{ multiGenerationCh }}
"""
MULTISEED_CHANNEL = """
    simCh
        .groupTuple(by: [2, 3], size: {size}, remainder: true)
        .map {{ tuple(it[0][0], it[1][0], it[2], it[3], it[0].size()) }}
        .set {{ multiSeedCh }}
"""
MULTIVARIANT_CHANNEL = """
    // Group once to deduplicate variant names and pickles
    // Group again into single value for entire experiment
    simCh
        .groupTuple(by: [2, 3], size: {size}, remainder: true)
        .map {{ tuple(it[0][0], it[1][0], it[2], it[3], it[0].size()) }}
        .groupTuple(by: [2])
        .map {{ tuple(it[0], it[1], it[2], it[3], it[4].sum()) }}
        .set {{ multiVariantCh }}
"""


def load_config_with_inheritance(config_path: str) -> dict:
    """
    Load a config file and recursively resolve all inheritance chains.

    Priority order: Current config > First inherited > ... > Last inherited
    If config A inherits from [B, D] and B inherits from [C]:
    Priority is A > B > C > D

    Args:
        config_path: Path to the configuration file

    Returns:
        Fully resolved configuration dictionary
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    if "inherit_from" not in config:
        return config

    # Build inheritance chain in reverse priority order (lowest to highest)
    inherit_chain = []
    for inherit_path in reversed(config["inherit_from"]):
        # Recursively load inherited config
        inherited = load_config_with_inheritance(
            os.path.join(CONFIG_DIR_PATH, inherit_path)
        )
        inherit_chain.append(inherited)

    # Start with empty base and apply configs from lowest to highest priority
    result: dict = {}
    for inherited_config in inherit_chain:
        _merge_configs(result, inherited_config)

    # Finally apply current config (highest priority)
    _merge_configs(result, config)

    return result


def _merge_configs(base_config: dict, overlay_config: dict):
    """
    Merge overlay_config into base_config, with overlay taking priority.
    Mutates base_config in place.

    Args:
        base_config: Configuration to update (lower priority)
        overlay_config: Configuration to merge in (higher priority)
    """
    for key, value in overlay_config.items():
        if key in LIST_KEYS_TO_MERGE:
            # For list keys, concatenate and deduplicate
            base_config.setdefault(key, [])
            base_config[key].extend(value)
            if key == "engine_process_reports":
                base_config[key] = [tuple(path) for path in base_config[key]]
            # Remove duplicates and sort
            base_config[key] = sorted(list(set(base_config[key])))
        elif (
            isinstance(value, dict)
            and key in base_config
            and isinstance(base_config[key], dict)
        ):
            # Recursively merge nested dictionaries
            _merge_configs(base_config[key], value)
        else:
            # Overlay value takes priority
            base_config[key] = value


def parse_uri(uri: str) -> tuple[Optional["AbstractFileSystem"], str]:
    """
    Parse URI and return appropriate filesystem and path.

    For cloud/remote URIs (when fsspec is available), returns fsspec filesystem.
    For local paths, returns None and absolute path.
    """
    if not FSSPEC_AVAILABLE:
        if parse.urlparse(uri).scheme in ("local", "file", ""):
            return None, os.path.abspath(uri)
        raise RuntimeError(
            "fsspec is not available. Please install fsspec to use remote URIs."
        )
    return url_to_fs(uri)


def compute_file_hash(path: str, chunk_size: int = 8192) -> str:
    """
    Compute SHA256 hash of a file.

    Works with both local files and cloud URIs (via fsspec when available).

    Args:
        path: Local path or cloud URI to the file
        chunk_size: Size of chunks to read at a time

    Returns:
        Hexadecimal SHA256 hash of the file contents
    """
    hasher = hashlib.sha256()
    fs, resolved_path = parse_uri(path)

    if fs is None:
        # Local file
        with open(resolved_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hasher.update(chunk)
    else:
        # Cloud file via fsspec
        with fs.open(resolved_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hasher.update(chunk)

    return hasher.hexdigest()


def merge_dicts(a, b):
    """
    Recursively merges dictionary b into dictionary a.
    This mutates dictionary a.
    """
    for key, value in b.items():
        if isinstance(value, dict) and key in a and isinstance(a[key], dict):
            # If both values are dictionaries, recursively merge
            merge_dicts(a[key], value)
        else:
            # Otherwise, overwrite or add the value from b to a
            a[key] = value


def generate_colony(seeds: int):
    """
    Create strings to import and compose Nextflow processes for colony sims.
    """
    return [], []


def generate_lineage(
    seed: int,
    n_init_sims: int,
    generations: int,
    single_daughters: bool,
    analysis_config: dict[str, dict[str, dict]],
    different_seeds_per_variant: bool = False,
):
    """
    Create strings to import and compose Nextflow processes for lineage sims:
    cells that divide for a number of generations but do not interact. Also
    contains import statements and workflow jobs for analysis scripts.

    Args:
        seed: First seed for first sim
        n_init_sims: Number of sims to initialize with different seeds
        generations: Number of generations to run for each seed
        single_daughters: If True, only simulate one daughter cell each gen
        different_seeds_per_variant: If True, each variant ``i`` is given seeds
            ``[seed + i*n_init_sims, seed + (i+1)*n_init_sims)`` so that
            different variants simulate statistically independent cells.
            If False (default), all variants share the same seed range
            ``[seed, seed + n_init_sims)``.
        analysis_config: Dictionary with any of the following keys::

            {
                'variant': analyses to run on output of all cells combined,
                'cohort': analyses to run on output grouped by variant,
                'multigen': analyses to run on output grouped by variant & seed,
                'single': analyses to run on output for each individual cell,
                'parca': analyses to run on parameter calculator output
            }

            Each key corresponds to a mapping from analysis name (as defined
            in ``ecol/analysis/__init__.py``) to keyword arguments.

    Returns:
        2-element tuple containing

        - **sim_imports**: All `include` statements for Nextflow sim processes
        - **sim_workflow**: Fully composed workflow for entire lineage
    """
    sim_imports = []
    if different_seeds_per_variant:
        # Emit relative seeds 0..<n_init_sims; absolute seeds are computed
        # per-variant in the gen-0 map below.
        sim_workflow = [f"\tchannel.of( 0..<{n_init_sims} ).set {{ seedCh }}"]
    else:
        sim_workflow = [
            f"\tchannel.of( {seed}..<{seed + n_init_sims} ).set {{ seedCh }}"
        ]

    all_sim_tasks = []
    for gen in range(generations):
        name = f"sim_gen_{gen + 1}"
        # Handle special case of 1st generation
        if gen == 0:
            sim_imports.append(
                f"include {{ simGen0 as {name} }} from '{NEXTFLOW_DIR}/sim'"
            )
            # variantCh emits (config_uri, config_hash, sim_data_uri, sim_data_hash, variant_name)
            # Combine with seedCh for lineage_seed, then add generation=1
            if different_seeds_per_variant:
                # Offset absolute seed by (variant index) * (# of seeds) so each variant gets
                # a distinct, non-overlapping seed range.
                sim_workflow.append(
                    f"\t{name}(variantCh.combine(seedCh)"
                    f".map {{ it[0..4] + [{seed} + it[4].toInteger() * {n_init_sims} + it[5], 1] }}, '0')"
                )
            else:
                sim_workflow.append(
                    (f"\t{name}(variantCh.combine(seedCh).map {{ it + [1] }}, '0')")
                )
            all_sim_tasks.append(f"{name}.out.metadata")
            if not single_daughters:
                sim_workflow.append(
                    f"\t{name}.out.nextGen0.mix({name}.out.nextGen1).set {{ {name}_nextGen }}"
                )
            else:
                sim_workflow.append(f"\t{name}.out.nextGen0.set {{ {name}_nextGen }}")
            continue
        sim_imports.append(f"include {{ sim as {name} }} from '{NEXTFLOW_DIR}/sim'")
        parent = f"sim_gen_{gen}"
        sim_workflow.append(f"\t{name}({parent}_nextGen)")
        if not single_daughters:
            sim_workflow.append(
                f"\t{name}.out.nextGen0.mix({name}.out.nextGen1).set {{ {name}_nextGen }}"
            )
        else:
            sim_workflow.append(f"\t{name}.out.nextGen0.set {{ {name}_nextGen }}")
        all_sim_tasks.append(f"{name}.out.metadata")

    # Channel that combines metadata for all sim tasks
    if len(all_sim_tasks) > 1:
        tasks = all_sim_tasks[0]
        other_tasks = ", ".join(all_sim_tasks[1:])
        sim_workflow.append(f"\t{tasks}.mix({other_tasks}).set {{ simCh }}")
    else:
        sim_workflow.append(f"\t{all_sim_tasks[0]}.set {{ simCh }}")

    sims_per_seed = generations if single_daughters else 2**generations - 1

    def _analysis_names(analysis_type: str) -> list[str]:
        analyses = analysis_config.get(analysis_type, {})
        if not isinstance(analyses, dict):
            return []
        return list(analyses.keys())

    def _append_analysis_channel(channel_name: str, analysis_names: list[str]) -> None:
        if not analysis_names:
            return
        names_list = ", ".join(json.dumps(name) for name in analysis_names)
        sim_workflow.append(f"\tChannel.of({names_list}).set {{ {channel_name} }}")

    multivariant_names = _analysis_names("multivariant")
    if multivariant_names:
        # Channel that groups all sim tasks
        sim_workflow.append(
            MULTIVARIANT_CHANNEL.format(size=sims_per_seed * n_init_sims)
        )
        _append_analysis_channel("multiVariantAnalysisNameCh", multivariant_names)
        sim_workflow.append("\tmultiVariantCh.combine(multiVariantAnalysisNameCh)")
        sim_workflow.append("\t    .set { multiVariantAnalysisCh }")
        sim_workflow.append(
            "\tanalysisMultiVariant(parca_out, multiVariantAnalysisCh, "
            "variantMetadataCh)"
        )
        sim_imports.append(
            f"include {{ analysisMultiVariant }} from '{NEXTFLOW_DIR}/analysis'"
        )

    multiseed_names = _analysis_names("multiseed")
    if multiseed_names:
        # Channel that groups sim tasks by variant sim_data
        sim_workflow.append(MULTISEED_CHANNEL.format(size=sims_per_seed * n_init_sims))
        _append_analysis_channel("multiSeedAnalysisNameCh", multiseed_names)
        sim_workflow.append("\tmultiSeedCh.combine(multiSeedAnalysisNameCh)")
        sim_workflow.append("\t    .set { multiSeedAnalysisCh }")
        sim_workflow.append(
            "\tanalysisMultiSeed(parca_out, multiSeedAnalysisCh, variantMetadataCh)"
        )
        sim_imports.append(
            f"include {{ analysisMultiSeed }} from '{NEXTFLOW_DIR}/analysis'"
        )

    multigeneration_names = _analysis_names("multigeneration")
    if multigeneration_names:
        # Channel that groups sim tasks by variant sim_data and initial seed
        sim_workflow.append(MULTIGENERATION_CHANNEL.format(size=sims_per_seed))
        _append_analysis_channel("multiGenerationAnalysisNameCh", multigeneration_names)
        sim_workflow.append(
            "\tmultiGenerationCh.combine(multiGenerationAnalysisNameCh)"
        )
        sim_workflow.append("\t    .set { multiGenerationAnalysisCh }")
        sim_workflow.append(
            "\tanalysisMultiGeneration(parca_out, multiGenerationAnalysisCh, "
            "variantMetadataCh)"
        )
        sim_imports.append(
            f"include {{ analysisMultiGeneration }} from '{NEXTFLOW_DIR}/analysis'"
        )

    multidaughter_names = _analysis_names("multidaughter")
    if multidaughter_names and not single_daughters:
        # Channel that groups sim tasks by variant sim_data, initial seed, and generation
        # When simulating both daughters, will have >1 cell for generation >1
        gen_size = (
            "[" + ", ".join([f"{g + 1}: {2**g}" for g in range(generations)]) + "]"
        )
        sim_workflow.append(MULTIDAUGHTER_CHANNEL.format(gen_size=gen_size))
        _append_analysis_channel("multiDaughterAnalysisNameCh", multidaughter_names)
        sim_workflow.append("\tmultiDaughterCh.combine(multiDaughterAnalysisNameCh)")
        sim_workflow.append("\t    .set { multiDaughterAnalysisCh }")
        sim_workflow.append(
            "\tanalysisMultiDaughter(parca_out, multiDaughterAnalysisCh, "
            "variantMetadataCh)"
        )
        sim_imports.append(
            f"include {{ analysisMultiDaughter }} from '{NEXTFLOW_DIR}/analysis'"
        )

    # Single analyses are batched in a single job per cell as they should all
    # be fast enough to run that it is not worth incurring scheduling overhead
    # to parallelize them. Using the same DuckDB connection may also come with
    # some performance benefits due to caching.
    if analysis_config.get("single", False):
        sim_workflow.append("\tanalysisSingle(parca_out, simCh, variantMetadataCh)")
        sim_imports.append(
            f"include {{ analysisSingle }} from '{NEXTFLOW_DIR}/analysis'"
        )

    if analysis_config.get("parca", False):
        sim_workflow.append("\tanalysisParca(parca_out)")

    return sim_imports, sim_workflow


def generate_code(config):
    sim_data_path = config.get("sim_data_path")
    if sim_data_path is not None:
        # Pre-existing sim_data: compute hashes for cache invalidation
        kb_dir = os.path.dirname(sim_data_path)
        kb_hash = compute_file_hash(sim_data_path)
        # Compute config hash from stripped config (same as parca would)
        stripped = strip_resource_keys(config)
        config_hash = hashlib.sha256(
            json.dumps(stripped, sort_keys=True).encode()
        ).hexdigest()
        run_parca = [
            f"\tfile('{kb_dir}').copyTo(\"${{params.publishDir}}/${{params.experimentId}}/parca/kb\")",
            # Create parca_out channel with config URI, config hash, kb URI, kb hash
            f"\tChannel.of(tuple(params.config, '{config_hash}', '{kb_dir}', '{kb_hash}')).set {{ parca_out }}",
        ]
    else:
        run_parca = [
            "\trunParca(params.config)",
            "\trunParca.out.parca_out.set { parca_out }",
        ]
    seed = config.get("seed", 0)
    generations = config.get("generations", 0)
    if generations:
        lineage_seed = config.get("lineage_seed", 0)
        n_init_sims = config.get("n_init_sims")
        print(
            f"Specified generations: initial lineage seed {lineage_seed}, {n_init_sims} initial sims"
        )
        single_daughters = config.get("single_daughters", True)
        sim_imports, sim_workflow = generate_lineage(
            lineage_seed,
            n_init_sims,
            generations,
            single_daughters,
            config.get("analysis_options", {}),
            config.get("different_seeds_per_variant", False),
        )
    else:
        sim_imports, sim_workflow = generate_colony(seed)
    return "\n".join(run_parca), "\n".join(sim_imports), "\n".join(sim_workflow)


def get_cluster_config(
    config: dict, outdir: str, experiment_id: str
) -> Optional[ClusterConfig]:
    """Resolve cluster settings using CLUSTER_PRESETS description."""

    selected_clusters: list[tuple[str, dict]] = []
    for name in CLUSTER_PRESETS:
        cluster_values = config.get(name)
        if cluster_values is not None:
            selected_clusters.append((name, cluster_values))

    if not selected_clusters:
        return None
    if len(selected_clusters) > 1:
        raise RuntimeError(
            "Multiple cluster configurations detected. Please specify only one cluster in the config."
        )

    cluster_name, cluster_values = selected_clusters[0]
    preset = CLUSTER_PRESETS[cluster_name]

    env_values = _load_cluster_env_values(cluster_name, preset.get("env_vars", {}))
    context = {"outdir": outdir, "experiment_id": experiment_id, **env_values}

    build_options = (
        _format_template_section(
            preset.get("build_image", {}), context, cluster_name, "build_image"
        )
        or {}
    )
    nextflow_options = (
        _format_template_section(
            preset.get("nextflow", {}), context, cluster_name, "nextflow"
        )
        or {}
    )
    nf_config_overrides = (
        _format_template_section(
            preset.get("nf_config_overrides", {}),
            context,
            cluster_name,
            "nf_config_overrides",
        )
        or {}
    )
    nf_setup = _format_template_section(
        preset.get("nf_setup"), context, cluster_name, "nf_setup"
    )

    return ClusterConfig(
        name=cluster_name,
        config_dict=cluster_values,
        build_options=build_options,
        nextflow_options=nextflow_options,
        nf_config_overrides=nf_config_overrides,
        nf_setup=nf_setup,
    )


def build_image_cmd(image_name, apptainer=False) -> list[str]:
    build_script = os.path.join(
        os.path.dirname(__file__), "container", "build-image.sh"
    )
    cmd = [build_script, "-i", image_name]
    if apptainer:
        cmd.append("-a")
    return cmd


def _ecr_image_exists(repo_name: str, image_tag: str, region: str) -> bool:
    """Return True if the given tag already exists in an ECR repository."""
    result = subprocess.run(
        [
            "aws",
            "ecr",
            "describe-images",
            "--repository-name",
            repo_name,
            "--image-ids",
            f"imageTag={image_tag}",
            "--region",
            region,
        ],
        capture_output=True,
    )
    return result.returncode == 0


def _gcloud_image_exists(full_image_uri: str) -> bool:
    """Return True if the given image already exists in GCP Artifact Registry."""
    result = subprocess.run(
        ["gcloud", "artifacts", "docker", "images", "describe", full_image_uri],
        capture_output=True,
    )
    return result.returncode == 0


def _confirm_overwrite(image: str) -> bool:
    """Prompt the user to confirm building and pushing container image to
    an existing AWS ECR repository or GCP Artifact Registry manifest.

    Returns True if the user confirms, False otherwise.
    """
    response = (
        input(f"Image '{image}' already exists in the registry. Continue? [y/N] ")
        .strip()
        .lower()
    )
    return response in ("y", "yes")


def run_ecr_script(image: str, build: bool, region: str = "us-gov-west-1") -> str:
    """
    Run the ECR build script to either build/push or just resolve the URI.

    Args:
        image: Image specification, either full URI or repo:tag format.
        build: If True, build and push the image. If False, just resolve the URI.
        region: AWS region for ECR (e.g., 'us-gov-west-1' for GovCloud).

    Returns:
        Full ECR image URI.
    """
    build_script = os.path.join(
        os.path.dirname(__file__), "container", "build-and-push-ecr.sh"
    )

    # Parse the container_image to extract repo name and tag
    # Expected format: <account>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>
    # or just <repo>:<tag> (script will create full URI)
    is_full_ecr_uri = False
    if "/" in image:
        # Extract hostname from URI (part before first /)
        hostname = image.split("/")[0]
        # Verify hostname actually ends with .amazonaws.com to prevent
        # bypass via URLs like evil.com/.amazonaws.com/path
        is_full_ecr_uri = hostname.endswith(".amazonaws.com")

    if is_full_ecr_uri:
        # Full URI provided, extract repo:tag
        repo_and_tag = image.split("/")[-1]
    else:
        repo_and_tag = image

    if ":" in repo_and_tag:
        repo_name, image_tag = repo_and_tag.rsplit(":", 1)
    else:
        repo_name = repo_and_tag
        image_tag = "latest"

    cmd = [build_script, "-i", image_tag, "-r", repo_name, "-R", region]
    if not build:
        cmd.append("-u")  # URI-only mode

    if build:
        if _ecr_image_exists(repo_name, image_tag, region) and not _confirm_overwrite(
            f"{repo_name}:{image_tag}"
        ):
            raise SystemExit("Aborted: will not supersede existing ECR image.")
        print(
            f"Building and pushing Docker image to ECR: {repo_name}:{image_tag} (region: {region})"
        )
    else:
        print(f"Resolving ECR image URI for: {repo_name}:{image_tag}")

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    if not build:
        # In URI-only mode, the script outputs just the URI
        return result.stdout.strip()

    # In build mode, extract the full image URI from the script output
    for line in result.stdout.split("\n"):
        if "Full Image URI:" in line:
            full_uri = line.split("Full Image URI:")[-1].strip()
            return full_uri

    # Fallback: return the original image
    return image


def build_cluster_container_image(
    cluster_config: ClusterConfig,
    experiment_id: str,
    local_outdir: str,
    thread_executor: ThreadPoolExecutor,
) -> None:
    """
    Build container image on HPC cluster using SLURM batch job.
    """
    if not cluster_config.build_image:
        return

    container_image = cluster_config.require_container_image()
    image_dir = os.path.abspath(os.path.dirname(container_image))
    if not os.path.exists(image_dir):
        warnings.warn(
            f"Container image directory does not exist, creating: {image_dir}."
        )
        os.makedirs(image_dir, exist_ok=True)

    options = {
        "job-name": f"build-image-{experiment_id}",
        **cluster_config.build_options,
    }
    options.setdefault("wait", True)
    directives = _render_slurm_directives(options)
    log_target = options["output"]
    log_path_obj = None
    log_stop_event: Optional[threading.Event] = None
    log_future = None
    if log_target is not None:
        os.makedirs(os.path.dirname(log_target), exist_ok=True)
        log_path_obj = pathlib.Path(log_target)
        log_stop_event = threading.Event()

    image_cmd = " ".join(build_image_cmd(container_image, True))
    image_build_script = os.path.join(local_outdir, "container.sh")
    script_contents = f"""#!/bin/bash
{directives}
set -e
{image_cmd}
"""
    with open(image_build_script, "w") as f:
        f.write(script_contents)

    if log_path_obj is not None and log_stop_event is not None:
        log_path_obj.touch(exist_ok=True)
        log_future = thread_executor.submit(stream_log, log_target, 1, log_stop_event)

    try:
        subprocess.run(["sbatch", image_build_script], check=True)
    finally:
        if log_stop_event is not None and log_future is not None:
            log_stop_event.set()
            try:
                log_future.result(timeout=5)
            except Exception:
                pass


def submit_cluster_nextflow_job(
    cluster_config: ClusterConfig,
    experiment_id: str,
    local_outdir: str,
    outdir: str,
    config_path: str,
    workflow_path: str,
    report_path: str,
    workdir: str,
    resume: bool,
    hyperqueue: bool,
    filesystem: Optional["AbstractFileSystem"],
    thread_executor: Optional[ThreadPoolExecutor] = None,
) -> None:
    """
    Submit Nextflow workflow as SLURM batch job on HPC cluster.
    """
    batch_script = os.path.join(local_outdir, "nextflow_job.sh")
    nf_profile = "slurm_hq" if hyperqueue else "slurm"
    if hyperqueue:
        hyperqueue_init, hyperqueue_exit = hyperqueue_snippets(outdir)
    else:
        hyperqueue_init, hyperqueue_exit = "", ""
    trap_line = ""
    if hyperqueue_exit:
        trap_line = (
            "# Ensure HyperQueue shutdown on failure or interruption\n"
            f"trap 'exitcode=$?; {hyperqueue_exit}' EXIT"
        )

    options = {"job-name": f"nf-{experiment_id}", **cluster_config.nextflow_options}
    if cluster_config.jenkins:
        options.setdefault("wait", True)
    directives = _render_slurm_directives(options)
    log_target = options["output"]
    if cluster_config.jenkins and log_target is None:
        raise RuntimeError(
            "A concrete --output path is required for Jenkins mode so logs can be streamed."
        )
    log_path_obj = None
    if log_target is not None:
        os.makedirs(os.path.dirname(log_target), exist_ok=True)
        log_path_obj = pathlib.Path(log_target)

    resume_flag = "-resume" if resume else ""
    nf_setup_block = cluster_config.nf_setup or ""
    script_contents = f"""#!/bin/bash
{directives}
set -e
{trap_line}
{nf_setup_block}
{hyperqueue_init}
nextflow -C {config_path} run {workflow_path} -profile {nf_profile} \
    -with-report {report_path} -work-dir {workdir} {resume_flag}
"""

    with open(batch_script, "w") as f:
        f.write(script_contents)

    copy_to_filesystem(
        batch_script, os.path.join(outdir, "nextflow_job.sh"), filesystem
    )

    stream_stop_event: Optional[threading.Event] = None
    stream_future = None
    if cluster_config.jenkins and log_path_obj is not None:
        log_path_obj.touch(exist_ok=True)
        if thread_executor is None:
            raise RuntimeError("Thread executor required for Jenkins mode")
        stream_stop_event = threading.Event()
        stream_future = thread_executor.submit(
            stream_log, log_target, 1, stream_stop_event
        )

    try:
        subprocess.run(["sbatch", batch_script], check=True)
    finally:
        if stream_stop_event is not None and stream_future is not None:
            stream_stop_event.set()
            try:
                stream_future.result(timeout=5)
            except Exception:
                pass


def copy_to_filesystem(
    source: str, dest: str, filesystem: Optional["AbstractFileSystem"] = None
):
    """
    Robustly copy the contents of a local source file to a destination path.

    Args:
        source: Path to source file on local filesystem
        dest: Path to destination file on filesystem
        filesystem: LocalFileSystem or fsspec filesystem
    """
    if filesystem is None:
        # Simple local file copy
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(source, dest)
        return
    # fsspec implementation
    with filesystem.open(dest, mode="wb") as stream:
        with open(source, "rb") as f:
            stream.write(f.read())


def stream_log(
    output_log: str, sleep_time: int = 1, stop_event: Optional[threading.Event] = None
):
    """Periodically stream appended content from ``output_log`` to stdout."""
    log_path = pathlib.Path(output_log)
    # Track last position read in output log file
    last_position = 0
    while True:
        # Read any new content from the log file
        if log_path.exists():
            with open(output_log, "r") as f:
                # Move to where we left off
                f.seek(last_position)
                # Read and print new content
                new_content = f.read()
                if new_content:
                    print(new_content, end="", flush=True)
                # Remember where we are now
                last_position = f.tell()
        else:
            break
        if stop_event is not None and stop_event.is_set():
            break
        time.sleep(sleep_time)


def main():
    parser = argparse.ArgumentParser()
    config_file = os.path.join(CONFIG_DIR_PATH, "default.json")
    parser.add_argument(
        "--config",
        action="store",
        default=config_file,
        help=(
            "Path to configuration file for the simulation. "
            "All key-value pairs in this file will be applied on top "
            f"of the options defined in {config_file}."
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume workflow with given experiment ID. The experiment ID must "
        "match the supplied configuration file and if suffix_time was used, must "
        "contain the full time suffix (suffix_time will not be applied again).",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        default=False,
        help="Only build workflow files (main.nf, nextflow.config, workflow_config.json) "
        "without executing the workflow. Temp files are preserved for inspection.",
    )
    args = parser.parse_args()
    config = load_config_with_inheritance(config_file)
    user_config = load_config_with_inheritance(args.config)
    _merge_configs(config, user_config)

    # Multi-parca support is being re-implemented on top of master's content-
    # addressed parca_out / variant_info channels. The previous implementation
    # (channel of (parca_id, kb, offset) tuples + mergeVariantMetadata) does
    # not compose with the new caching API. Fail loudly so users with old
    # configs aren't silently downgraded to single-parca.
    # See: .claude/plans/dataset-sensitivity-exploration.md (Part 4).
    if config.get("parca_variants"):
        raise NotImplementedError(
            "`parca_variants` (multi-parca workflow) is temporarily disabled "
            "after the master merge introduced content-addressed parca caching. "
            "It will be re-implemented on top of the new tuple shape — see the "
            "follow-up task tracked in the merge commit. For now, run a single "
            "parca per workflow invocation, or sequence multiple workflow runs."
        )

    experiment_id = config["experiment_id"]
    if experiment_id is None:
        raise RuntimeError("No experiment ID was provided.")
    if args.resume is not None:
        experiment_id = args.resume
        config["experiment_id"] = args.resume
    elif config["suffix_time"]:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_id = experiment_id + "_" + current_time
        config["experiment_id"] = experiment_id
    config["suffix_time"] = False
    # Special characters are messy so do not allow them
    if experiment_id != parse.quote_plus(experiment_id):
        raise TypeError(
            "Experiment ID cannot contain special characters"
            f"that change the string when URL quoted: {experiment_id}"
            f" != {parse.quote_plus(experiment_id)}"
        )
    # Resolve output directory
    out_bucket = ""
    if "out_uri" not in config["emitter_arg"]:
        out_uri = os.path.abspath(config["emitter_arg"]["out_dir"])
        config["emitter_arg"]["out_dir"] = out_uri
        assert parse.urlparse(out_uri).scheme == "", (
            "Output directory must be a local path, not a URI. "
            "Specify URIs using 'out_uri' under 'emitter_arg'."
        )
    else:
        out_uri = config["emitter_arg"]["out_uri"]
        parsed_uri = parse.urlparse(out_uri)
        if parsed_uri.scheme not in ("local", "file") and not FSSPEC_AVAILABLE:
            raise RuntimeError(
                f"URI '{out_uri}' specified but fsspec is not available. "
                "Install fsspec or provide a local URI/out directory."
            )
        out_bucket = parsed_uri.netloc
    # Resolve sim_data_path if provided
    if config["sim_data_path"] is not None:
        config["sim_data_path"] = os.path.abspath(config["sim_data_path"])
    filesystem, outdir = parse_uri(out_uri)
    outdir = os.path.join(outdir, experiment_id, "nextflow")
    exp_outdir = os.path.dirname(outdir)
    out_uri = os.path.join(out_uri, experiment_id, "nextflow")
    cluster_config = get_cluster_config(config, outdir, experiment_id)

    # Use random seed for Jenkins CI runs
    if cluster_config and cluster_config.jenkins:
        config["lineage_seed"] = random.randint(0, 2**31 - 1)

    repo_dir = os.path.dirname(os.path.dirname(__file__))
    local_outdir = os.path.join(repo_dir, "nextflow_temp", experiment_id)
    os.makedirs(local_outdir, exist_ok=True)
    if filesystem is None:
        if os.path.exists(exp_outdir) and not args.resume:
            raise RuntimeError(
                f"Output directory already exists: {exp_outdir}. "
                "Please use a different experiment ID or output directory. "
                "Alternatively, move, delete, or rename the existing directory."
            )
        os.makedirs(outdir, exist_ok=True)
    else:
        if filesystem.exists(exp_outdir) and not args.resume:
            raise RuntimeError(
                f"Output directory already exists: {exp_outdir}. "
                "Please use a different experiment ID or output directory. "
                "Alternatively, move, delete, or rename the existing directory."
            )
        filesystem.makedirs(outdir, exist_ok=True)

    # Save full config for human reference
    temp_config_path = f"{local_outdir}/workflow_config.json"
    final_config_path = os.path.join(outdir, "workflow_config.json")
    with open(temp_config_path, "w") as f:
        json.dump(config, f)
    if args.resume is None:
        copy_to_filesystem(temp_config_path, final_config_path, filesystem)

    # Save stripped config (without resource-only keys) for Nextflow processes.
    # This ensures changing resource allocations doesn't invalidate the cache.
    stripped_config = strip_resource_keys(config)
    temp_stripped_path = f"{local_outdir}/workflow_config_stripped.json"
    final_stripped_path = os.path.join(outdir, "workflow_config_stripped.json")
    final_stripped_uri = os.path.join(out_uri, "workflow_config_stripped.json")
    with open(temp_stripped_path, "w") as f:
        json.dump(stripped_config, f)
    if args.resume is None:
        copy_to_filesystem(temp_stripped_path, final_stripped_path, filesystem)

    nf_config = os.path.join(os.path.dirname(__file__), "nextflow", "config.template")
    with open(nf_config, "r") as f:
        nf_config = f.readlines()
    nf_config = "".join(nf_config)
    nf_overrides = {
        "EXPERIMENT_ID": experiment_id,
        "CONFIG_FILE": final_stripped_uri,
        "BUCKET": out_bucket,
        "PUBLISH_DIR": os.path.dirname(os.path.dirname(out_uri)),
        "PARCA_CPUS": config["parca_options"]["cpus"],
        "PARCA_MEM": config["parca_options"]["memory_gb"],
        "PARCA_TIME": config["parca_options"]["slurm_time_hrs"],
        "ANALYSIS_CPUS": config["analysis_options"]["cpus"],
        "ANALYSIS_MEM": config["analysis_options"]["memory_gb"],
        "ANALYSIS_TIME": config["analysis_options"]["slurm_time_hrs"],
        "DUCKDB_THREADS": config["analysis_options"].get(
            "duckdb_threads", config["analysis_options"]["cpus"]
        ),
        "HQ_CORES": config["HQ_CORES"],
        "SIM_TIME": config["SIM_TIME"],
        "SIM_MEM": config["SIM_MEM"],
        "SIM_CPUS": config["SIM_CPUS"],
        "HYPERQUEUE": config["HYPERQUEUE"],
    }
    nf_config = _apply_nf_config_overrides(nf_config, nf_overrides)

    # By default, assume running on local device
    nf_profile = "standard"
    thread_executor = None

    # If not running on a local device, build container images according
    # to options under aws, gcloud, or cluster configuration keys
    aws_config_dict = config.get("aws", None)
    gcloud_config = config.get("gcloud", None)

    if aws_config_dict is not None:
        nf_profile = "aws"
        container_image = aws_config_dict.get("container_image", None)
        if container_image is None:
            raise RuntimeError("Must supply name for container image.")
        aws_region = aws_config_dict.get("region", "us-gov-west-1")
        full_image_uri = run_ecr_script(
            container_image,
            build=aws_config_dict.get("build_image", False),
            region=aws_region,
        )
        nf_config = nf_config.replace("IMAGE_NAME", full_image_uri)
        nf_config = nf_config.replace(
            "QUEUE", aws_config_dict.get("batch_queue", "vecoli")
        )
        nf_config = nf_config.replace("AWS_REGION", aws_region)
    elif gcloud_config is not None:
        nf_profile = "gcloud"
        project_id = subprocess.run(
            ["gcloud", "config", "get", "project"], stdout=subprocess.PIPE, text=True
        ).stdout.strip()
        region = subprocess.run(
            ["gcloud", "config", "get", "compute/region"],
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        image_prefix = f"{region}-docker.pkg.dev/{project_id}/vecoli/"
        container_image = gcloud_config.get("container_image", None)
        if container_image is None:
            raise RuntimeError("Must supply name for container image.")
        if gcloud_config.get("build_image", False):
            full_gcloud_uri = image_prefix + container_image
            if _gcloud_image_exists(full_gcloud_uri) and not _confirm_overwrite(
                full_gcloud_uri
            ):
                raise SystemExit(
                    "Aborted: will not supersede existing Artifact Registry image."
                )
            image_cmd = build_image_cmd(container_image)
            subprocess.run(image_cmd, check=True)
        nf_config = nf_config.replace("IMAGE_NAME", image_prefix + container_image)
    elif cluster_config is not None:
        # Start a new thread to forward output of submitted jobs to stdout
        thread_executor = ThreadPoolExecutor(max_workers=1)

        # Build container image if requested
        build_cluster_container_image(
            cluster_config, experiment_id, local_outdir, thread_executor
        )

        image_name = cluster_config.require_container_image()
        nf_config = nf_config.replace("IMAGE_NAME", image_name)
        nf_config = cluster_config.apply_config_overrides(nf_config)
    local_config = os.path.join(local_outdir, "nextflow.config")
    with open(local_config, "w") as f:
        f.writelines(nf_config)

    run_parca, sim_imports, sim_workflow = generate_code(config)

    nf_template_path = os.path.join(
        os.path.dirname(__file__), "nextflow", "template.nf"
    )
    with open(nf_template_path, "r") as f:
        nf_template = f.readlines()
    nf_template = "".join(nf_template)
    nf_template = nf_template.replace("RUN_PARCA", run_parca)
    nf_template = nf_template.replace("IMPORTS", sim_imports)
    nf_template = nf_template.replace("WORKFLOW", sim_workflow)
    local_workflow = os.path.join(local_outdir, "main.nf")
    with open(local_workflow, "w") as f:
        f.writelines(nf_template)

    copy_to_filesystem(local_workflow, os.path.join(outdir, "main.nf"), filesystem)
    copy_to_filesystem(
        local_config, os.path.join(outdir, "nextflow.config"), filesystem
    )

    # If build-only mode, skip execution and preserve temp files
    if args.build_only:
        print(
            f"Build-only mode: files generated in {local_outdir} and copied to {out_uri}"
        )
        print("  - main.nf")
        print("  - nextflow.config")
        print("  - workflow_config.json")
        return local_outdir

    # Start nextflow workflow
    report_path = os.path.join(
        out_uri,
        f"{experiment_id}_report.html",
    )
    if filesystem is None:
        if os.path.exists(report_path):
            raise RuntimeError(
                f"Report file already exists: {report_path}. "
                "Please move, delete, or rename it, then run with --resume again."
            )
    else:
        if filesystem.exists(report_path):
            raise RuntimeError(
                f"Report file already exists: {report_path}. "
                "Please move, delete, or rename it, then run with --resume again."
            )
    workdir = os.path.join(out_uri, "nextflow_workdirs")
    try:
        if cluster_config is None:
            resume_flag = ["-resume"] if args.resume is not None else []
            subprocess.run(
                [
                    "nextflow",
                    "-C",
                    local_config,
                    "run",
                    local_workflow,
                    "-profile",
                    nf_profile,
                    "-with-report",
                    report_path,
                    "-work-dir",
                    workdir,
                    *resume_flag,
                ],
                check=True,
            )
        elif cluster_config is not None:
            submit_cluster_nextflow_job(
                cluster_config=cluster_config,
                experiment_id=experiment_id,
                local_outdir=local_outdir,
                outdir=outdir,
                config_path=os.path.join(out_uri, "nextflow.config"),
                workflow_path=os.path.join(out_uri, "main.nf"),
                report_path=report_path,
                workdir=workdir,
                resume=args.resume is not None,
                hyperqueue=config["HYPERQUEUE"],
                filesystem=filesystem,
                thread_executor=thread_executor,
            )
    finally:
        shutil.rmtree(local_outdir)


if __name__ == "__main__":
    main()
