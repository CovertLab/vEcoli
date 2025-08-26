import argparse
import json
import os
import pathlib
import random
import shutil
import subprocess
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from urllib import parse
from typing import Optional

# Try to import fsspec, but make it optional
try:
    from fsspec import url_to_fs
    from fsspec.spec import AbstractFileSystem

    FSSPEC_AVAILABLE = True
except ImportError:
    FSSPEC_AVAILABLE = False


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

CONFIG_DIR_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
)
NEXTFLOW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nextflow")

MULTIDAUGHTER_CHANNEL = """
    generationSize = {gen_size}
    simCh
        .map {{ tuple(groupKey(it[1..4], generationSize[it[4]]), it[0], it[1], it[2], it[3], it[4] ) }}
        .groupTuple(remainder: true)
        .map {{ tuple(it[1][0], it[2][0], it[3][0], it[4][0], it[5][0]) }}
        .set {{ multiDaughterCh }}
"""
MULTIGENERATION_CHANNEL = """
    simCh
        .groupTuple(by: [1, 2, 3], size: {size}, remainder: true)
        .map {{ tuple(it[0][0], it[1], it[2], it[3]) }}
        .set {{ multiGenerationCh }}
"""
MULTISEED_CHANNEL = """
    simCh
        .groupTuple(by: [1, 2], size: {size}, remainder: true)
        .map {{ tuple(it[0][0], it[1], it[2]) }}
        .set {{ multiSeedCh }}
"""
MULTIVARIANT_CHANNEL = """
    // Group once to deduplicate variant names and pickles
    // Group again into single value for entire experiment
    simCh
        .groupTuple(by: [1, 2], size: {size}, remainder: true)
        .map {{ tuple(it[0][0], it[1], it[2]) }}
        .groupTuple(by: [1])
        .set {{ multiVariantCh }}
"""


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
    sim_workflow = [f"\tchannel.of( {seed}..<{seed + n_init_sims} ).set {{ seedCh }}"]

    all_sim_tasks = []
    for gen in range(generations):
        name = f"sim_gen_{gen + 1}"
        # Handle special case of 1st generation
        if gen == 0:
            sim_imports.append(
                f"include {{ simGen0 as {name} }} from '{NEXTFLOW_DIR}/sim'"
            )
            sim_workflow.append(
                (
                    f"\t{name}(params.config, variantCh.combine(seedCh).combine([1]), '0')"
                )
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

    if analysis_config.get("multivariant", False):
        # Channel that groups all sim tasks
        sim_workflow.append(
            MULTIVARIANT_CHANNEL.format(size=sims_per_seed * n_init_sims)
        )
        sim_workflow.append(
            "\tanalysisMultiVariant(params.config, kb, multiVariantCh, "
            "variantMetadataCh)"
        )
        sim_imports.append(
            f"include {{ analysisMultiVariant }} from '{NEXTFLOW_DIR}/analysis'"
        )

    if analysis_config.get("multiseed", False):
        # Channel that groups sim tasks by variant sim_data
        sim_workflow.append(MULTISEED_CHANNEL.format(size=sims_per_seed * n_init_sims))
        sim_workflow.append(
            "\tanalysisMultiSeed(params.config, kb, multiSeedCh, variantMetadataCh)"
        )
        sim_imports.append(
            f"include {{ analysisMultiSeed }} from '{NEXTFLOW_DIR}/analysis'"
        )

    if analysis_config.get("multigeneration", False):
        # Channel that groups sim tasks by variant sim_data and initial seed
        sim_workflow.append(MULTIGENERATION_CHANNEL.format(size=sims_per_seed))
        sim_workflow.append(
            "\tanalysisMultiGeneration(params.config, kb, multiGenerationCh, "
            "variantMetadataCh)"
        )
        sim_imports.append(
            f"include {{ analysisMultiGeneration }} from '{NEXTFLOW_DIR}/analysis'"
        )

    if analysis_config.get("multidaughter", False) and not single_daughters:
        # Channel that groups sim tasks by variant sim_data, initial seed, and generation
        # When simulating both daughters, will have >1 cell for generation >1
        gen_size = (
            "[" + ", ".join([f"{g + 1}: {2**g}" for g in range(generations)]) + "]"
        )
        sim_workflow.append(MULTIDAUGHTER_CHANNEL.format(gen_size=gen_size))
        sim_workflow.append(
            "\tanalysisMultiDaughter(params.config, kb, multiDaughterCh, "
            "variantMetadataCh)"
        )
        sim_imports.append(
            f"include {{ analysisMultiDaughter }} from '{NEXTFLOW_DIR}/analysis'"
        )

    if analysis_config.get("single", False):
        sim_workflow.append(
            "\tanalysisSingle(params.config, kb, simCh, variantMetadataCh)"
        )
        sim_imports.append(
            f"include {{ analysisSingle }} from '{NEXTFLOW_DIR}/analysis'"
        )

    if analysis_config.get("parca", False):
        sim_workflow.append("\tanalysisParca(params.config, kb)")

    return sim_imports, sim_workflow


def generate_code(config):
    sim_data_path = config.get("sim_data_path")
    if sim_data_path is not None:
        kb_dir = os.path.dirname(sim_data_path)
        run_parca = [
            f"\tfile('{kb_dir}').copyTo(\"${{params.publishDir}}/${{params.experimentId}}/parca/kb\")",
            f"\tChannel.fromPath('{kb_dir}').toList().set {{ kb }}",
        ]
    else:
        run_parca = ["\trunParca(params.config)", "\trunParca.out.toList().set {kb}"]
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
        )
    else:
        sim_imports, sim_workflow = generate_colony(seed)
    return "\n".join(run_parca), "\n".join(sim_imports), "\n".join(sim_workflow)


def build_image_cmd(image_name, apptainer=False) -> list[str]:
    build_script = os.path.join(
        os.path.dirname(__file__), "container", "build-image.sh"
    )
    cmd = [build_script, "-i", image_name]
    if apptainer:
        cmd.append("-a")
    return cmd


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


def stream_log(output_log: str, sleep_time: int = 1):
    """
    As long as output_log exists, read and print new content to stdout.
    """
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
    args = parser.parse_args()
    with open(config_file, "r") as f:
        config = json.load(f)
    if args.config is not None:
        config_file = args.config
        with open(args.config, "r") as f:
            user_config = json.load(f)
            for key in LIST_KEYS_TO_MERGE:
                user_config.setdefault(key, [])
                user_config[key].extend(config.get(key, []))
                if key == "engine_process_reports":
                    user_config[key] = [tuple(path) for path in user_config[key]]
                # Ensures there are no duplicates in d2
                user_config[key] = list(set(user_config[key]))
                user_config[key].sort()
            merge_dicts(config, user_config)

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
    # Use random seed for Jenkins CI runs
    if config.get("sherlock", {}).get("jenkins", False):
        config["lineage_seed"] = random.randint(0, 2**31 - 1)
    filesystem, outdir = parse_uri(out_uri)
    outdir = os.path.join(outdir, experiment_id, "nextflow")
    exp_outdir = os.path.dirname(outdir)
    out_uri = os.path.join(out_uri, experiment_id, "nextflow")
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
    temp_config_path = f"{local_outdir}/workflow_config.json"
    final_config_path = os.path.join(outdir, "workflow_config.json")
    final_config_uri = os.path.join(out_uri, "workflow_config.json")
    with open(temp_config_path, "w") as f:
        json.dump(config, f)
    if args.resume is None:
        copy_to_filesystem(temp_config_path, final_config_path, filesystem)

    nf_config = os.path.join(os.path.dirname(__file__), "nextflow", "config.template")
    with open(nf_config, "r") as f:
        nf_config = f.readlines()
    nf_config = "".join(nf_config)
    nf_config = nf_config.replace("EXPERIMENT_ID", experiment_id)
    nf_config = nf_config.replace("CONFIG_FILE", final_config_uri)
    nf_config = nf_config.replace("BUCKET", out_bucket)
    nf_config = nf_config.replace(
        "PUBLISH_DIR", os.path.dirname(os.path.dirname(out_uri))
    )
    nf_config = nf_config.replace("PARCA_CPUS", str(config["parca_options"]["cpus"]))

    # By default, assume running on local device
    nf_profile = "standard"
    # If not running on a local device, build container images according
    # to options under gcloud or sherlock configuration keys
    cloud_config = config.get("gcloud", None)
    if cloud_config is not None:
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
        container_image = cloud_config.get("container_image", None)
        if container_image is None:
            raise RuntimeError("Must supply name for container image.")
        if cloud_config.get("build_image", False):
            image_cmd = build_image_cmd(container_image)
            subprocess.run(image_cmd, check=True)
        nf_config = nf_config.replace("IMAGE_NAME", image_prefix + container_image)
    sherlock_config = config.get("sherlock", None)
    if sherlock_config is not None:
        if nf_profile == "gcloud":
            raise RuntimeError(
                "Cannot set both Sherlock and Google Cloud options in the input JSON."
            )
        nf_profile = "sherlock"
        # Suggest that users turn off background thread for Parquet emitter
        if config["emitter"] == "parquet" and config["emitter_arg"].get(
            "threaded", True
        ):
            warnings.warn(
                "Using a background thread in the Parquet emitter may degrade "
                "performance on Sherlock, where each simulation is allocated a "
                "single CPU core. Consider setting 'threaded' to False "
                "under 'emitter_arg'."
            )
        # Start a new thread to forward output of submitted jobs to stdout
        thread_executor = ThreadPoolExecutor(max_workers=1)
        container_image = sherlock_config.get("container_image", None)
        if container_image is None:
            raise RuntimeError("Must supply name for container image.")
        image_dir = os.path.abspath(os.path.dirname(container_image))
        if not os.path.exists(image_dir):
            warnings.warn(
                f"Container image directory does not exist, creating: {image_dir}."
            )
            os.makedirs(image_dir, exist_ok=True)
        if sherlock_config.get("build_image", False):
            image_cmd = " ".join(build_image_cmd(container_image, True))
            image_build_script = os.path.join(local_outdir, "container.sh")
            log_file = os.path.join(local_outdir, "build-image.out")
            with open(image_build_script, "w") as f:
                f.write(f"""#!/bin/bash
#SBATCH --job-name="build-image-{experiment_id}"
#SBATCH --time=30:00
#SBATCH --cpus-per-task 2
#SBATCH --mem=8GB
#SBATCH --partition=owners,normal
#SBATCH --wait
#SBATCH --output={log_file}
{image_cmd}
""")
            # Create empty log file for thread to stream from
            log_path = pathlib.Path(log_file)
            log_path.touch(exist_ok=True)
            thread_executor.submit(stream_log, log_file)
            try:
                subprocess.run(["sbatch", image_build_script], check=True)
            finally:
                # Always delete log file to stop streaming thread
                log_path.unlink()
        nf_config = nf_config.replace("IMAGE_NAME", container_image)
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

    workflow_path = os.path.join(out_uri, "main.nf")

    config_path = os.path.join(out_uri, "nextflow.config")
    if args.resume is None:
        copy_to_filesystem(local_workflow, os.path.join(outdir, "main.nf"), filesystem)
        copy_to_filesystem(
            local_config, os.path.join(outdir, "nextflow.config"), filesystem
        )

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
    if nf_profile == "standard" or nf_profile == "gcloud":
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
                "-resume" if args.resume is not None else "",
            ],
            check=True,
        )
    elif nf_profile == "sherlock":
        batch_script = os.path.join(local_outdir, "nextflow_job.sh")
        hyperqueue_init = ""
        hyperqueue_exit = ""
        if sherlock_config.get("hyperqueue", False):
            nf_profile = "sherlock_hq"
            hyperqueue_init = f"""
# Set the directory which HyperQueue will use 
export HQ_SERVER_DIR={os.path.join(outdir, ".hq-server")}
mkdir -p ${{HQ_SERVER_DIR}}

# Start the server in the background (&) and wait until it has started
hq server start --journal {os.path.join(outdir, ".hq-server/journal")} &
until hq job list &>/dev/null ; do sleep 1 ; done

"""
            hyperqueue_exit = "hq job wait all; hq worker stop all; hq server stop"
        nf_slurm_output = os.path.join(outdir, f"{experiment_id}_slurm.out")
        with open(batch_script, "w") as f:
            f.write(f"""#!/bin/bash
#SBATCH --job-name="nf-{experiment_id}"
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task 1
#SBATCH --mem=4GB
#SBATCH --partition=mcovert
#SBATCH --output={nf_slurm_output}
{"#SBATCH --wait" if sherlock_config.get("jenkins", False) else ""}
set -e
# Ensure HyperQueue shutdown on failure or interruption
trap 'exitcode=$?; {hyperqueue_exit}' EXIT
{hyperqueue_init}
nextflow -C {config_path} run {workflow_path} -profile {nf_profile} \
    -with-report {report_path} -work-dir {workdir} {"-resume" if args.resume is not None else ""}
""")
        copy_to_filesystem(
            batch_script, os.path.join(outdir, "nextflow_job.sh"), filesystem
        )
        # Make stdout of workflow viewable in Jenkins
        if sherlock_config.get("jenkins", False):
            # Create empty log file for thread to stream from
            log_path = pathlib.Path(nf_slurm_output)
            log_path.touch(exist_ok=True)
            thread_executor.submit(stream_log, nf_slurm_output)
            try:
                subprocess.run(["sbatch", batch_script], check=True)
            finally:
                # Always delete log file to stop streaming thread
                log_path.unlink()
        else:
            subprocess.run(["sbatch", batch_script], check=True)
    shutil.rmtree(local_outdir)


if __name__ == "__main__":
    main()
