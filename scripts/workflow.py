import argparse
import json
import os
import subprocess
import warnings
from datetime import datetime

CONFIG_DIR_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ecoli",
    "composites",
    "ecoli_configs",
)
NEXTFLOW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nextflow")

MULTIDAUGHTER_CHANNEL = """
    generationSize = {gen_size}
    simCh
        .map {{ tuple(groupKey(it[1..4], generationSize[it[4]]), it[0], it[1], it[2], it[3], it[4] ) }}
        .groupTuple()
        .map {{ tuple(it[1][0], it[2][0], it[3][0], it[4][0], it[5][0]) }}
        .set {{ multiDaughterCh }}
"""
MULTIGENERATION_CHANNEL = """
    simCh
        .groupTuple(by: [1, 2, 3], size: {size})
        .map {{ tuple(it[0][0], it[1], it[2], it[3]) }}
        .set {{ multiGenerationCh }}
"""
MULTISEED_CHANNEL = """
    simCh
        .groupTuple(by: [1, 2], size: {size})
        .map {{ tuple(it[0][0], it[1], it[2]) }}
        .set {{ multiSeedCh }}
"""
MULTIVARIANT_CHANNEL = """
    // Group once to deduplicate variant names and pickles
    // Group again into single value for entire experiment
    simCh
        .groupTuple(by: [1, 2], size: {size})
        .map {{ tuple(it[0][0], it[1], it[2]) }}
        .groupTuple(by: [1])
        .set {{ multiVariantCh }}
"""


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
    sim_workflow = [f"\tchannel.of( {seed}..{seed + n_init_sims} ).set {{ seedCh }}"]

    all_sim_tasks = []
    for gen in range(generations):
        name = f"sim_gen_{gen + 1}"
        # Handle special case of 1st generation
        # Start with agent ID 1 to avoid leading zeros
        if gen == 0:
            sim_imports.append(
                f"include {{ simGen0 as {name} }} from '{NEXTFLOW_DIR}/sim'")
            sim_workflow.append((
                f"\t{name}(params.config, "
                "variantCh.combine(seedCh).combine([1]), '0')"
            ))
            all_sim_tasks.append(f"{name}.out.metadata")
            if not single_daughters:
                sim_workflow.append(f"\t{name}.out.nextGen0.mix({name}.out.nextGen1).set {{ {name}_nextGen }}")
            else:
                sim_workflow.append(f"\t{name}.out.nextGen0.set {{ {name}_nextGen }}")
            continue
        sim_imports.append(
            f"include {{ sim as {name} }} from '{NEXTFLOW_DIR}/sim'"
        )
        parent = f"sim_gen_{gen}"
        sim_workflow.append(f"\t{name}({parent}_nextGen)")
        if not single_daughters:
            sim_workflow.append(f"\t{name}.out.nextGen0.mix({name}.out.nextGen1).set {{ {name}_nextGen }}")
        else:
            sim_workflow.append(f"\t{name}.out.nextGen0.set {{ {name}_nextGen }}")
        all_sim_tasks.append(f"{name}.out.metadata")

    # Channel that combines metadata for all sim tasks
    tasks = all_sim_tasks[0]
    other_tasks = ", ".join(all_sim_tasks[1:])
    sim_workflow.append(f"\t{tasks}.mix({other_tasks}).set {{ simCh }}")

    sims_per_seed = generations if single_daughters else 2**generations - 1

    if analysis_config.get("multi_variant", False):
        # Channel that groups all sim tasks
        sim_workflow.append(MULTIVARIANT_CHANNEL.format(size=sims_per_seed * n_init_sims))
        sim_workflow.append("\tanalysisMultiVariant(params.config, kb, multiVariantCh)")
        sim_imports.append(
            f"include {{ analysisMultiVariant }} from '{NEXTFLOW_DIR}/analysis'"
        )

    if analysis_config.get("multi_seed", False):
        # Channel that groups sim tasks by variant sim_data
        sim_workflow.append(MULTISEED_CHANNEL.format(size=sims_per_seed * n_init_sims))
        sim_workflow.append("\tanalysisMultiSeed(params.config, kb, multiSeedCh)")
        sim_imports.append(
            f"include {{ analysisMultiSeed }} from '{NEXTFLOW_DIR}/analysis'"
        )

    if analysis_config.get("multi_generation", False):
        # Channel that groups sim tasks by variant sim_data and initial seed
        sim_workflow.append(MULTIGENERATION_CHANNEL.format(size=sims_per_seed))
        sim_workflow.append(
            "\tanalysisMultiGeneration(params.config, kb, multiGenerationCh)"
        )
        sim_imports.append(
            f"include {{ analysisMultiGeneration }} from '{NEXTFLOW_DIR}/analysis'"
        )

    if analysis_config.get("multi_daughter", False) and not single_daughters:
        # Channel that groups sim tasks by variant sim_data, initial seed, and generation
        # When simulating both daughters, will have >1 cell for generation >1
        gen_size = "[" + ", ".join([f"{g+1}: {2**g}" for g in range(generations)]) + "]"
        sim_workflow.append(MULTIDAUGHTER_CHANNEL.format(gen_size=gen_size))
        sim_workflow.append(
            "\tanalysisMultiDaughter(params.config, kb, multiDaughterCh)"
        )
        sim_imports.append(
            f"include {{ analysisMultiDaughter }} from '{NEXTFLOW_DIR}/analysis'"
        )

    if analysis_config.get("single", False):
        sim_workflow.append("\tanalysisSingle(params.config, kb, simCh)")
        sim_imports.append(
            f"include {{ analysisSingle }} from '{NEXTFLOW_DIR}/analysis'"
        )

    if analysis_config.get("parca", False):
        sim_workflow.append("\tanalysisParca(params.config, kb)")

    return sim_imports, sim_workflow


def generate_code(config):
    seed = config.get("seed", 0)
    generations = config.get("generations", 0)
    if generations:
        n_init_sims = config.get("n_init_sims")
        single_daughters = config.get("single_daughters", True)
        sim_imports, sim_workflow = generate_lineage(
            seed,
            n_init_sims,
            generations,
            single_daughters,
            config.get("analysis_options", {}),
        )
    else:
        sim_imports, sim_workflow = generate_colony(seed, n_init_sims)
    return "\n".join(sim_imports), "\n".join(sim_workflow)


def build_runtime_image(image_name):
    build_script = os.path.join(
        os.path.dirname(__file__), "container", "build-runtime.sh"
    )
    subprocess.run([build_script, "-r", image_name], check=True)


def build_wcm_image(image_name, runtime_image_name):
    build_script = os.path.join(os.path.dirname(__file__), "container", "build-wcm.sh")
    if runtime_image_name is None:
        warnings.warn(
            "No runtime image name supplied. By default, "
            "we build the model image from the runtime "
            "image with name " + os.environ["USER"] + '-wcm-code." '
            'If this is correct, add this under "gcloud" > '
            '"runtime_image_name" in your config JSON.'
        )
    subprocess.run(
        [build_script, "-w", image_name, "-r", runtime_image_name], check=True
    )


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
    args = parser.parse_args()
    with open(config_file, "r") as f:
        config = json.load(f)
    if args.config is not None:
        config_file = args.config
        with open(args.config, "r") as f:
            config = {**config, **json.load(f)}

    experiment_id = config["experiment_id"]
    if experiment_id is None:
        raise RuntimeError("No experiment ID was provided.")
    if config["suffix_time"]:
        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        experiment_id = experiment_id + "_" + current_time

    nf_config = os.path.join(os.path.dirname(__file__), "nextflow", "config.template")
    with open(nf_config, "r") as f:
        nf_config = f.readlines()
    nf_config = "".join(nf_config)
    nf_config = nf_config.replace("EXPERIMENT_ID", experiment_id)
    nf_config = nf_config.replace("CONFIG_FILE", config_file)

    # By default, assume running on local device
    nf_profile = "standard"
    cloud_config = config.get("gcloud", None)
    if cloud_config is not None:
        nf_profile = "gcloud"
        runtime_image_name = cloud_config.get("runtime_image_name", None)
        if cloud_config.get("build_runtime_image", False):
            if runtime_image_name is None:
                raise RuntimeError("Must supply name for runtime image.")
            build_runtime_image(runtime_image_name)
        if cloud_config.get("build_wcm_image", False):
            wcm_image_name = cloud_config.get("wcm_image_name", None)
            if wcm_image_name is None:
                raise RuntimeError("Must supply name for WCM image.")
            build_wcm_image(wcm_image_name, runtime_image_name)
        nf_config = nf_config.replace("IMAGE_NAME", wcm_image_name)
    elif config.get("sherlock", None) is not None:
        nf_profile = "sherlock"

    repo_dir = os.path.dirname(os.path.dirname(__file__))
    out_dir = os.path.join(repo_dir, "out", experiment_id)
    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, "nextflow.config")
    with open(config_path, "w") as f:
        f.writelines(nf_config)

    sim_imports, sim_workflow = generate_code(config)

    nf_template_path = os.path.join(
        os.path.dirname(__file__), "nextflow", "template.nf"
    )
    with open(nf_template_path, "r") as f:
        nf_template = f.readlines()
    nf_template = "".join(nf_template)
    nf_template = nf_template.replace("IMPORTS", sim_imports)
    nf_template = nf_template.replace("WORKFLOW", sim_workflow)
    workflow_path = os.path.join(out_dir, "main.nf")
    with open(workflow_path, "w") as f:
        f.writelines(nf_template)

    # Start nextflow workflow
    subprocess.run(
        [
            "nextflow",
            "-C",
            config_path,
            "run",
            workflow_path,
            "-profile",
            nf_profile,
            "-with-report",
            f"{experiment_id}_report.html",
        ]
    )


if __name__ == "__main__":
    main()
