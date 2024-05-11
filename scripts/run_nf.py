import argparse
import json
import os
import subprocess
import warnings
from datetime import datetime

CONFIG_DIR_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
    'ecoli',
    'composites',
    'ecoli_configs',
)
NEXTFLOW_DIR = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)
    ),
    'nextflow'
)

SIM_TAG = "Seed{seed}Gen{gen}Cell{cell}"
SIM_GEN_0_INC = "include {{ simGen0 as sim{name} }} from '{nf_dir}/sim'"
SIM_INC = "include {{ sim as sim{name} }} from '{nf_dir}/sim'"
SIM_GEN_0_FLOW = ("\tsim{name}(params.config, "
    "variantCh.combine([{seed}]).combine([0]), {seed}, 0)")
SIM_FLOW = ("\tsim{name}(sim{parent}.out.config, sim{parent}.out.nextGen, "
    "sim{parent}.out.{daughter}, {seed}, {cell})")

ALL_SIM_CHANNEL = """
    {task_one}
        .mix({other_tasks})
        .set {{ simCh }}
"""
MULTICELL_CHANNEL = """
    simCh
        .groupTuple(by: [2, 3, 4], size: {size})
        .set {{ multiCellCh }}
"""
MULTIGENERATION_CHANNEL = """
    simCh
        .groupTuple(by: [2, 3], size: {size})
        .set {{ multiGenerationCh }}
"""
MULTISEED_CHANNEL = """
    simCh
        .groupTuple(by: 2, size: {size})
        .set {{ multiSeedCh }}
"""
MULTIVARIANT_CHANNEL = """
    simCh
        .groupTuple(by: 1)
        .set {{ multiVariantCh }}
"""


def generate_colony(seeds: int):
    """
    Create strings to import and compose Nextflow processes for colony sims.
    """
    return [], []


def generate_lineage(seed: int, n_init_sims: int, generations: int, 
    single_daughters: bool, analysis_config: dict[str, dict[str, dict]]):
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
    sim_workflow = []

    all_sim_tasks = []
    for seed in range(seed, seed + n_init_sims):
        for gen in range(generations):
            for cell in (range(2**gen) if not single_daughters else [0]):
                name = SIM_TAG.format(seed=seed, gen=gen, cell=cell)

                # Compile list of metadata outputs for all sim tasks
                all_sim_tasks.append(f'sim{name}.out.metadata')
                
                # Handle special case of 1st generation
                if gen == 0:
                    sim_imports.append(SIM_GEN_0_INC.format(name=name, nf_dir=NEXTFLOW_DIR))
                    sim_workflow.append(SIM_GEN_0_FLOW.format(name=name, seed=seed))
                    continue
                
                sim_imports.append(SIM_INC.format(name=name, nf_dir=NEXTFLOW_DIR))
                # Cell 0 has daughters 0 & 1 in next gen, 1 has 2 & 3, etc.
                parent = SIM_TAG.format(seed=seed, gen=gen-1, cell=cell//2)
                daughter = 'd1'
                if cell % 2 == 1:
                    daughter = 'd2'
                sim_workflow.append(SIM_FLOW.format(name=name, parent=parent,
                    daughter=daughter, seed=seed+gen+cell, cell=cell))

    # Channel that combines metadata for all sim tasks
    sim_workflow.append(ALL_SIM_CHANNEL.format(task_one=all_sim_tasks[0],
        other_tasks=', '.join(all_sim_tasks[1:])))

    if analysis_config.get('multivariant', False):
        # Channel that groups all sim tasks
        sim_workflow.append(MULTIVARIANT_CHANNEL)
        sim_workflow.append("\tanalysisMultivariant(params.config, kb, multiVariantCh)")
        sim_imports.append(f"include {{ analysisMultivariant }} from '{NEXTFLOW_DIR}/analysis'")
    
    if analysis_config.get('multiseed', False):
        # Channel that groups sim tasks by variant sim_data
        sim_workflow.append(MULTISEED_CHANNEL.format(
            size=int(len(all_sim_tasks) / n_init_sims)))
        sim_workflow.append("\tanalysisMultiseed(params.config, kb, multiSeedCh)")
        sim_imports.append(f"include {{ analysisMultiseed }} from '{NEXTFLOW_DIR}/analysis'")
    
    if analysis_config.get('multigeneration', False):
        # Channel that groups sim tasks by variant sim_data and initial seed
        num_daughters = 1 if single_daughters else 2
        sim_workflow.append(MULTIGENERATION_CHANNEL.format(
            size=generations * num_daughters))
        sim_workflow.append("\tanalysisMultigeneration(params.config, kb, multiGenerationCh)")
        sim_imports.append(f"include {{ analysisMultigeneration }} from '{NEXTFLOW_DIR}/analysis'")

    if analysis_config.get('multicell', False):
        # Channel that groups sim tasks by variant sim_data, initial seed, and generation
        size = 1 if single_daughters else 2
        sim_workflow.append(MULTICELL_CHANNEL.format(size=size))
        sim_workflow.append("\tanalysisMulticell(params.config, kb, multiCellCh)")
        sim_imports.append(f"include {{ analysisMulticell }} from '{NEXTFLOW_DIR}/analysis'")
    
    if analysis_config.get('single', False):
        sim_workflow.append("\tanalysisSingle(params.config, kb, simCh)")
        sim_imports.append(f"include {{ analysisSingle }} from '{NEXTFLOW_DIR}/analysis'")
    
    if analysis_config.get('parca', False):
        sim_workflow.append("\tanalysisParca(params.config, kb)")

    return sim_imports, sim_workflow


def generate_code(config):
    seed = config.get('seed', 0)
    generations = config.get('generations', 0)
    if generations:
        n_init_sims = config.get('n_init_sims')
        single_daughters = config.get('single_daughters', True)
        sim_imports, sim_workflow = generate_lineage(
            seed, n_init_sims, generations, single_daughters,
            config.get('analysis_options', {}))
    else:
        sim_imports, sim_workflow = generate_colony(seed, n_init_sims)
    return '\n'.join(sim_imports), '\n'.join(sim_workflow)


def build_runtime_image(image_name):
    build_script = os.path.join(os.path.dirname(__file__),
                                'container', 'build-runtime.sh')
    subprocess.run([build_script, '-r', image_name], check=True)


def build_wcm_image(image_name, runtime_image_name):
    build_script = os.path.join(os.path.dirname(__file__),
                                'container', 'build-wcm.sh')
    if runtime_image_name is None:
        warnings.warn('No runtime image name supplied. By default, '
                      'we build the model image from the runtime '
                      'image with name ' + os.environ['USER'] + '-wcm-code." '
                      'If this is correct, add this under "gcloud" > '
                      '"runtime_image_name" in your config JSON.')
    subprocess.run([build_script, '-w', image_name,
                    '-r', runtime_image_name], check=True)


def main():
    parser = argparse.ArgumentParser()
    config_file = os.path.join(CONFIG_DIR_PATH, 'default.json')
    parser.add_argument(
        '--config', action='store',
        default=config_file,
        help=(
            'Path to configuration file for the simulation. '
            'All key-value pairs in this file will be applied on top '
            f'of the options defined in {config_file}.'))
    args = parser.parse_args()
    with open(config_file, 'r') as f:
        config = json.load(f)
    if args.config is not None:
        config_file = args.config
        with open(os.path.join(CONFIG_DIR_PATH, args.config), 'r') as f:
            config = {**config, **json.load(f)}

    experiment_id = config['experiment_id']
    current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    if experiment_id is None:
        warnings.warn("No experiment ID was provided. Using "
                      f"current time: {current_time}")
        experiment_id = current_time
    elif config['suffix_time']:
        experiment_id = experiment_id + '_' + current_time
    
    nf_config = os.path.join(os.path.dirname(__file__),
                             'nextflow', 'config.template')
    with open(nf_config, 'r') as f:
        nf_config = f.readlines()
    nf_config = "".join(nf_config)
    nf_config = nf_config.replace("EXPERIMENT_ID", experiment_id)
    nf_config = nf_config.replace("CONFIG_FILE", config_file)

    cloud_config = config.get('gcloud', None)
    if cloud_config is not None:
        # Add logic for starting MongoDB VM
        runtime_image_name = cloud_config.get('runtime_image_name', None)
        if cloud_config.get('build_runtime_image', False):
            if runtime_image_name is None:
                raise RuntimeError('Must supply name for runtime image.')
            build_runtime_image(runtime_image_name)
        if cloud_config.get('build_wcm_image', False):
            wcm_image_name = cloud_config.get('wcm_image_name', None)
            if wcm_image_name is None:
                raise RuntimeError('Must supply name for WCM image.')
            build_wcm_image(wcm_image_name, runtime_image_name)
        nf_config = nf_config.replace("IMAGE_NAME", wcm_image_name)
    
    repo_dir = os.path.dirname(os.path.dirname(__file__))
    out_dir = os.path.join(repo_dir, 'out', experiment_id)
    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, 'nextflow.config')
    with open(config_path, 'w') as f:
        f.writelines(nf_config)

    sim_imports, sim_workflow = generate_code(config)

    nf_template_path = os.path.join(os.path.dirname(__file__), 'nextflow', 'template.nf')
    with open(nf_template_path, 'r') as f:
        nf_template = f.readlines()
    nf_template = ''.join(nf_template)
    nf_template = nf_template.replace("IMPORTS", sim_imports)
    nf_template = nf_template.replace("WORKFLOW", sim_workflow)
    workflow_path = os.path.join(out_dir, 'main.nf')
    with open(workflow_path, 'w') as f:
        f.writelines(nf_template)


if __name__ == '__main__':
    main()
