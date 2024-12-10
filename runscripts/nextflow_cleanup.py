import json
import shutil
import os
import argparse



def run_default():

    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow_config")
    args = parser.parse_args()
    workflow_config = args.workflow_config

    config_path = os.path.join(os.getcwd(),'ecoli','composites','ecoli_configs',str(workflow_config))

    with open(os.path.join(config_path), 'r+') as f:
        config_json = json.load(f)

    experiment_id = config_json['experiment_id']

    nextflow_path = os.path.join(os.getcwd(),'out',str(experiment_id),'nextflow')

    shutil.rmtree(nextflow_path)

if __name__ == '__main__':
    run_default()