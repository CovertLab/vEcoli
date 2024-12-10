import json
import os
import argparse



def run_default():

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_idx")
    args = parser.parse_args()
    sample_idx = int(args.sample_idx)

    config_dir = 'ecoli/composites/ecoli_configs'


    with open(os.path.join(config_dir,'kcats_sampling_default.json'), 'r+') as f:
        config_template = json.load(f)

        config_template['experiment_id'] = f'kcats_sampling_{str(sample_idx)}'

        with open( os.path.join(config_dir,f'kcats_sampling_{str(sample_idx)}.json'),'w') as f_out:
            json.dump(config_template, f_out, indent=4)


if '__main__' == __name__:
    run_default()



#%%
