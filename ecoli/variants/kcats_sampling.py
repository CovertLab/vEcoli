import numpy as np
from scipy.stats import truncnorm
import json
import os
from ecoli.library.sim_data import LoadSimData




def run_default():

    sim_data_path = 'reconstruction/sim_data/kb/simData.cPickle'
    sim_data_loaded = LoadSimData(sim_data_path)
    sim_data = sim_data_loaded.sim_data
    config_dir = 'ecoli/composites/ecoli_configs'
    config_out_dir = os.path.join(config_dir, 'kcats_sampling')
    os.makedirs(config_out_dir, exist_ok=True)
    kcats_rxns = sim_data.process.metabolism.kinetic_constraint_reactions
    kcats_default = sim_data.process.metabolism._kcats
    for kcat_idx in range(len(kcats_default)):
        kcat_current = kcats_default[kcat_idx][1]
        rxn_current = kcats_rxns[kcat_idx]
        a = 0
        b = np.inf
        kcat_loc = kcat_current
        kcat_scale = np.sqrt(kcat_current)
        a_transformed, b_transformed = (a - kcat_loc) / kcat_scale, (b - kcat_loc) / kcat_scale
        rv_kcat = truncnorm(a_transformed, b_transformed, loc=kcat_loc, scale=kcat_scale)
        kcat_samples = rv_kcat.rvs(size=100)

        with open(os.path.join(config_dir,'kcats_rxns.json'), 'r+') as f:
            config_template = json.load(f)

            config_template['experiment_id'] = f'kcats_rxns_{str(kcat_idx)}'
            config_template['variants']['kcats_rxns']['rxn_id']['value'] = [f'{str(rxn_current)}']
            config_template['variants']['kcats_rxns']['kcat_val']['value'] = list(kcat_samples)
            with open( os.path.join(config_out_dir,f'kcats_rxns_{kcat_idx}.json'),'w') as f_out:
                json.dump(config_template, f_out, indent=4)


if '__main__' == __name__:
    run_default()



#%%
