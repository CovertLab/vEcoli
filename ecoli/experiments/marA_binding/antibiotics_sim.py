import os
import json
import numpy as np
from matplotlib import pyplot as plt

from ecoli.composites.ecoli_engine_process import run_simulation
from ecoli.experiments.ecoli_master_sim import CONFIG_DIR_PATH, SimConfig
from ecoli.experiments.marA_binding.marA_master import ids_of_interest

def run_sim(tet_conc=0, accumulation=True):
    config = SimConfig()
    config.update_from_json(os.path.join(
        CONFIG_DIR_PATH, "antibiotics_tetracycline_cephaloridine.json"))
    tetracycline_gradient = {
        'total_time': 100,
        'emitter': 'database',
        'mar_regulon': True,
        'initial_state_file': f'wcecoli_marA_{int(tet_conc*1000)}',
        'spatial_environment_config': {
            'reaction_diffusion': {
                'gradient': {
                    'molecules': {
                        'tetracycline': tet_conc   
                    }
                }
            },
            'field_timeline': {
                'timeline': [
                    [1000, {
                        "tetracycline": 0
                    }]
                ]
            }
        },
        'engine_process_reports': [
            ['bulk', 'tetracycline'],
            ['bulk', 'marR-tet[c]'],
            ['bulk', 'CPLX0-7710[c]'],
            ['unique', 'RNA']
        ]
    }
    marA_regulated = [monomer['variable'] for monomer in ids_of_interest()]
    tetracycline_gradient['engine_process_reports'] += marA_regulated
    if not os.path.exists(f'data/wcecoli_marA_{int(tet_conc*1000)}.json'):
        with open('data/wcecoli_t0.json') as f:
            initial_state = json.load(f)
        # Add bulk tetracycline and marR-tet complex
        initial_state['bulk']['tetracycline[c]'] = 0
        initial_state['bulk']['marR-tet[c]'] = 0
        # Add promoter binding data for marA and marR
        for promoter_data in initial_state['unique']['promoter'].values():
            promoter_data['bound_TF'] += [False, False]
        with open(f'data/wcecoli_marA_{int(tet_conc*1000)}.json', 'w') as f:
            json.dump(initial_state, f)
    config.update_from_dict(tetracycline_gradient)
    if not accumulation:
        config._config['process_configs']['antibiotic-transport-steady-state'][
            'initial_reaction_parameters']['tetracycline']['diffusion'].pop(
                'accumulation_factor')
        config._config['topology']['antibiotic-transport-steady-state'][
            'tetracycline']['reaction_parameters']['diffusion'].pop(
                'accumulation_factor')
    engine = run_simulation(config)
    
    active = engine.emitter.saved_data[4.0]['agents']['0']['bulk']['CPLX0-7710[c]']
    inactive = engine.emitter.saved_data[4.0]['agents']['0']['bulk']['marR-tet[c]']
    print(tet_conc)
    if active+inactive > 0:
        print(inactive/(active+inactive))
    else:
        print(0)

def test_rates():
    run_sim(0.003375, True)
        
if __name__ == "__main__":
    test_rates()
    