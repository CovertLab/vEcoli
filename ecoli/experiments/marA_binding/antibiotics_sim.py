import os
import json

from ecoli.composites.ecoli_engine_process import run_simulation
from ecoli.experiments.ecoli_master_sim import CONFIG_DIR_PATH, SimConfig

def run_sim(tet_conc=0, baseline=False, 
    seed=0, total_time=10000, check_initial_state=True):
    config = SimConfig()
    config.update_from_json(os.path.join(
        CONFIG_DIR_PATH, "antibiotics_tetracycline.json"))
    tetracycline_gradient = {
        'total_time': total_time,
        'spatial_environment_config': {
            'reaction_diffusion': {
                'gradient': {
                    'molecules': {
                        'tetracycline': tet_conc   
                    }
                }
            },
            "field_timeline": {
                "timeline": [
                    [100000, {
                        "tetracycline": 0
                    }],
                ]
            },
        },
        'seed': seed
    }
    if check_initial_state:
        test_initial_state()
    config.update_from_dict(tetracycline_gradient)
    if baseline:
        config._config['add_processes'].remove('antibiotic-transport-odeint')
        config._config['add_processes'].remove('ecoli-rna-interference')
        config._config['add_processes'].remove('tetracycline-ribosome-equilibrium')
        config._config['process_configs'].pop('ecoli-rna-interference')
        config._config['process_configs'].pop('antibiotic-transport-odeint')
        config._config['process_configs'].pop('tetracycline-ribosome-equilibrium')
        config._config['topology'].pop('antibiotic-transport-odeint')
        config._config['topology'].pop('tetracycline-ribosome-equilibrium')
        config._config['engine_process_reports'].remove(['bulk', 'marR-tet[c]'])
        config._config['engine_process_reports'].remove(['bulk', 'CPLX0-3953-tetracycline[c]'])
        config._config['engine_process_reports'].remove(['bioscrape_deltas',])
        config._config['flow'].pop('ecoli-polypeptide-initiation_requester')
        config._config['mar_regulon'] = False
        config._config['initial_state_file'] = 'wcecoli_t0'
        config._config['division_threshold'] = 668
    run_simulation(config)

# Running this is slow (including the import statement)    
def test_initial_state():
    from migration.migration_utils import recursive_compare
    with open('data/wcecoli_t0.json') as f:
        initial_state = json.load(f)
    # Add initial count for marR-tet complex
    initial_state['bulk']['marR-tet[c]'] = 0
    # Add promoter binding data for marA and marR
    for promoter_data in initial_state['unique']['promoter'].values():
        promoter_data['bound_TF'] += [False, False]
    if os.path.exists('data/wcecoli_tet.json'):
        with open('data/wcecoli_tet.json') as f:
            existing_initial_state = json.load(f)
        if recursive_compare(initial_state, existing_initial_state):
            return
        else:
            print('wcecoli_tet.json out of date, updating')
    with open('data/wcecoli_tet.json', 'w') as f:
        json.dump(initial_state, f)

def generate_data():
    run_sim(0.003375, seed = 0, baseline=True, check_initial_state=False)
    run_sim(0.003375, seed = 0, check_initial_state=False, total_time=20000)
        
if __name__ == "__main__":
    generate_data()
