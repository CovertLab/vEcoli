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
        config['add_processes'].remove('antibiotic-transport-odeint')
        config['add_processes'].remove('ecoli-rna-interference')
        config['add_processes'].remove('tetracycline-ribosome-equilibrium')
        config['process_configs'].pop('ecoli-rna-interference')
        config['process_configs'].pop('antibiotic-transport-odeint')
        config['process_configs'].pop('tetracycline-ribosome-equilibrium')
        config['topology'].pop('antibiotic-transport-odeint')
        config['topology'].pop('tetracycline-ribosome-equilibrium')
        config['engine_process_reports'].remove(['bulk', 'marR-tet[c]'])
        config['engine_process_reports'].remove(['bulk', 'CPLX0-3953-tetracycline[c]'])
        config['flow']['ecoli-mass-listener'] = [('ecoli-metabolism',)]
        config['flow'].pop('ecoli-rna-interference')
        config['flow'].pop('tetracycline-ribosome-equilibrium')
        config['mar_regulon'] = False
        config['initial_state_file'] = 'wcecoli_t0'
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
    run_sim(0.003375, seed = 0, baseline=True, check_initial_state=False, total_time=15000)
    run_sim(0.003375, seed = 0, check_initial_state=False, total_time=15000)
        
if __name__ == "__main__":
    generate_data()
