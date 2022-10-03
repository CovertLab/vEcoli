import os
import json

from ecoli.composites.ecoli_engine_process import run_simulation
from ecoli.experiments.ecoli_master_sim import CONFIG_DIR_PATH, SimConfig
from migration.migration_utils import recursive_compare

def run_sim(amp_conc=0, baseline=False, seed=0, total_time=10000,
    cloud=False, initial_colony_file=None, start_time=0
    ):
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
        'seed': seed,
        'start_time': start_time
    }
    if initial_colony_file:
        config['initial_colony_file'] = f'tet_{initial_colony_file}'
    # if tet_conc > 0:
    #     make_tet_initial_state(initial_colony_file)
    if cloud:
        tetracycline_gradient['save'] = True
        if baseline:
            tetracycline_gradient['save_times'] = [11550, 23100, 27000]
            tetracycline_gradient['total_time'] = 27000
        else:
            tetracycline_gradient['save_times'] = [11550]
            tetracycline_gradient['total_time'] = 15540
        tetracycline_gradient['emitter_arg'] = [
            ["host", "10.138.0.75:27017"],
            ["emit_limit", 5000000]
        ]
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

def add_mar_tf(data):
    # Add initial count for marR-tet complex
    data['bulk']['marR-tet[c]'] = 0
    # Add promoter binding data for marA and marR
    for promoter_data in data['unique']['promoter'].values():
        promoter_data['bound_TF'] += [False, False]
    return data

def make_tet_initial_state(initial_colony_file):
    with open(f'data/{initial_colony_file}.json') as f:
        initial_state = json.load(f)
    for agent_id, agent_data in initial_state['agents'].items():
        initial_state['agents'][agent_id] = add_mar_tf(agent_data)
    if os.path.exists(f'data/tet_{initial_colony_file}.json'):
        with open(f'data/tet_{initial_colony_file}.json') as f:
            existing_initial_state = json.load(f)
        if recursive_compare(initial_state, existing_initial_state):
            return
        else:
            print(f'tet_{initial_colony_file}.json out of date, updating')
    with open(f'data/tet_{initial_colony_file}.json', 'w') as f:
        json.dump(initial_state, f)

def generate_data():
    # run_sim(0, seed = 0, baseline=True, cloud=True)
    run_sim(0.003375, seed = 0, cloud=True, 
        initial_colony_file='seed_0_colony_t11550', start_time=11550)
        
if __name__ == "__main__":
    generate_data()
