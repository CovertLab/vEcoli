import os
import json
import argparse

from ecoli.composites.ecoli_engine_process import run_simulation
from ecoli.experiments.ecoli_master_sim import CONFIG_DIR_PATH, SimConfig
from migration.migration_utils import recursive_compare

def run_sim(tet_conc=0, baseline=False, seed=0, total_time=10000,
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
                # Set timeline with arbitrarily high time
                # so that it is not used
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
            tetracycline_gradient['save_times'] = [11550, 23100, 26000]
            # 26000 catches the start of the 9th round of division
            tetracycline_gradient['total_time'] = 26000
        else:
            tetracycline_gradient['save_times'] = [11550]
            tetracycline_gradient['total_time'] = 14450
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
        config["initial_state_overrides"] = ["overrides/reduced_murein"]
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


def generate_data(seed, cloud, conc, total_time, initial_colony_file, baseline):
    if baseline:
        print('Running baseline sim.')
        run_sim(
            0,
            seed=seed,
            cloud=cloud,
            start_time=0,
            total_time=27000,
            baseline=baseline)
    else:
        print(f"Tetracycline concentration: {conc}")
        run_sim(
            conc,
            seed=seed,
            cloud=cloud,
            initial_colony_file=initial_colony_file,
            start_time=11550,
            total_time=total_time,
        )


def main():
    parser = argparse.ArgumentParser("Run tetracycline simulations.")

    parser.add_argument(
        "-s", "--seed", default=0, type=int, help="Random seed for simulation."
    )
    parser.add_argument("-l", "--local", action="store_true")
    parser.add_argument("-b", "--baseline", action="store_true")
    parser.add_argument(
        "-i",
        "--initial_file",
        help="colony save state to run the simulation off of",
    )
    parser.add_argument(
        "-t",
        "--total_time",
        default=15450,
        type=int,
        help="total time to run the simulation",
    )
    parser.add_argument(
        "-c",
        "--concentration",
        default=0.003375,
        type=float,
        help="Starting tetracycline concentration"
    )

    args = parser.parse_args()

    print(f"""Running tetracycline simulation with simulation with
              seed = {args.seed}
              for {args.total_time} seconds.""")

    generate_data(
        args.seed,
        cloud=(not args.local),
        conc=args.concentration,
        total_time=args.total_time,
        initial_colony_file=args.initial_file,
        baseline=args.baseline
    )


if __name__ == "__main__":
    main()
