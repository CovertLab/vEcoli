import os
import json
import argparse

from vivarium.library.units import units

from ecoli.composites.ecoli_engine_process import run_simulation
from ecoli.experiments.ecoli_master_sim import CONFIG_DIR_PATH, SimConfig
from migration.migration_utils import recursive_compare
from ecoli.library.parameters import param_store

def run_sim(tet_conc=0, amp_conc=0, baseline=False, seed=0,
    cloud=False, initial_colony_file=None, start_time=0
    ):
    config = SimConfig()
    config.update_from_json(os.path.join(
        CONFIG_DIR_PATH, "antibiotics_ampicillin.json"))
    config.update_from_json(os.path.join(
        CONFIG_DIR_PATH, "antibiotics_tetracycline.json"))
    add_opts = {
        'spatial_environment_config': {
            'reaction_diffusion': {
                'gradient': {
                    'molecules': {
                        'tetracycline': tet_conc * units.mM,
                        'ampicillin[p]': amp_conc * units.mM  
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
        'start_time': start_time,
        'colony_save_prefix': 'tet_amp',
        'save': True
    }
    config.update_from_dict(add_opts)
    if initial_colony_file:
        config._config.pop('initial_state_file')
        config['initial_colony_file'] = initial_colony_file
    else:
        make_tet_initial_state('wcecoli_t0')
    if baseline:
        print(f'Running baseline sim (seed = {seed}).')
        config["colony_save_prefix"] = "glc_combined"
        config['save_times'] = [11550, 23100, 26000]
        # 26000 catches the start of the 9th round of division
        # Run for one timestep past that to catch inner sim emits at 26000
        config['total_time'] = 26002
        # Ensure that sim starts with correctly reduced murein counts
        config["initial_state_overrides"] = ["overrides/reduced_murein"]
    else:
        print(f"Seed: {seed}")
        print(f"Tetracycline concentration: {tet_conc}")
        print(f"Ampicillin concentration: {amp_conc}")
        config["colony_save_prefix"] = f"amp_{amp_conc}_tet_{tet_conc}"
        config['save_times'] = [11550]
        config['total_time'] = 14452
    if cloud:
        config['emitter_arg'] = [
            ["host", "10.138.0.75:27017"],
            ["emit_limit", 5000000]
        ]

    run_simulation(config)


def add_mar_tf(data):
    # Add initial count for marR-tet complex
    data['bulk']['marR-tet[c]'] = 0
    # Add promoter binding data for marA and marR
    for promoter_data in data['unique']['promoter'].values():
        promoter_data['bound_TF'] += [False, False]
    return data


def make_tet_initial_state(initial_file):
    with open(f'data/{initial_file}.json') as f:
        initial_state = json.load(f)
    # Modify each cell in colony individually
    if 'agents' in initial_state:
        for agent_id, agent_data in initial_state['agents'].items():
            initial_state['agents'][agent_id] = add_mar_tf(agent_data)
    else:
        initial_state = add_mar_tf(initial_state)
    if os.path.exists(f'data/tet_{initial_file}.json'):
        with open(f'data/tet_{initial_file}.json') as f:
            existing_initial_state = json.load(f)
        if recursive_compare(initial_state, existing_initial_state):
            return
        else:
            print(f'tet_{initial_file}.json out of date, updating')
    with open(f'data/tet_{initial_file}.json', 'w') as f:
        json.dump(initial_state, f)


def generate_data(seed, cloud, tet_conc, amp_conc,
                  initial_colony_file, baseline):
    if baseline:
        run_sim(
            tet_conc=0,
            amp_conc=0,
            seed=seed,
            cloud=cloud,
            start_time=0,
            baseline=baseline)
    else:
        run_sim(
            tet_conc=tet_conc,
            amp_conc=amp_conc,
            seed=seed,
            cloud=cloud,
            initial_colony_file=initial_colony_file,
            start_time=11550,
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
        "--tet_conc",
        default=0.003375,
        type=float,
        help="Starting external tetracycline concentration (mM)"
    )
    parser.add_argument(
        "-a",
        "--amp_conc",
        default=param_store.get(("ampicillin", "mic")),
        type=float,
        help="Starting external ampicillin concentration (mM)"
    )

    args = parser.parse_args()

    generate_data(
        args.seed,
        cloud=(not args.local),
        tet_conc=args.tet_conc,
        amp_conc=args.amp_conc,
        initial_colony_file=args.initial_file,
        baseline=args.baseline
    )


if __name__ == "__main__":
    import multiprocessing; multiprocessing.set_start_method("spawn")
    main()
