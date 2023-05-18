import ast
import os
import json
import argparse

from vivarium.library.units import units

from ecoli.composites.ecoli_engine_process import run_simulation
from ecoli.experiments.ecoli_master_sim import CONFIG_DIR_PATH, SimConfig
from ecoli.library.parameters import param_store

def run_sim(tet_conc=0, amp_conc=0, baseline=False, seed=0,
    cloud=False, initial_colony_file=None, initial_state_file=None,
    start_time=0, runtime=None,
    ):
    config = SimConfig()
    config.update_from_json(os.path.join(
        CONFIG_DIR_PATH, "antibiotics_ampicillin.json"))
    config.update_from_json(os.path.join(
        CONFIG_DIR_PATH, "antibiotics_tetracycline.json"))
    add_opts = {
        "spatial_environment_config": {
            "reaction_diffusion": {
                "gradient": {
                    "molecules": {
                        "tetracycline": tet_conc * units.mM,
                        "ampicillin[p]": amp_conc * units.mM  
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
        "seed": seed,
        "start_time": start_time,
        "colony_save_prefix": "tet_amp",
    }
    config.update_from_dict(add_opts)
    if initial_colony_file:
        config._config.pop("initial_state_file")
        config["initial_colony_file"] = initial_colony_file
    elif initial_state_file:
        config["initial_state_file"] = initial_state_file
    else:
        make_initial_state("wcecoli_t0", rnai_data=config[
            "process_configs"]["ecoli-rna-interference"])
    if baseline:
        print(f"Running baseline sim (seed = {seed}).")
        config["colony_save_prefix"] = "glc_combined"
        # 26000 allows 8th round of division to mostly complete
        # Run for one timestep past that to catch inner sim emits at 26000
        if not runtime:
            runtime = 26002
            config["save"] = True
            config["save_times"] = [11550, 23100]
        config["total_time"] = runtime
        # Ensure that sim starts with correctly reduced murein counts
        config["initial_state_overrides"] = ["overrides/reduced_murein"]
    else:
        print(f"Seed: {seed}")
        print(f"Tetracycline concentration: {tet_conc}")
        print(f"Ampicillin concentration: {amp_conc}")
        config["colony_save_prefix"] = f"amp_{amp_conc}_tet_{tet_conc}"
        if not runtime:
            runtime = 14452
            config["save"] = True
            config["save_times"] = [11550]
        config["total_time"] = runtime
    if cloud:
        config["emitter_arg"] = [
            ["host", "10.138.0.75:27017"],
            ["emit_limit", 5000000]
        ]

    run_simulation(config)


def update_agent(data, rnai_data=None):
    new_bulk = [
        ("marR-tet[c]", 0),
        ("tetracycline[p]", 0),
        ("tetracycline[c]", 0),
        ("CPLX0-3953-tetracycline[c]", 0),
        ("ampicillin[p]", 0),
        ("ampicillin_hydrolyzed[p]", 0)
    ]
    # Add RNA duplexes
    if rnai_data:
        new_bulk += [
            (str(duplex_id), 0)
            for duplex_id in rnai_data["duplex_ids"]
        ]
    data["bulk"].extend(new_bulk)
    # Add promoter binding data for marA and marR
    for promoter_data in data["unique"]["promoter"]:
        # Bound TF boolean mask should be 4th attr
        promoter_data[3] += [False, False]
    # Bound TF boolean mask now has 26 TFs
    data["unique_dtypes"]["promoter"] = ast.literal_eval(
        data["unique_dtypes"]["promoter"])
    data["unique_dtypes"]["promoter"][3] = ('bound_TF', '?', (26,))
    data["unique_dtypes"]["promoter"] = str(data["unique_dtypes"]["promoter"])
    return data


def make_initial_state(initial_file, rnai_data=None):
    with open(f"data/{initial_file}.json") as f:
        initial_state = json.load(f)
    # Modify each cell in colony individually
    if "agents" in initial_state:
        for agent_id, agent_data in initial_state["agents"].items():
            initial_state["agents"][agent_id] = update_agent(
                agent_data, rnai_data)
    else:
        initial_state = update_agent(initial_state, rnai_data)
    with open(f"data/antibiotics_{initial_file}.json", "w") as f:
        json.dump(initial_state, f)


def generate_data(seed, cloud, tet_conc, amp_conc,
    initial_colony_file, initial_state_file, baseline,
    start_time, runtime):
    if baseline:
        run_sim(
            tet_conc=0,
            amp_conc=0,
            seed=seed,
            cloud=cloud,
            start_time=0,
            initial_state_file=initial_state_file,
            baseline=baseline,
            runtime=runtime)
    else:
        run_sim(
            tet_conc=tet_conc,
            amp_conc=amp_conc,
            seed=seed,
            cloud=cloud,
            initial_colony_file=initial_colony_file,
            initial_state_file=initial_state_file,
            start_time=start_time,
            runtime=runtime
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
        "--initial_colony_file",
        help="Colony save state to run the simulation off of",
    )
    parser.add_argument(
        "-f",
        "--initial_state_file",
        help="Single cell initial state (e.g. from wcEcoli)",
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
    parser.add_argument(
        "-r",
        "--runtime",
        default=None,
        type=int,
        help="Custom simulation run time."
    )
    parser.add_argument(
        "-n",
        "--start_time",
        default=None,
        type=int,
        help="Custom simulation start time."
    )

    args = parser.parse_args()

    generate_data(
        args.seed,
        cloud=(not args.local),
        tet_conc=args.tet_conc,
        amp_conc=args.amp_conc,
        initial_colony_file=args.initial_colony_file,
        initial_state_file=args.initial_state_file,
        baseline=args.baseline,
        runtime=args.runtime,
        start_time=args.start_time
    )


if __name__ == "__main__":
    import multiprocessing; multiprocessing.set_start_method("spawn")
    main()
