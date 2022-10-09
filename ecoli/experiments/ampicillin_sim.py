import os
import argparse

from ecoli.composites.ecoli_engine_process import run_simulation
from ecoli.experiments.ecoli_master_sim import CONFIG_DIR_PATH, SimConfig
from ecoli.library.parameters import param_store
from migration.migration_utils import recursive_compare


def run_sim(
    amp_conc=0,
    baseline=False,
    seed=0,
    total_time=10000,
    cloud=False,
    initial_colony_file=None,
    start_time=0,
):
    config = SimConfig()
    config.update_from_json(
        os.path.join(CONFIG_DIR_PATH, "antibiotics_ampicillin.json")
    )
    config["initial_colony_file"] = initial_colony_file

    ampicillin_gradient = {
        "total_time": total_time,
        "spatial_environment_config": {
            "reaction_diffusion": {"gradient": {"molecules": {"ampicillin[p]": amp_conc}}},
            "field_timeline": {
                "timeline": [
                    [
                        100000,
                        {"ampicillin[p]": 0},
                    ],  # using 100000 sec because we don't actually want to set ampicillin to 0
                ]
            },
        },
        "seed": seed,
        "start_time": start_time,
        "colony_save_prefix": "amp"
    }

    if cloud:
        ampicillin_gradient["save"] = True
        if baseline:
            ampicillin_gradient["save_times"] = [11550, 23100, 26000]
            # 26000 seconds catches the start of the 9th round of division
            ampicillin_gradient["total_time"] = 26000
        else:
            ampicillin_gradient["save_times"] = [11550]
            ampicillin_gradient["total_time"] = 14450
        ampicillin_gradient["emitter_arg"] = [
            ["host", "10.138.0.75:27017"],
            ["emit_limit", 5000000],
        ]
    config.update_from_dict(ampicillin_gradient)
    if baseline:
        # Note: keeps reduced murein
        config["add_processes"].remove("ecoli-cell-wall")
        config["add_processes"].remove("ecoli-pbp-binding")

        config["process_configs"].pop("ecoli-cell-wall")
        config["process_configs"].pop("ecoli-pbp-binding")

        config["topology"].pop("ecoli-cell-wall")
        config["topology"].pop("ecoli-pbp-binding")

        config["engine_process_reports"].clear()

        config["flow"]["ecoli-mass-listener"] = [("ecoli-metabolism",)]
        config["flow"].pop("ecoli-pbp-binding")

        config["initial_state_file"] = "wcecoli_t0"
        config["initial_state_overrides"] = ["overrides/reduced_murein"]
        config["colony_save_prefix"] = "glc"
    run_simulation(config)


def generate_data(seed, cloud, conc, total_time, initial_colony_file):
    print(f"Ampicillin concentration: {conc}")

    run_sim(
        conc,
        seed=seed,
        cloud=cloud,
        initial_colony_file=initial_colony_file,
        start_time=11550,
        total_time=total_time,
    )


def cli():
    parser = argparse.ArgumentParser("Run ampicillin simulations.")

    parser.add_argument(
        "-s", "--seed", default=0, type=int, help="Random seed for simulation."
    )
    parser.add_argument("-l", "--local", action="store_true")
    parser.add_argument(
        "-i",
        "--initial_file",
        default="glc_reduced_murein_seed_0_colony_t11550",
        help="colony save state to run the simulation off of",
    )
    parser.add_argument(
        "-t",
        "--total_time",
        default=10000,
        type=int,
        help="total time to run the simulation",
    )
    parser.add_argument(
        "-c",
        "--concentration",
        default=param_store.get(("ampicillin", "mic")),
        type=float,
        help="Starting tetracycline concentration"
    )

    args = parser.parse_args()

    print(f"""Running ampicillin simulation with simulation with
              seed = {args.seed}
              for {args.total_time} seconds.""")

    generate_data(
        args.seed,
        cloud=(not args.local),
        conc=args.concentration,
        total_time=args.total_time,
        initial_colony_file=args.initial_file,
    )


def main():
    cli()


if __name__ == "__main__":
    cli()
