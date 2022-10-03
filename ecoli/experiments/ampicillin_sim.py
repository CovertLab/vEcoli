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
        os.path.join(CONFIG_DIR_PATH, "antibiotics_tetracycline.json")
    )
    ampicillin_gradient = {
        "total_time": total_time,
        "spatial_environment_config": {
            "reaction_diffusion": {"gradient": {"molecules": {"ampicillin": amp_conc}}},
            "field_timeline": {
                "timeline": [
                    [
                        100000,
                        {"ampicillin": 0},
                    ],  # using 100000 sec because we don't actually want to set ampicillin to 0
                ]
            },
        },
        "seed": seed,
        "start_time": start_time,
    }
    if cloud:
        ampicillin_gradient["save"] = True
        if baseline:
            ampicillin_gradient["save_times"] = [11550, 23100, 27000]
            ampicillin_gradient["total_time"] = 27000
        else:
            ampicillin_gradient["save_times"] = [11550]
            ampicillin_gradient["total_time"] = 15540
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
    run_simulation(config)


# def add_mar_tf(data):
#     # Add initial count for marR-tet complex
#     data["bulk"]["marR-tet[c]"] = 0
#     # Add promoter binding data for marA and marR
#     for promoter_data in data["unique"]["promoter"].values():
#         promoter_data["bound_TF"] += [False, False]
#     return data


# def make_tet_initial_state(initial_colony_file):
#     with open(f"data/{initial_colony_file}.json") as f:
#         initial_state = json.load(f)
#     for agent_id, agent_data in initial_state["agents"].items():
#         initial_state["agents"][agent_id] = add_mar_tf(agent_data)
#     if os.path.exists(f"data/tet_{initial_colony_file}.json"):
#         with open(f"data/tet_{initial_colony_file}.json") as f:
#             existing_initial_state = json.load(f)
#         if recursive_compare(initial_state, existing_initial_state):
#             return
#         else:
#             print(f"tet_{initial_colony_file}.json out of date, updating")
#     with open(f"data/tet_{initial_colony_file}.json", "w") as f:
#         json.dump(initial_state, f)


def generate_data(seed, cloud):
    # run_sim(0, seed = 0, baseline=True, cloud=True)
    run_sim(
        param_store.get(("ampicillin", "mic")),  # 0.003375,
        seed=seed,
        cloud=cloud,
        initial_colony_file="seed_0_colony_t11550",
        start_time=11550,
    )


def cli():
    parser = argparse.ArgumentParser("Run ampicillin simulations.")

    parser.add_argument(
        "-s", "--seed", default=0, type=int, help="Random seed for simulation."
    )
    parser.add_argument("-l", "--local", action="store_true")

    args = parser.parse_args()

    generate_data(args.seed, cloud=(not args.local))


def main():
    cli()


if __name__ == "__main__":
    cli()
