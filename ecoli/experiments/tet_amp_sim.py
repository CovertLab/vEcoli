import os
import argparse
import numpy as np
from scipy import constants

from vivarium.library.units import units

from ecoli.experiments.ecoli_engine_process import run_simulation
from ecoli.experiments.ecoli_master_sim import CONFIG_DIR_PATH, SimConfig
from ecoli.library.parameters import param_store
from ecoli.library.sim_data import LoadSimData

AVOGADRO = constants.N_A / units.mol


def run_sim(
    tet_conc=0,
    amp_conc=0,
    baseline=False,
    seed=0,
    cloud=False,
    initial_colony_file=None,
    initial_state_file=None,
    start_time=0,
    runtime=None,
):
    config = SimConfig()
    config.update_from_json(
        os.path.join(CONFIG_DIR_PATH, "antibiotics_ampicillin.json")
    )
    config.update_from_json(
        os.path.join(CONFIG_DIR_PATH, "antibiotics_tetracycline.json")
    )
    add_opts = {
        "spatial_environment_config": {
            "reaction_diffusion": {
                "gradient": {
                    "molecules": {
                        "tetracycline": tet_conc * units.mM,
                        "ampicillin[p]": amp_conc * units.mM,
                    }
                }
            },
            "field_timeline": {
                # Set timeline with arbitrarily high time
                # so that it is not used
                "timeline": [
                    [100000, {"tetracycline": 0}],
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
        load_sim_data = LoadSimData(**config)
        initial_state = load_sim_data.generate_initial_state()
        config["initial_state"] = make_initial_state(
            initial_state, rnai_data=config["process_configs"]["ecoli-rna-interference"]
        )
    if baseline:
        print(f"Running baseline sim (seed = {seed}).")
        config["colony_save_prefix"] = "glc_combined"
        # 26000 allows 8th round of division to mostly complete
        # Run for one timestep past that to catch inner sim emits at 26000
        if not runtime:
            runtime = 26002
            config["save"] = True
            config["save_times"] = [11550, 23100]
        config["max_duration"] = runtime
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
        config["max_duration"] = runtime
    if cloud:
        config["emitter_arg"] = {"host": "10.138.0.75:27017", "emit_limit": 5000000}

    run_simulation(config)


def update_agent(data, rnai_data=None):
    submasses = []
    metabolite_idx = -1
    for key in data["bulk"].dtype.fields.keys():
        if "submass" in key:
            submasses.append(key)
            if "metabolite" in key:
                metabolite_idx = len(submasses) - 1
    amp_mass = np.zeros(len(submasses))
    amp_mass[metabolite_idx] = (
        param_store.get(("ampicillin", "molar_mass")).to(units.fg / units.mol)
        / AVOGADRO
    ).magnitude
    tet_mass = np.zeros(len(submasses))
    tet_mass[metabolite_idx] = (
        param_store.get(("tetracycline", "mass")).to(units.fg / units.mol) / AVOGADRO
    ).magnitude
    tet_ribo_30s_mass = data["bulk"][submasses][
        data["bulk"]["id"] == "CPLX0-3953[c]"
    ].copy()
    tet_ribo_30s_mass["metabolite_submass"] += tet_mass[metabolite_idx]
    marR_tet_mass = data["bulk"][submasses][
        data["bulk"]["id"] == "CPLX0-7710[c]"
    ].copy()
    marR_tet_mass["metabolite_submass"] += tet_mass[metabolite_idx]
    new_bulk = [
        ("marR-tet[c]", 0) + tuple(marR_tet_mass[0]),
        ("tetracycline[p]", 0) + tuple(tet_mass),
        ("tetracycline[c]", 0) + tuple(tet_mass),
        ("CPLX0-3953-tetracycline[c]", 0) + tuple(tet_ribo_30s_mass[0]),
        ("ampicillin[p]", 0) + tuple(amp_mass),
        ("ampicillin_hydrolyzed[p]", 0) + tuple(tet_mass),
    ]

    # Add RNA duplexes
    if rnai_data:
        # Get RNA mass data
        duplex_masses = []
        for srna_id, target_id in zip(rnai_data["srna_ids"], rnai_data["target_ids"]):
            srna_mass = np.array(
                list(data["bulk"][submasses][data["bulk"]["id"] == srna_id][0])
            )
            target_mass = np.array(
                list(data["bulk"][submasses][data["bulk"]["id"] == target_id][0])
            )
            duplex_masses.append(srna_mass + target_mass)
        new_bulk += [
            (str(duplex_id), 0) + tuple(duplex_mass)
            for duplex_id, duplex_mass in zip(rnai_data["duplex_ids"], duplex_masses)
        ]
    new_bulk = np.array(new_bulk, dtype=data["bulk"].dtype)
    data["bulk"] = np.append(data["bulk"], new_bulk)
    # Add promoter binding data for marA and marR
    new_bound_TF = []
    for bound_TF in data["unique"]["promoter"]["bound_TF"]:
        # Bound TF boolean mask should be 4th attr
        new_bound_TF.append(np.append(bound_TF, [False, False]))
    new_dtype = [
        dt for dt in data["unique"]["promoter"].dtype.descr if "bound_TF" not in dt
    ]
    new_dtype.append(("bound_TF", "?", (len(new_bound_TF[0]),)))
    new_promoters = np.empty((len(data["unique"]["promoter"]),), new_dtype)
    for field in data["unique"]["promoter"].dtype.names:
        if field != "bound_TF":
            new_promoters[field] = data["unique"]["promoter"][field]
        else:
            new_promoters[field] = new_bound_TF
    data["unique"]["promoter"] = new_promoters
    return data


def make_initial_state(initial_state, rnai_data=None):
    # Modify each cell in colony individually
    if "agents" in initial_state:
        for agent_id, agent_data in initial_state["agents"].items():
            initial_state["agents"][agent_id] = update_agent(agent_data, rnai_data)
            # Save bulk and unique dtypes
            agent_data["bulk_dtypes"] = str(agent_data["bulk"].dtype)
            agent_data["unique_dtypes"] = {}
            for name, mols in agent_data["unique"].items():
                agent_data["unique_dtypes"][name] = str(mols.dtype)
    else:
        initial_state = update_agent(initial_state, rnai_data)
        initial_state["bulk_dtypes"] = str(initial_state["bulk"].dtype)
        initial_state["unique_dtypes"] = {}
        for name, mols in initial_state["unique"].items():
            initial_state["unique_dtypes"][name] = str(mols.dtype)
    return initial_state


def generate_data(
    seed,
    cloud,
    tet_conc,
    amp_conc,
    initial_colony_file,
    initial_state_file,
    baseline,
    start_time,
    runtime,
):
    if baseline:
        run_sim(
            tet_conc=0,
            amp_conc=0,
            seed=seed,
            cloud=cloud,
            start_time=0,
            initial_state_file=initial_state_file,
            baseline=baseline,
            runtime=runtime,
        )
    else:
        run_sim(
            tet_conc=tet_conc,
            amp_conc=amp_conc,
            seed=seed,
            cloud=cloud,
            initial_colony_file=initial_colony_file,
            initial_state_file=initial_state_file,
            start_time=start_time,
            runtime=runtime,
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
        help="Starting external tetracycline concentration (mM)",
    )
    parser.add_argument(
        "-a",
        "--amp_conc",
        default=param_store.get(("ampicillin", "mic")),
        type=float,
        help="Starting external ampicillin concentration (mM)",
    )
    parser.add_argument(
        "-r", "--runtime", default=None, type=int, help="Custom simulation run time."
    )
    parser.add_argument(
        "-n",
        "--start_time",
        default=None,
        type=int,
        help="Custom simulation start time.",
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
        start_time=args.start_time,
    )


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    main()
