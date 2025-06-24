"""
==================================
Metabolism using user-friendly FBA
==================================
"""

import argparse

# vivarium-core imports
import pytest

# vivarium-ecoli imports
from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH
from ecoli.library.json_state import get_state_from_file


import numpy as np
import pathlib
import datetime
import dill


def run_ecoli_with_metabolism_redux(
    filename="metabolism_redux",
    max_duration=5000,
    divide=True,
    initial_state_file="wcecoli_t0",
    progress_bar=True,
    log_updates=False,
    emitter="timeseries",
    name="metabolism-redux",
    raw_output=False,
    save=False,
    # save_times=4,
):
    # filename = 'default'
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + filename + ".json")
    sim.max_duration = max_duration
    sim.divide = divide
    sim.progress_bar = progress_bar
    sim.log_updates = log_updates
    sim.emitter = emitter
    sim.initial_state = get_state_from_file(path=f"data/{initial_state_file}.json")
    sim.raw_output = raw_output
    sim.save = save

    sim.build_ecoli()
    sim.run()

    query = []
    folder = f"out/fbagd/{name}_{max_duration}_{datetime.date.today()}/"
    save_sim_output(folder, query, sim, save_model=True)


# disables growth rate control
def run_ecoli_with_metabolism_redux_classic(
    filename="metabolism_redux_classic",
    max_duration=10,
    divide=True,
    # initial_state_file='wcecoli_t0', # 'met_division_test_state',
    progress_bar=True,
    log_updates=False,
    emitter="timeseries",  # 'timeseries',
    name="convex_kinetics_minimal",
    raw_output=False,
    save=True,
    save_times=[1, 10],
    condition="basal",  # basal, with_aa
    fixed_media="minimal",  # minimal, minimal_plus_amino_acids
):
    # filename = 'default'
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + filename + ".json")
    sim.max_duration = max_duration
    sim.divide = divide
    sim.progress_bar = progress_bar
    sim.log_updates = log_updates
    sim.emitter = emitter
    # sim.initial_state = get_state_from_file(path=f'data/{initial_state_file}.json')
    sim.raw_output = raw_output
    sim.save = save
    sim.save_times = save_times

    sim.condition = condition
    sim.fixed_media = fixed_media

    sim.seed = 12

    # # simplify working with uptake
    # sim.initial_state['environment']['exchange_data']['constrained'] = {}
    # sim.initial_state['environment']['exchange_data']['unconstrained'].add('GLC[p]')
    #
    # # in sim.initial_state['environment']['exchange_data']['unconstrained'], edit the set of molecules to be exchanged
    # sim.initial_state['environment']['exchange_data']['unconstrained'].remove('GLC[p]')
    # sim.initial_state['environment']['exchange_data']['unconstrained'].add('FRU[p]')

    # this means that sims will not create conflicting random indices when loading from saved state
    # if initial_state_file == 'wcecoli_t0':
    #     sim.seed += 1
    # else:
    #     sim.seed += int(sim.initial_state['agents']['0']['global_time'])

    sim.build_ecoli()

    sim.run()

    query = []
    folder = f"out/cofactors/{name}_{max_duration}_{datetime.date.today()}/"
    save_sim_output(folder, query, sim, save_model=True)


def run_colony(
    filename="metabolism_redux_classic",
    max_duration=1400,
    divide=True,
    # initial_state_file='wcecoli_t0', # 'met_division_test_state',
    progress_bar=True,
    log_updates=False,
    emitter="timeseries",  # 'timeseries',
    name="metabolism-redux-classic-rich",
    raw_output=False,
    save=True,
    save_times=[1, 200, 400, 1300],
    condition="with_aa",  # basal, with_aa
    fixed_media="minimal_plus_amino_acids",  # minimal, minimal_plus_amino_acids
):
    # filename = 'default'
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + filename + ".json")
    sim.max_duration = max_duration
    sim.divide = divide
    sim.progress_bar = progress_bar
    sim.log_updates = log_updates
    sim.emitter = emitter
    # sim.initial_state = get_state_from_file(path=f'data/{initial_state_file}.json')
    sim.raw_output = raw_output
    sim.save = save
    sim.save_times = save_times

    sim.condition = condition
    sim.fixed_media = fixed_media

    for seed in [i for i in range(4, 9, 1)]:
        sim.seed = seed

        sim.build_ecoli()

        sim.run()

        query = []
        folder = f"out/cofactors/rich-{seed}/"
        save_sim_output(folder, query, sim, save_model=False)


@pytest.mark.slow
def test_ecoli_with_metabolism_redux(
    filename="metabolism_redux",
    max_duration=4,
    divide=False,
    progress_bar=True,
    log_updates=False,
    emitter="timeseries",
):
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + filename + ".json")
    sim.max_duration = max_duration
    sim.divide = divide
    sim.progress_bar = progress_bar
    sim.log_updates = log_updates
    sim.emitter = emitter
    sim.build_ecoli()

    # run simulation and add asserts to output
    sim.run()


@pytest.mark.slow
def test_ecoli_with_metabolism_redux_div(
    filename="metabolism_redux",
    max_duration=4,
    divide=True,
    emitter="timeseries",
):
    # TODO (Cyrus) - Add test that affirms structure of output query.
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + filename + ".json")
    sim.max_duration = max_duration
    sim.divide = divide
    sim.emitter = emitter
    sim.build_ecoli()

    sim.run()

    query = []
    agents = sim.query()["agents"].keys()
    for agent in agents:
        query.extend(
            [
                ("agents", agent, "listeners", "fba_results"),
                ("agents", agent, "listeners", "mass"),
                ("agents", agent, "bulk"),
            ]
        )
    output = sim.query(query)

    # test that water is being used (model is running)
    assert (
        sum(
            output["agents"][agent]["listeners"]["fba_results"][
                "estimated_exchange_dmdt"
            ]["WATER"]
        )
        != 0
    )


@pytest.mark.slow
def test_ecoli_with_metabolism_classic(
    filename="metabolism_redux_classic",
    max_duration=4,
    divide=False,
    progress_bar=True,
    log_updates=False,
    emitter="timeseries",
):
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + filename + ".json")
    sim.max_duration = max_duration
    sim.divide = divide
    sim.progress_bar = progress_bar
    sim.log_updates = log_updates
    sim.emitter = emitter
    sim.build_ecoli()

    # run simulation and add asserts to output
    sim.run()


# @pytest.mark.slow
# def test_ecoli_with_metabolism_classic_div(
#         filename='metabolism_redux_classic',
#         max_duration=10,
#         divide=True,
#         emitter='timeseries',
#         initial_state_file='met_division_test_state',
# ):
#     # TODO (Cyrus) - Add test that affirms structure of output query.
#     sim = EcoliSim.from_file(CONFIG_DIR_PATH + filename + '.json')
#     sim.max_duration = max_duration
#     sim.initial_state = get_state_from_file(path=f'data/{initial_state_file}.json')
#
#     sim.divide = divide
#     sim.emitter = emitter
#
#     # this means that sims will not create conflicting random indices
#     sim.seed += int(sim.initial_state['agents']['0']['global_time'])
#
#     sim.build_ecoli()
#
#     sim.run()
#
#     # assert division occured
#     assert len(sim.query()['agents']) == 3, "Cell did not divide in metabolism division test"


def run_ecoli_with_default_metabolism(
    filename="default",
    max_duration=10,
    divide=False,
    progress_bar=True,
    log_updates=False,
    emitter="timeseries",
):
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + filename + ".json")
    sim.max_duration = max_duration
    sim.divide = divide
    sim.progress_bar = progress_bar
    sim.log_updates = log_updates
    sim.emitter = emitter
    sim.build_ecoli()

    sim.run()
    # output = sim.query()
    output = sim.ecoli_experiment.emitter.get_timeseries()

    folder = f"out/fbagd/{max_duration}/{datetime.datetime.now()}/"
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    np.save(folder + "fba_results.npy", output["listeners"]["fba_results"])
    np.save(folder + "mass.npy", output["listeners"]["mass"])
    np.save(folder + "bulk.npy", output["bulk"])
    np.save(
        folder + "stoichiometry.npy",
        sim.ecoli_experiment.steps["ecoli-metabolism"].model.stoichiometry,
    )


experiment_library = {
    "2": run_ecoli_with_metabolism_redux,
    "2a": run_ecoli_with_metabolism_redux_classic,
    "2b": run_colony,
    "3": test_ecoli_with_metabolism_redux,
    "3a": test_ecoli_with_metabolism_classic,
    "4": test_ecoli_with_metabolism_redux_div,
    "5": run_ecoli_with_default_metabolism,
}


def save_sim_output(folder, query, sim, save_model=False):
    agents = sim.query()["agents"].keys()
    for agent in agents:
        query = []
        query.extend(
            [
                ("agents", agent, "listeners", "fba_results"),
                ("agents", agent, "listeners", "mass"),
                ("agents", agent, "listeners", "unique_molecule_counts"),
                ("agents", agent, "bulk"),
            ]
        )
        output = sim.query(query)
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        np.save(folder + f"{agent}_output.npy", output)

    if save_model:
        f = open(folder + "agent_steps.pkl", "wb")
        dill.dump(sim.ecoli_experiment.steps["agents"][agent], f)
        f.close()


# run experiments with command line arguments: python ecoli/experiments/metabolism_redux_sim.py -n exp_id
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="user-friendly metabolism")
    parser.add_argument("--name", "-n", default=[], nargs="+", help="test ids to run")
    args = parser.parse_args()
    run_all = not args.name

    for name in args.name:
        experiment_library[name]()
    if run_all:
        for name, test in experiment_library.items():
            test()
