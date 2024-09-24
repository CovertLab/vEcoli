"""

Ecoli Composer.
- ecoli/composites/ecoli_master.py

EcoliSim. The main interface for running the ecoli model
- ecoli/experiments/ecoli_master_sim.py

Metabolism process
- ecoli/processes/metabolism.py

Metabolism dataclass:
- reconstruction/ecoli/dataclasses/process/metabolism.py
"""

from vivarium.core.engine import pf
from ecoli.experiments.ecoli_master_sim import EcoliSim


def run_default():
    sim = EcoliSim.from_file()
    sim.total_time = 2
    print(f'INITIAL PROCESS CONFIGS: {pf(sim.config["process_configs"])}')

    # # TODO update process configs
    # sim.override_config({
    #     'process_configs': {
    #         'ecoli-metabolism': {
    #             'config': {}
    #         }
    #     }
    # })
    # print(f'NEW PROCESS CONFIGS: {pf(sim.config["process_configs"])}')

    # build the ecoli model
    sim.build_ecoli()

    # retrieve kinetic parameters
    sim.processes['ecoli-metabolism'].parameters['metabolism'].kinetic_constraint_reactions
    sim.processes['ecoli-metabolism'].parameters['metabolism'].kinetic_constraint_enzymes
    sim.processes['ecoli-metabolism'].parameters['metabolism'].kinetic_constraint_substrates
    sim.processes['ecoli-metabolism'].parameters['metabolism']._kcats
    sim.processes['ecoli-metabolism'].parameters['metabolism']._saturations
    sim.processes['ecoli-metabolism'].parameters['metabolism']._enzymes
    sim.processes['ecoli-metabolism'].parameters['metabolism'].constraint_is_kcat_only
    sim.processes['ecoli-metabolism'].parameters['metabolism']._kinetic_constraints

    # # get the kinetic constraints function
    # # this is the way to get kinetic constraints given the enzymes and substrates
    # sim.processes['ecoli-metabolism'].parameters['metabolism'].get_kinetic_constraints(enzymes: Unum, substrates: Unum)

    # run the simulation
    sim.run()

    # get the data
    timeseries = sim.ecoli_experiment.emitter.get_timeseries()


if __name__ == "__main__":
    run_default()
