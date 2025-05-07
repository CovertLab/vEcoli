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
from ecoli.composites.ecoli_master import Ecoli
from vivarium.library.topology import assoc_path



def run_default():
    sim = EcoliSim.from_file(filepath='ecoli/composites/ecoli_configs/test_configs/test_vkecoli.json')
    # sim.total_time = 20
    print(f'INITIAL PROCESS CONFIGS: {pf(sim.config["process_configs"])}')

    # build the ecoli model
    # sim.build_ecoli()

    # build ecoli manually, reproducing the build_ecoli method
    sim.processes = sim._retrieve_processes(
        sim.processes,
        sim.add_processes,
        sim.exclude_processes,
        sim.swap_processes,
    )
    sim.topology = sim._retrieve_topology(
        sim.topology, sim.processes, sim.swap_processes, sim.log_updates
    )
    sim.process_configs = sim._retrieve_process_configs(
        sim.process_configs, sim.processes
    )

    # Prevent clashing unique indices by reseeding when loading
    # a saved state (assumed to have name 'vivecoli_t{save time}')
    initial_state_path = sim.config.get("initial_state_file", "")
    if initial_state_path.startswith("vivecoli"):
        time_str = initial_state_path[len("vivecoli_t"):]
        seed = int(float(time_str))
        sim.config["seed"] += seed

    # initialize the ecoli composer
    ecoli_composer = Ecoli(sim.config)


    # rebuild the processes and steps with the new config
    ecoli_composer.processes_and_steps = ecoli_composer.generate_processes_and_steps(ecoli_composer.config)

    # set path at which agent is initialized
    path = tuple()
    if sim.divide or sim.spatial_environment:
        path = (
            "agents",
            sim.agent_id,
        )

    # get initial state
    initial_cell_state = ecoli_composer.initial_state()
    initial_cell_state = assoc_path({}, path, initial_cell_state)

    # generate the composite at the path
    sim.ecoli = ecoli_composer.generate(path=path)
    # Some processes define their own initial_state methods
    # Incoporate them into the generated initial state
    sim.generated_initial_state = sim.ecoli.initial_state(
        {"initial_state": initial_cell_state}
    )


    # run the simulation
    sim.run()

    # get the data
    timeseries = sim.ecoli_experiment.emitter.get_timeseries()

    print(timeseries['agents']['0']['species_store'])

#%%


if __name__ == "__main__":
    run_default()



#%%