"""
========================
E. coli master composite
========================
"""

import os
import argparse

from vivarium.core.composer import Composer
from vivarium.core.engine import pp, Engine
from vivarium.library.topology import assoc_path
from vivarium.library.dict_utils import deep_merge

# sim data
from ecoli.library.sim_data import LoadSimData

# logging
from vivarium.library.wrappers import make_logging_process

# vivarium-ecoli processes
from ecoli.processes.cell_division import Division
from ecoli.plots.topology import get_ecoli_master_topology_settings

# state
from ecoli.states.wcecoli_state import get_state_from_file

# plotting
from vivarium.plots.topology import plot_topology


RAND_MAX = 2**31
SIM_DATA_PATH = 'reconstruction/sim_data/kb/simData.cPickle'

MINIMAL_MEDIA_ID = 'minimal'
AA_MEDIA_ID = 'minimal_plus_amino_acids'
ANAEROBIC_MEDIA_ID = 'minimal_minus_oxygen'


class Ecoli(Composer):

    defaults = {
        'time_step': 2.0,
        'parallel': False,
        'seed': 0,
        'sim_data_path': SIM_DATA_PATH,
        'daughter_path': tuple(),
        'agent_id': '0',
        'agents_path': ('..', '..', 'agents',),
        'division': {
            'threshold': 2220},  # fg
        'divide': False,
        'blame': False
    }

    def __init__(self, config):
        super().__init__(config)

        self.load_sim_data = LoadSimData(
            sim_data_path=self.config['sim_data_path'],
            seed=self.config['seed'])

        self.processes = config['processes']
        self.topology = config['topology']

    def initial_state(self, config=None, path=()):
        if config:
            initial_time = config.get("initial_time", 0)
        initial_state = get_state_from_file(path=f'data/wcecoli_t{initial_time}.json')
        embedded_state = {}
        assoc_path(embedded_state, path, initial_state)
        return embedded_state

    def generate_processes(self, config):
        time_step = config['time_step']
        parallel = config['parallel']

        # get process configs
        process_configs = config['process_configs']
        for process in process_configs.keys():
            if process_configs[process] == "sim_data":
                process_configs[process] = self.load_sim_data.get_config_by_name(process)
            elif process_configs[process] == "default":
                process_configs[process] = None
            else:
                # user passed a dict, deep-merge with config from LoadSimData
                # if it exists, else, deep-merge with default
                try:
                    default = self.load_sim_data.get_config_by_name(process)
                except KeyError:
                    default = self.processes[process].defaults
                
                process_configs[process] = deep_merge(dict(default), process_configs[process])

        # make the processes
        processes = {
            process_name: (process(process_configs[process_name])
                           if not config['log_updates']
                           else make_logging_process(process)(process_configs[process_name]))
            for (process_name, process) in self.processes.items()
        }

        # add division
        if self.config['divide']:
            division_config = dict(
                config['division'],
                agent_id=self.config['agent_id'],
                composer=self)
            processes['division'] = Division(division_config)

        return processes

    def generate_topology(self, config):
        topology = {}

        # make the topology
        for process_id, ports in self.topology.items():
            topology[process_id] = ports
            if config['blame']:
                topology[process_id]['log_update'] = ('log_update', process_id,)

        # add division
        if self.config['divide']:
            topology['division'] = {
                'variable': ('listeners', 'mass', 'cell_mass'),
                'agents': config['agents_path']}

        return topology


def run_ecoli(
        total_time=10,
        divide=False,
        progress_bar=True,
        log_updates=False,
        time_series=True
):
    """
    Simple way to run ecoli_master simulations. For full API, see ecoli.experiments.ecoli_master_sim.

    Arguments:
        * **total_time** (:py:class:`int`): the total runtime of the experiment
        * **divide** (:py:class:`bool`): whether to incorporate division
        * **progress_bar** (:py:class:`bool`): whether to show a progress bar
        * **log_updates**  (:py:class:`bool`): whether to save updates from each process
        * **time_series** (:py:class:`bool`): whether to return data in timeseries format
    Returns:
        * output data
    """
    
    from ecoli.experiments.ecoli_master_sim import EcoliSim
    
    sim = EcoliSim.from_file()
    sim.total_time = total_time
    sim.divide = divide
    sim.progress_bar = progress_bar
    sim.log_updates = log_updates
    sim.raw_output = not time_series

    return sim.run()


def test_division():
    """
    Work in progress to get division working
    * TODO -- unique molecules need to be divided between daughter cells!!! This can get sophisticated
    """

    from ecoli.experiments.ecoli_master_sim import EcoliSim

    sim = EcoliSim.from_file()
    sim.config['division'] = {'threshold' : 1170}
    sim.total_time = 10
    sim.divide = True
    sim.progress_bar = False

    output = sim.run()


def ecoli_topology_plot(config={}, filename=None, out_dir=None):
    """Make a topology plot of Ecoli"""
    agent_id_config = {'agent_id': '1'}
    ecoli = Ecoli({**agent_id_config, **config})
    settings = get_ecoli_master_topology_settings()
    topo_plot = plot_topology(
        ecoli,
        filename=filename,
        out_dir=out_dir,
        settings=settings)
    return topo_plot


test_library = {
    '0': run_ecoli,
    '1': test_division,
}


def main():
    out_dir = os.path.join('out', 'ecoli_master')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    parser = argparse.ArgumentParser(description='ecoli_master')
    parser.add_argument(
        '--name', '-n', default=[], nargs='+',
        help='test ids to run')
    parser.add_argument(
        '--topology', '-t', action='store_true', default=False,
        help='save a topology plot of ecoli master')
    args = parser.parse_args()

    if args.topology:
        ecoli_topology_plot(filename='ecoli_master', out_dir=out_dir)
    elif args.name:
        for name in args.name:
            test_library[name]()
    else:
        output = run_ecoli(log_updates=True)


if __name__ == '__main__':
    main()