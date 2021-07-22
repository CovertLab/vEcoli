"""
Run simulations of Ecoli Master
"""

import argparse

from vivarium.core.engine import Engine
from ecoli.composites.ecoli_master import Ecoli
from ecoli import process_registry



def ecoli_from_file(file_path):
    pass



class EcoliSim:

    def build_from_cli(self):
        parser = argparse.ArgumentParser(description='ecoli_master')
        parser.add_argument(
            '--topology', '-t', action='store_true', default=False,
            help='save a topology plot of ecoli master')
        parser.add_argument(
            '--blame', '-b', action='store_true', default=False,
            help='when running simulation, create a report of which processes affected which molecules')
        parser.add_argument(
            '--debug', '-d', action='store_true', default=False,
            help='run tests, generating a report of failures/successes')
        args = parser.parse_args()

        # get processes from process_registry with

        # make the ecoli model
        self.ecoli = ecoli_from_file(args.filepath, args.blame)

        self.total_time = args.total_time

    def run(self):
        # make the experiment
        ecoli_experiment = Engine({
            'processes': self.ecoli.processes,
            'topology': self.ecoli.topology,
            'initial_state': self.initial_state,
            'progress_bar': self.progress_bar,
        })

        # run the experiment
        ecoli_experiment.update(self.total_time)

        # retrieve the data
        if self.timeseries_output:
            return ecoli_experiment.emitter.get_timeseries()
        elif self.data_output:
            return ecoli_experiment.emitter.get_data()


# python ecoli/experiment/ecoli_master_sim.py -p [path to file] -e [emitter] -t [run time] -s [seed]
if __name__ == '__main__':
    ecoli_sim = EcoliSim()
    ecoli_sim.build_from_cli()
    ecoli_sim.run()
