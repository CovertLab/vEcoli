"""
==================================
Metabolism using user-friendly FBA
==================================
"""

import argparse

# vivarium-core imports
from vivarium.core.engine import Engine
from vivarium.core.composer import Composer
from vivarium.library.dict_utils import deep_merge

# vivarium-ecoli imports
from ecoli.library.sim_data import LoadSimData
from ecoli.processes.metabolism_redux import MetabolismRedux
from ecoli.processes.stubs.exchange_stub import Exchange
from ecoli.processes.registries import topology_registry

from wholecell.utils import units

import numpy as np

# get topology from ecoli_master
metabolism_topology = topology_registry.access("ecoli-metabolism")


# make a composite with Exchange
class MetabolismExchange(Composer):
    defaults = {
        "metabolism": {
            "kinetic_rates": [],
        },
        "exchanger": {},
        "sim_data_path": "",
        "seed": 0,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.load_sim_data = LoadSimData(
            sim_data_path=self.config["sim_data_path"], seed=self.config["seed"]
        )

    def generate_processes(self, config):
        # configure metabolism
        metabolism_config = self.load_sim_data.get_metabolism_redux_config()
        metabolism_config = deep_merge(metabolism_config, config["metabolism"])
        metabolism_process = MetabolismRedux(metabolism_config)

        example_update = {
            "ATP[c]": 9064,
            "DATP[c]": 2222,
            "DCTP[c]": 1649,
            "DGTP[c]": 1647,
            "FAD[c]": 171,
            "GTP[c]": 20122,
            "LEU[c]": 325,
            "METHYLENE-THF[c]": 223,
            "NAD[c]": 769,
            "PHENYL-PYRUVATE[c]": 996,
            "REDUCED-MENAQUINONE[c]": 240,
            "UTP[c]": 14648,
        }

        # configure exchanger stub process
        # TODO -- this needs a dictionary with {mol_id: exchanged counts/sec}
        exchanger_config = {
            "exchanges": example_update,
            "time_step": metabolism_config["time_step"],
        }
        exchanger_process = Exchange(exchanger_config)

        return {
            "metabolism": metabolism_process,
            "exchanger": exchanger_process,
        }

    def generate_topology(self, config):
        return {
            "metabolism": metabolism_topology,
            "exchanger": {
                "bulk": ("bulk",),
            },
        }


def run_metabolism():
    # load the sim data
    load_sim_data = LoadSimData(sim_data_path="out/kb/simData.cPickle", seed=0)

    # Create process, experiment, loading in initial state from file.
    config = load_sim_data.get_metabolism_redux_config()
    metabolism_process = MetabolismRedux(config)

    initial_state = load_sim_data.generate_initial_state()

    metabolism_composite = metabolism_process.generate()
    metabolism_composite["topology"]["ecoli-metabolism-redux"]["bulk_total"] = ("bulk",)
    experiment = Engine(
        steps=metabolism_composite["steps"],
        topology=metabolism_composite["topology"],
        initial_state=initial_state,
    )

    experiment.update(10)

    data = experiment.emitter.get_timeseries()
    assert data is not None


def run_metabolism_composite():
    composer = MetabolismExchange({"sim_data_path": "out/kb/simData.cPickle"})
    metabolism_composite = composer.generate()
    metabolism_composite["topology"]["metabolism"]["bulk_total"] = ("bulk",)

    initial_state = composer.load_sim_data.generate_initial_state()
    initial_state["process_state"] = {
        "polypeptide_elongation": {
            "aa_exchange_rates": units.mol
            / (units.L * units.s)
            * np.array(
                [
                    -4.520e-07,
                    -3.786e-07,
                    -5.700e-09,
                    -8.880e-08,
                    0.000e00,
                    -6.600e-09,
                    -1.200e-09,
                    -6.670e-08,
                    -3.200e-09,
                    -4.800e-09,
                    -1.251e-07,
                    -7.870e-08,
                    -2.200e-09,
                    -3.860e-08,
                    -4.600e-09,
                    -1.486e-07,
                    -8.400e-09,
                    -1.560e-08,
                    -3.800e-08,
                    0.000e00,
                    -8.300e-09,
                ]
            ),
            "gtp_to_hydrolyze": 1009121.4,
        }
    }

    experiment = Engine(
        processes=metabolism_composite["processes"],
        topology=metabolism_composite["topology"],
        initial_state=initial_state,
    )

    experiment.update(10)

    data = experiment.emitter.get_data()
    assert data is not None


experiment_library = {
    "0": run_metabolism,
    "1": run_metabolism_composite,
}


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
