"""
tests that vivarium-ecoli process update are the same as saved wcEcoli updates

"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, chisquare

from vivarium.processes.clock import Clock

from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH
from migration.migration_utils import *

from ecoli.processes.listeners.mass_listener import MassListener

from ecoli.library.schema import arrays_from
from ecoli.states.wcecoli_state import get_state_from_file
from migration.plots import qqplot
from migration.migration_utils import array_diffs_report


load_sim_data = LoadSimData(sim_data_path=SIM_DATA_PATH,
                            seed=0)


def test_mass_listener():
    # Create process, experiment, loading in initial state from file.
    config = load_sim_data.get_mass_listener_config()
    process = MassListener(config)

    topology = {
        'bulk': ('bulk',),
        'unique': ('unique',),
        'listeners': ('listeners',)
    }

    initial_state = get_state_from_file(
        path=f'data/wcecoli_t{0}.json')

    # make the experiment
    ecoli_experiment = Engine({
        'processes': {'mass': process, 'clock' : Clock({'time_step' : 2.0})},
        'topology': {'mass': topology, 'clock' : {'global_time' : ('global_time',)}},
        'initial_state': initial_state,
        'progress_bar': False
    })

    # run the experiment and get output
    ecoli_experiment.update(4)
    actual_update_t0 = ecoli_experiment.emitter.get_timeseries()['listeners']['mass']

    with open("data/mass_listener_update_t0.json") as f:
        wc_update_t0 = json.load(f)

    assertions(actual_update_t0, wc_update_t0)


def assertions(actual_update, expected_update):
    test_structure = {
        'cell_mass' : scalar_almost_equal,
        'water_mass' : scalar_almost_equal,
        'dry_mass' : scalar_almost_equal,
        'rnaMass' : scalar_almost_equal,
        'rRnaMass' : scalar_almost_equal,
        'tRnaMass' : scalar_almost_equal,
        'mRnaMass' : scalar_almost_equal,
        'dnaMass' : scalar_almost_equal,
        'proteinMass' : scalar_almost_equal,
        'smallMoleculeMass' : scalar_almost_equal,
        'volume' : scalar_almost_equal,
        'proteinMassFraction' : scalar_almost_equal,
        'rnaMassFraction' : scalar_almost_equal,
        'growth' : scalar_almost_equal,
        'instantaniousGrowthRate' : scalar_almost_equal,
        'dryMassFoldChange' : scalar_almost_equal,
        'proteinMassFoldChange' : scalar_almost_equal,
        'rnaMassFoldChange' : scalar_almost_equal,
        'smallMoleculeFoldChange' : scalar_almost_equal
    }

    tests = ComparisonTestSuite(test_structure, fail_loudly=False)
    tests.run_tests(actual_update,
                    expected_update,
                    verbose=True)

    # print(tests.report)
    # tests.dump_report()

    tests.fail()


if __name__ == "__main__":
    test_mass_listener()
