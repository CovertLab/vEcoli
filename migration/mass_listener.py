"""
tests that vivarium-ecoli process update are the same as saved wcEcoli updates

"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, chisquare

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

    actual_update_t0 = run_ecoli_process(process, topology, initial_time=0)

    with open("data/mass_listener_update_t0.json") as f:
        wc_update_t0 = json.load(f)

    assertions(actual_update_t0, wc_update_t0)


def assertions(actual_update, expected_update):
    test_structure = {
        'listeners': {
        }
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
