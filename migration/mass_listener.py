"""
tests that vivarium-ecoli process update are the same as saved wcEcoli updates

"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, chisquare

from vivarium.processes.clock import Clock

from ecoli.library.schema import array_from
from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH
from migration.migration_utils import *

from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH

from ecoli.library.schema import arrays_from
from ecoli.states.wcecoli_state import get_state_from_file
from migration.plots import qqplot
from migration.migration_utils import array_diffs_report


def test_mass_listener():
    ecoli_sim = EcoliSim.from_file(CONFIG_DIR_PATH + '/test_configs/test_mass_listener.json')
    actual_updates = ecoli_sim.run()

    # actual_updates = run_ecoli(total_time=4, divide=False, blame=False, time_series=False)
    
    actual_update_t0 = actual_updates[0.0]['listeners']['mass']
    actual_update_t2 = actual_updates[2.0]['listeners']['mass']

    with open("data/mass_listener_update_t0.json") as f:
        wc_update_t0 = json.load(f)
    
    with open("data/mass_listener_update_t2.json") as f:
        wc_update_t2 = json.load(f)

    assertions(actual_update_t0, wc_update_t0, time=0)
    
    #TODO: wcEcoli shrinks from time 0 to 2 for some reason...
    #assertions(actual_update_t2, wc_update_t2, time=2)
    


def assertions(actual_update, expected_update, time=0):
    def both_nan(x, y):
        return np.all(np.isnan([x, y]))

    # TODO: mRNA and dna are affected by partitioning.
    # See massDiffs in wcEcoli/wholecell/sim/unique_molecules.py.
    # Also, add tests for compartment masses.
    test_structure = {
        'cell_mass': scalar_almost_equal,
        'water_mass': scalar_almost_equal,
        'dry_mass': scalar_almost_equal,
        'rnaMass': scalar_almost_equal,
        'rRnaMass': scalar_almost_equal,
        'tRnaMass': scalar_almost_equal,
        #'mRnaMass': scalar_almost_equal,
        #'dnaMass': scalar_almost_equal,
        'proteinMass': scalar_almost_equal,
        'smallMoleculeMass': scalar_almost_equal,
        'volume': scalar_almost_equal,
        'proteinMassFraction': scalar_almost_equal,
        'rnaMassFraction': scalar_almost_equal,
        'growth': both_nan if time == 0 else scalar_almost_equal,
        'instantaniousGrowthRate': both_nan if time == 0 else scalar_almost_equal,
        'dryMassFoldChange': scalar_almost_equal,
        'proteinMassFoldChange': scalar_almost_equal,
        'rnaMassFoldChange': scalar_almost_equal,
        'smallMoleculeFoldChange': scalar_almost_equal
    }

    with open(f'out/migration/mass_listener_diffs_t{time}.txt', 'w') as f:
        f.write(array_diffs_report(array_from(actual_update),
                                   [expected_update[k]
                                       for k in actual_update.keys()],
                                   names=list(actual_update.keys())))

    tests = ComparisonTestSuite(test_structure, fail_loudly=False)
    tests.run_tests(actual_update,
                    expected_update,
                    verbose=False)

    #tests.dump_report()

    tests.fail()


if __name__ == "__main__":
    test_mass_listener()
