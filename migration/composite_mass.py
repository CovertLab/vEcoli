"""
tests that ecoli_master composite matches wcEcoli states at select time steps
"""

import json
import numpy as np
from scipy.stats import linregress
from matplotlib import pyplot as plt

from migration.migration_utils import ComparisonTestSuite
from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH


RELATIVE_TOLERANCES = {
    'mRnaMass': 0.05,
    'rRnaMass': 1e-4,
    'dry_mass': 0.01,
    'rnaMass': 0.001,
    'water_mass': 0.01,
    'smallMoleculeMass': 0.01,
    'proteinMass': 1e-3,
    'cell_mass': 0.005,
    'dnaMass': 2e-15,
    'tRnaMass': 1e-3,
}


def test_composite_mass(total_time=30):
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + "default.json")
    sim.total_time = total_time
    sim.build_ecoli()

    # run the composite and save specified states
    sim.run()

    timeseries = sim.ecoli_experiment.emitter.get_timeseries([
        ('agents', '0', 'listeners', 'mass')])

    actual_timeseries = timeseries['agents']['0']['listeners']['mass']
    # Skip data from t=0
    wcecoli_timeseries = {}
    for submass in actual_timeseries.keys():
        actual_timeseries[submass] = actual_timeseries[submass][1:]
        wcecoli_timeseries[submass] = np.zeros(len(actual_timeseries[submass]))

    vivarium_keys = set(actual_timeseries.keys())
    wcecoli_keys = set()
    with open(f"data/migration/mass_data.json") as f:
        wc_mass_data = json.load(f)
    for index, time in enumerate(timeseries['time'][:-1]):
        actual_update = {
            submass: data[index]
            for submass, data in actual_timeseries.items()
        }
        wc_update = wc_mass_data[str(int(time))]['mass']
        for key, data in wc_update.items():
            wcecoli_timeseries[key][index] = data
        wcecoli_keys = set(wc_update.keys())
        print(time)
        assertions(actual_update, wc_update, wcecoli_keys & vivarium_keys)
    only_wcecoli = wcecoli_keys - vivarium_keys
    print('These keys only exist in the wcEcoli mass listener: ' + str(list(only_wcecoli)))
    only_vivarium = vivarium_keys - wcecoli_keys
    print('These keys only exist in the vivarium mass listener: ' + str(list(only_vivarium)))
    plots(actual_timeseries, wcecoli_timeseries, wcecoli_keys & vivarium_keys)


def _make_assert(key):
    def custom_assert(a, b):
        rtol = RELATIVE_TOLERANCES.get(key, 0.1)
        # rtol = 0
        if np.isnan(a) and np.isnan(b):
            return True
        close = np.isclose(a, b, rtol=rtol)
        if not close:
            rdiff = (np.absolute(a-b) / b).max()
            print(f'Failure for {key}: rdiff {rdiff} > rtol {rtol}')
        return close
    return custom_assert


def assertions(actual_update, expected_update, keys):
    test_structure = {
        key : _make_assert(key)
        for key in keys
    }

    tests = ComparisonTestSuite(test_structure, fail_loudly=False)
    tests.run_tests(actual_update, expected_update, verbose=False)

    tests.fail()

def plots(actual_timeseries, wcecoli_timeseries, keys):
    # Pytest gives warnings about overlapping axes without this
    plt.close('all')
    n_keys = len(keys)
    rows = int(np.ceil(n_keys/3))
    for index, key in enumerate(keys):
        plt.subplot(rows, 3, index+1)
        plt.scatter(wcecoli_timeseries[key], actual_timeseries[key])
        if not np.all(wcecoli_timeseries[key] == wcecoli_timeseries[key][0]):
            slope, intercept, r_value, p_value, std_err = linregress(
                wcecoli_timeseries[key], actual_timeseries[key])
            # assert r_value >= 0.90, (
            #     f"Correlation for {key} = {r_value} <= 0.90")
            best_fit = np.poly1d([slope, intercept])
            plt.plot(wcecoli_timeseries[key], best_fit(wcecoli_timeseries[key]),
                    'b-', label=f'r = {r_value}')
        plt.plot(wcecoli_timeseries[key], wcecoli_timeseries[key], 'r-',
                 label='Parity')
        plt.title(str(key))
        plt.legend()
        plt.xlabel('wcEcoli')
        plt.ylabel('Vivarium')
    plt.gcf().set_size_inches(16, 16)
    plt.tight_layout()
    plt.savefig(f"out/migration/composite_mass.png")
    plt.close()

if __name__ == "__main__":
    test_composite_mass()
