from datetime import datetime
import json
import os
import unittest

import numpy as np
import unum
from vivarium.core.engine import Engine

from ecoli.composites.ecoli_configs import CONFIG_DIR_PATH
from ecoli.experiments.ecoli_master_sim import EcoliSim


PARALLELIZED_PROCESSES = (
    'ecoli-transcript-elongation',
    'ecoli-transcript-initiation',
    #'ecoli-chromosome-structure',
    #'ecoli-metabolism',
    #'ecoli-tf-binding',
    'ecoli-rna-degradation',
    'ecoli-polypeptide-initiation',
    'ecoli-polypeptide-elongation',
    #'ecoli-complexation',
    #'ecoli-two-component-system',
    #'ecoli-equilibrium',
    'ecoli-protein-degradation',
    #'ecoli-chromosome-replication',
    #'ecoli-mass-listener',
    #'mRNA_counts_listener'
)
TIMESTEP = 2
TOTAL_TIME = 10


def assert_equal(data1, data2, **kwargs):
    if data1 == data2:
        return
    raise AssertionError(
        f'Values unequal: {data1} != {data2}\n'
        f'Metadata: {kwargs}'
    )


def assert_random_state_equal(rs1, rs2, **kwargs):
    type1, keys1, pos1, has_gauss1, gauss1 = rs1.get_state()
    type2, keys2, pos2, has_gauss2, gauss2 = rs2.get_state()

    if type1 != type2:
        raise AssertionError(
            f'BitGenerator types unequal: {type1} != {type2}\n'
            f'Metadata: {kwargs}'
        )
    np.testing.assert_array_equal(
        keys1, keys2, f'RandomState keys unequal. Meta: {kwargs}')
    if pos1 != pos2:
        raise AssertionError(
            f'pos values unequal: {pos1} != {pos2}\n'
            f'Metadata: {kwargs}'
        )
    if has_gauss1 != has_gauss2:
        raise AssertionError(
            f'has_gauss values unequal: {has_gauss1} != {has_gauss2}\n'
            f'Metadata: {kwargs}'
        )
    if gauss1 != gauss2:
        raise AssertionError(
            f'cached_gaussian values unequal: {gauss1} != {gauss2}\n'
            f'Metadata: {kwargs}'
        )


def assert_dict_equal(data1, data2, path=tuple()):
    assert_equal(type(data1), type(data2), path=path)
    if isinstance(data1, np.ndarray):
        np.testing.assert_array_equal(
            data1, data2, f'Path: {path}')
    elif isinstance(data1, unum.Unum):
        assert_equal(data1.strUnit(), data2.strUnit(), path=path)
        assert_dict_equal(data1.asNumber(), data2.asNumber(), path=path)
    elif str(type(data1)) == "<class 'method'>":
        assert_dict_equal(data1.__func__, data2.__func__, path=path)
    elif str(type(data1)) == "<class 'function'>":
        assert_equal(data1.__qualname__, data2.__qualname__, path=path)
    elif isinstance(data1, np.random.RandomState):
        assert_random_state_equal(data1, data2, path=path)
    elif isinstance(data1, dict):
        assert_equal(data1.keys(), data2.keys(), path=path)
        for key in data1.keys():
            assert_dict_equal(
                data1[key], data2[key], path=path + (key,))
    else:
        assert_equal(data1, data2, path=path)


class TestableEcoliSim(EcoliSim):

    def __init__(self, parallel_processes=tuple()):
        path = os.path.join(CONFIG_DIR_PATH, 'default.json')
        with open(path, 'r') as f:
            config = json.load(f)
        config['save_times'] = []
        config['initial_state'] = 'wcecoli_t3142'
        for process in parallel_processes:
            proc_config = config['process_configs'].setdefault(
                process, {})
            proc_config['_parallel'] = True
        self.ecoli_experiment = None
        super().__init__(config)

    def prepare(self):
        self.build_ecoli()

        experiment_config = {
            'description': self.description,
            'processes': self.ecoli.processes,
            'topology': self.ecoli.topology,
            'initial_state': self.initial_state,
            'progress_bar': self.progress_bar,
            'emit_topology': self.emit_topology,
            'emit_processes': self.emit_processes,
            'emit_config': self.emit_config,
            'emitter': self.emitter,
        }
        if self.experiment_id:
            experiment_config['experiment_id'] = self.experiment_id
            if self.suffix_time:
                experiment_config['experiment_id'] += datetime.now().strftime(
                    "_%d/%m/%Y %H:%M:%S")

        self.ecoli_experiment = Engine(**experiment_config)

    def step(self, timestep=2):
        self.ecoli_experiment.update(timestep)

    def get_state(self):
        return self.ecoli_experiment.state.emit_data()

    def end(self):
        self.ecoli_experiment.end()


class EcoliParallelizationTests(unittest.TestCase):

    def _assertDataEqual(self, data1, data2, path=tuple(), **kwargs):
        try:
            self.assertEqual(type(data1), type(data2))
        except AssertionError as e:
            kwargs['path'] = path
            print('Metadata:', kwargs)
            raise e

        if isinstance(data1, np.ndarray):
            try:
                np.testing.assert_array_equal(data1, data2)
            except AssertionError as e:
                kwargs['path'] = path
                print('Metadata:', kwargs)
                raise e
        elif isinstance(data1, dict):
            try:
                self.assertEqual(data1.keys(), data2.keys())
            except AssertionError as e:
                kwargs['path'] = path
                print('Metadata:', kwargs)
                raise e

            for key in data1.keys():
                self._assertDataEqual(
                    data1[key], data2[key], path + (key,), **kwargs)
        else:
            try:
                self.assertEqual(data1, data2)
            except AssertionError as e:
                kwargs['path'] = path
                print('Metadata:', kwargs)
                raise e

    def test_parallelization(self):
        parallel_sim = TestableEcoliSim(PARALLELIZED_PROCESSES)
        non_parallel_sim = TestableEcoliSim()

        parallel_sim.prepare()
        non_parallel_sim.prepare()

        for time in range(0, TOTAL_TIME, TIMESTEP):
            parallel_sim.step(TIMESTEP)
            non_parallel_sim.step(TIMESTEP)

            parallel_state = parallel_sim.get_state()
            non_parallel_state = non_parallel_sim.get_state()

            self._assertDataEqual(
                parallel_state,
                non_parallel_state,
                time=time,
            )

        parallel_sim.end()
        non_parallel_sim.end()


def main():
    sim = TestableEcoliSim(PARALLELIZED_PROCESSES)
    sim.prepare()


if __name__ == '__main__':
    # Added for PDB. See https://stackoverflow.com/a/60922965.
    __spec__ = None
    main()
