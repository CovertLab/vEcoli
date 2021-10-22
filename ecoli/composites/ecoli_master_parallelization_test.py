from datetime import datetime
import json
import os
import unittest

import numpy as np
from vivarium.core.engine import Engine

from ecoli.composites.ecoli_configs import CONFIG_DIR_PATH
from ecoli.experiments.ecoli_master_sim import EcoliSim


PARALLELIZED_PROCESSES = (
    'ecoli-transcript-elongation',
)
PARALLELIZED_PROCESSES = tuple()
TIMESTEP = 2
TOTAL_TIME = 10


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
        with self.subTest(path=path, **kwargs):
            self.assertEqual(type(data1), type(data2))
        if isinstance(data1, np.ndarray):
            with self.subTest(path=path, **kwargs):
                np.testing.assert_array_equal(data1, data2)
        elif isinstance(data1, dict):
            with self.subTest(path=path, **kwargs):
                self.assertEqual(data1.keys(), data2.keys())
            for key in data1.keys():
                self._assertDataEqual(
                    data1[key], data2[key], path + (key,), **kwargs)
        else:
            with self.subTest(path=path, **kwargs):
                self.assertEqual(data1, data2)

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
