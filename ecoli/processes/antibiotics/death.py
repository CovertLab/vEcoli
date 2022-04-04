'''
===============================
Model of Conditional Cell Death
===============================

Fully-capitalized words and phrases have the meanings specified in
:rfc:`2119`.

This module holds machinery for modeling cell death that is conditioned
on the state of the cell. This machinery consists of *death detector
classes* and *death process classes*.

----------------------
Death Detector Classes
----------------------

Death detector classes encode a model (which may be configured upon
instantiation) for when a cell should die. These classes can be
instantiated to give death detectors, which are used by the death
processes we describe below.

Death detector classes MUST subclass :py:class:`DetectorInterface` and
implement :py:meth:`DetectorInterface.check_can_survive` as specified in
its documentation.

---------------------
Death Process Classes
---------------------

During a simulation, death is executed by a death :term:`process`, whose
model is declared in a death :term:`process class`. Each of these
process classes SHOULD use a death detector, as detailed above, to
determine when the cell should die. The mechanism by which the cell's
death is modeled depends on the form of death being modeled. For an
example, see :py:class:`DeathFreezeState`.
'''

import os

from vivarium.core.composition import (
    simulate_composite,
    PROCESS_OUT_DIR,
)
from vivarium.core.process import Process
from vivarium.core.composer import Composer
from vivarium.processes.injector import Injector
from vivarium.plots.simulation_output import plot_simulation_output
from vivarium.library.units import units

TOY_ANTIBIOTIC_THRESHOLD = 5.0 * units.mM
TOY_INJECTION_RATE = 2.0 * units.mM  # implicitly per second


class DetectorInterface:
    '''Interface that MUST be subclassed by all death detectors

    Each subclass SHOULD check for a condition that might kill the cell.

    For an example of a death detector class, see
    :py:class:`AntibioticDetector`.
    '''

    def __init__(self):
        self.needed_state_keys = {}

    def check_can_survive(self, states):
        '''Check whether the current state is survivable by the cell

        Each subclass MUST implement this method. The implementation
        SHOULD check whether the cell can survive given its current
        state.

        Arguments:
            states (dict): The states of each port in the cell, as a
                dictionary.

        Returns:
            bool: True if the cell can survive, False if it cannot.
        '''
        raise NotImplementedError(
            'Detector should implement check_can_survive')


class AntibioticDetector(DetectorInterface):

    def __init__(
        self, antibiotic_threshold=0.9, antibiotic_key='antibiotic'
    ):
        '''Death detector for antibiotics

        Checks whether the cell can survive the current internal
        antibiotic concentrations.

        Arguments:
            antibiotic_threshold (float): The maximum internal
                antibiotic concentration the cell can survive.
            antibiotic_key (str): The name of the variable storing
                the cell's internal antibiotic concentration.
        '''
        super().__init__()
        self.threshold = antibiotic_threshold
        self.key = antibiotic_key
        self.needed_state_keys.setdefault(
            'internal', set()).add(antibiotic_key)

    def check_can_survive(self, states):
        '''Checks if the current antibiotic concentration is survivable

        The internal antibiotic concentration MUST be stored in a
        variable of a port named ``internal``.

        Returns:
            bool: False if the antibiotic concentration is strictly
                greater than the the threshold. True otherwise.
        '''
        concentration = states['internal'][self.key]
        if concentration > self.threshold:
            return False
        return True


#: Map from detector class names to detector classes
DETECTOR_CLASSES = {
    'antibiotic': AntibioticDetector,
}


class DeathFreezeState(Process):

    name = 'death'
    defaults = {
        'detectors': {},
    }

    def __init__(self, initial_parameters=None):
        '''Model Death by Removing Processes

        This process class models death by, with a few exceptions,
        freezing the internal state of the cell. We implement this by
        removing from this process's :term:`compartment` all processes,
        specified with the ``targets`` configuration.

        Configuration:

        * **``detectors``**: A list of the names of the detector classes
          to include. Death will be triggered if any one of these
          triggers death. Names are specified in
          :py:const:`DETECTOR_CLASSES`.
        * **``targets``**: A list of the names of the processes
          that will be removed when the cell dies. The names are
          specified in the compartment's :term:`topology`.

        :term:`Ports`:

        * **``internal``**: The internal state of the cell.
        * **``global``**: Should be linked to the ``global``
          :term:`store`.
        * **``processes``**: Should be linked to the store that has
          the processes as children.
        '''
        super().__init__(initial_parameters)
        self.detectors = [
            DETECTOR_CLASSES[name](**config)
            for name, config in self.parameters['detectors'].items()
        ]
        # List of names of processes that will be removed upon death
        self.targets = initial_parameters.get('targets', [])


    def ports_schema(self):
        schema = {
            'global': {
                'dead': {
                    '_default': 0,
                    '_emit': True,
                    '_updater': 'set',
                },
            },
            'processes': {}
        }

        # detector ports
        for detector in self.detectors:
            needed_keys = detector.needed_state_keys
            for port, states in needed_keys.items():
                if port not in schema:
                    schema[port] = {}
                for state in states:
                    schema[port][state] = {'_default': 0 * units.mM}

        return schema

    def next_update(self, timestep, states):
        '''If any detector triggers death, kill the cell

        When we kill the cell, we convey this by setting the ``dead``
        variable in the ``global`` port to ``1`` instead of its default
        ``0``.
        '''
        for detector in self.detectors:
            if not detector.check_can_survive(states):
                # kill the cell
                update = {
                    'global': {
                        'dead': 1,
                    },
                    'processes': {
                        '_delete': self.targets,
                    },
                }
                return update
        return {}


class ToyDeath(Composer):

    def generate_processes(self, config):
        death_parameters = {
            'detectors': {
                'antibiotic': {
                    'antibiotic_threshold': TOY_ANTIBIOTIC_THRESHOLD,
                }
            },
            'targets': ['injector', 'death'],
        }
        death_process = DeathFreezeState(death_parameters)
        injector_parameters = {
            'substrate_rate_map': {
                'antibiotic': TOY_INJECTION_RATE,
            },
        }
        injector_process = Injector(injector_parameters)
        enduring_parameters = {
            'substrate_rate_map': {
                'enduring_antibiotic': TOY_INJECTION_RATE,
            },
        }
        enduring_process = Injector(enduring_parameters)

        return {
            'death': death_process,
            'injector': injector_process,
            'enduring_injector': enduring_process,
        }

    def generate_topology(self, config):
        return {
            'death': {
                'internal': ('cell',),
                'global': ('global',),
                'processes': tuple(),
            },
            'injector': {
                'internal': ('cell',),
            },
            'enduring_injector': {
                'internal': ('cell',),
            },
        }


def test_death_freeze_state(end_time=10, asserts=True):
    toy_death_compartment = ToyDeath({}).generate()

    init_state = {
        'cell': {
            'antibiotic': 0.0 * units.mM,
            'enduring_antibiotic': 0.0 * units.mM},
        'global': {
            'dead': 0}}

    settings = {
        'total_time': end_time,
        'initial_state': init_state}
    saved_states = simulate_composite(
        toy_death_compartment, settings)

    if asserts:
        # Add 1 because dies when antibiotic strictly above threshold
        expected_death = 1 + (
            TOY_ANTIBIOTIC_THRESHOLD // TOY_INJECTION_RATE)
        expected_saved_states = {
            'cell': {
                ('antibiotic', 'millimolar'): [],
                ('enduring_antibiotic', 'millimolar'): [],
            },
            'global': {
                'dead': [],
            },
            'time': [],
        }
        for time in range(end_time + 1):
            expected_saved_states['cell'][
                ('antibiotic', 'millimolar')
            ].append(
                (time * TOY_INJECTION_RATE).magnitude
                if time <= expected_death
                # Add one because death will only be detected
                # the iteration after antibiotic above
                # threshold. This happens because death and
                # injector run "concurrently" in the composite,
                # so their updates are applied after both have
                # finished.
                else (
                    (expected_death + 1) * TOY_INJECTION_RATE
                ).magnitude
            )
            expected_saved_states['cell'][
                ('enduring_antibiotic', 'millimolar')
            ].append(
                (time * TOY_INJECTION_RATE).magnitude)
            expected_saved_states['global']['dead'].append(
                0 if time <= expected_death else 1)
            expected_saved_states['time'].append(float(time))

        assert expected_saved_states == saved_states

    return saved_states


def plot_death_freeze_state_test():
    out_dir = os.path.join(PROCESS_OUT_DIR, 'death_freeze_state')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    timeseries = test_death_freeze_state(asserts=False)
    plot_settings = {}
    plot_simulation_output(timeseries, plot_settings, out_dir)


if __name__ == '__main__':
    plot_death_freeze_state_test()
