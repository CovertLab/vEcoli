import copy

import numpy as np
from scipy.constants import N_A
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from vivarium.core.process import Process
from vivarium.library.units import units, Quantity

from ecoli.library.parameters import param_store
from ecoli.library.units import add_units, remove_units
from ecoli.processes.antibiotics.antibiotic_transport_steady_state import (
    SPECIES,
    SPECIES_TO_INDEX,
    EXTERNAL_SPECIES,
    REACTIONS,
    REACTIONS_TO_INDEX,
    STOICH,
    UNITS,
    species_derivatives,
    species_array_to_dict,
    species_dict_to_array,
)


AVOGADRO = N_A / units.mol


def update_from_odeint(
        initial_state, reaction_params, internal_bias, timestep):
    '''Compute an update given a steady-state solution.

    Beginning from the initial state, numerically integrates the
    species ODEs to find the final state.

    Args:
        initial_state: Initial state dictionary.
        reaction_params: Dictionary of reaction parameters.
        internal_bias: See :py:func:`species_derivatives`.
        timestep: Timestep for update.

    Returns:
        Update dictionary.
    '''
    assert set(SPECIES) == initial_state.keys()
    initial_state_arr = species_dict_to_array(
        initial_state, SPECIES_TO_INDEX)

    args = (
        reaction_params,
        internal_bias,
    )
    result = solve_ivp(
        lambda t, state_arr, *args: species_derivatives(
            state_arr, *args),
        [0, timestep],
        initial_state_arr,
        args=args,
    )
    assert result.success
    final_state_arr = result.y[:, -1].T

    delta_arr = final_state_arr - initial_state_arr
    delta = species_array_to_dict(delta_arr, SPECIES_TO_INDEX)
    return {
        'species': delta,
        'delta_species': delta,
    }


class AntibioticTransportOdeint(Process):
    name = 'antibiotic-transport-odeint'
    defaults = {
        'initial_reaction_parameters': {},
        'diffusion_only': False,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.antibiotics = self.parameters[
            'initial_reaction_parameters'].keys()

    def initial_state(self, config=None):
        state = {
            antibiotic: {
                'delta_species': {
                    species: 0 * units.mM
                    for species in SPECIES
                    if species not in EXTERNAL_SPECIES
                },
                'reaction_parameters': self.parameters[
                    'initial_reaction_parameters'][antibiotic]
            }
            for antibiotic in self.antibiotics
        }
        return state

    def ports_schema(self):
        schema = {
            antibiotic: {
                'species': {
                    species: {
                        '_default': 0.0 * units.mM,
                        '_updater': 'accumulate',
                        '_emit': True,
                        '_divider': 'set',
                    }
                    for species in SPECIES
                    if species not in EXTERNAL_SPECIES
                },
                'delta_species': {
                    species: {
                        '_default': 0.0 * units.mM,
                        '_updater': 'set',
                        '_emit': True,
                    }
                    for species in SPECIES
                    if species not in EXTERNAL_SPECIES
                },
                'exchanges': {
                    species: {
                        '_default': 0,
                        '_emit': True,
                    }
                    for species in EXTERNAL_SPECIES
                    if not species.endswith('_delta')
                },
                'external': {
                    species: {
                        '_default': 0 * units.mM,
                        '_emit': True,
                    }
                    for species in EXTERNAL_SPECIES
                    if not species.endswith('_delta')
                },
                'reaction_parameters': {
                    reaction: {
                        parameter: {
                            '_default': 0 * unit,
                            '_emit': True,
                        }
                        for parameter, unit in reaction_params.items()
                    }
                    for reaction, reaction_params in self.parameters[
                        'initial_reaction_parameters'][antibiotic].items()
                },
            }
            for antibiotic in self.antibiotics
        }

        return schema

    def next_update(self, timestep, state):
        update = {}
        for antibiotic in self.antibiotics:
            antibiotic_state = state[antibiotic]
            # Prepare the state for the bioscrape process by moving
            # species from 'external' to 'species' where the bioscrape
            # process expects them to be, renaming variables, and doing
            # unit conversions.
            prepared_state = {
                'species': copy.deepcopy(antibiotic_state['species']),
                'external': copy.deepcopy(antibiotic_state['external']),
                'reaction_parameters': copy.deepcopy(
                    antibiotic_state['reaction_parameters']),
            }
            for species in EXTERNAL_SPECIES:
                if species.endswith('_delta'):
                    continue
                assert species not in prepared_state['species']
                prepared_state['species'][species] = prepared_state['external'].pop(species)
                # The delta species is just for tracking updates to the
                # external environment for later conversion to exchanges. We
                # assume that `species` is not updated. Note that the
                # `*_delta` convention must be obeyed for this to work
                # correctly.
                delta_species = f'{species}_delta'
                assert delta_species not in species
                prepared_state['species'][delta_species] = 0 * units.mM

            prepared_state, saved_units = remove_units(
                prepared_state, UNITS)

            # Compute the update.
            charge = prepared_state['reaction_parameters']['diffusion']['charge']
            faraday = param_store.get(('faraday_constant',)).to(
                units.C / units.mol)
            potential = param_store.get(('donnan_potential',)).to(units.V)
            gas_constant = param_store.get(('gas_constant',)).to(
                units.J / units.mol / units.K)
            temperature = param_store.get(('temperature',)).to(units.K)
            # Biases diffusion to favor higher internal concentrations
            # according to the Goldman-Hodgkin-Katz flux equation assuming
            # the outer membrane has a potential from the Donnan equilibrium.
            internal_bias = charge * faraday * potential / gas_constant / temperature

            if self.parameters['diffusion_only']:
                prepared_state['reaction_parameters']['export'][
                    'kcat'] = 0 / units.sec
                prepared_state['reaction_parameters']['hydrolysis'][
                    'kcat'] = 0 / units.sec
            else:
                antibiotic_update = update_from_odeint(
                    prepared_state['species'],
                    prepared_state['reaction_parameters'],
                    internal_bias,
                    timestep,
                )

            # Make sure there are no NANs in the update.
            assert not np.any(np.isnan(list(antibiotic_update['species'].values())))

            # Add units back in
            antibiotic_update['species'] = add_units(
                antibiotic_update['species'],
                saved_units['species'],
                strict=not self.parameters['diffusion_only'])
            antibiotic_update['delta_species'] = add_units(
                antibiotic_update['delta_species'],
                saved_units['species'],
                strict=not self.parameters['diffusion_only'])

            # Postprocess the update to convert changes to external
            # molecules into exchanges.
            antibiotic_update.setdefault('exchanges', {})
            for species in EXTERNAL_SPECIES:
                if species.endswith('_delta'):
                    continue
                delta_species = f'{species}_delta'
                if species in antibiotic_update['species']:
                    assert antibiotic_update['species'][species] == 0
                    del antibiotic_update['species'][species]
                if species in antibiotic_update['delta_species']:
                    assert antibiotic_update['delta_species'][species] == 0
                    del antibiotic_update['delta_species'][species]
                if delta_species in antibiotic_update['species']:
                    exchange = (
                        antibiotic_update['species'][delta_species]
                        * antibiotic_state['reaction_parameters']['diffusion']['volume']
                        * AVOGADRO)
                    assert species not in antibiotic_update['exchanges']
                    # Exchanges flowing out of the cell are positive.
                    antibiotic_update['exchanges'][species] = exchange.to(
                        units.counts).magnitude
                    del antibiotic_update['species'][delta_species]
                    del antibiotic_update['delta_species'][delta_species]

            update[antibiotic] = antibiotic_update
        return update

def _dummy_derivative(_, y):
    e, i, h = y

    # Note that we assume constant e
    diffusion = 2 * 3 * (3 - i) / 2
    export = 4 * 1 * i / (2 + i)
    hydrolysis = 4 * 0.5 * i / (2 + i)

    dedt = export - diffusion
    didt = diffusion - export - hydrolysis
    dhdt = hydrolysis

    return dedt, didt, dhdt


def test_antibiotic_transport_odeint():
    proc = AntibioticTransportOdeint({
        'initial_reaction_parameters': {
            'antibiotic': {},
        },
    })
    initial_state = {
        'antibiotic': {
            'external': {
                'external': 3 * units.mM,
            },
            'species': {
                'internal': 0 * units.mM,
                'hydrolyzed': 0 * units.mM,
            },
            'reaction_parameters': {
                'diffusion': {
                    'permeability': 2 * units.dm / units.sec,
                    'area': 3 * units.dm**2,
                    'volume': 2 * units.L,
                    'charge': 0 * units.count,
                },
                'export': {
                    'kcat': 4 / units.sec,
                    'km': 2 * units.mM,
                    'enzyme_conc': 1 * units.mM,
                    'n': 1 * units.count,
                },
                'hydrolysis': {
                    'kcat': 4 / units.sec,
                    'km': 2 * units.mM,
                    'enzyme_conc': 0.5 * units.mM,
                    'n': 1 * units.count,
                },
            },
        },
    }
    update = proc.next_update(1, initial_state)['antibiotic']

    # Compute expected update
    initial_arr = np.array([3, 0, 0])
    result = solve_ivp(_dummy_derivative, (0, 1), initial_arr)
    assert result.success
    final_arr = result.y[:,-1]
    delta_arr = final_arr - initial_arr

    expected_update = {
        'species': {
            'internal': delta_arr[1] * units.mM,
            'hydrolyzed': delta_arr[2] * units.mM,
        },
        'delta_species': {
            'internal': delta_arr[1] * units.mM,
            'hydrolyzed': delta_arr[2] * units.mM,
        },
        'exchanges': {
            # Exchanges are in units of counts, but the species are in
            # units of mM with a volume of 1L.
            'external': (
                delta_arr[0] * units.mM
                * initial_state['antibiotic'][
                    'reaction_parameters']['diffusion']['volume']
                * AVOGADRO).to(units.count).magnitude,
        },
    }
    assert update.keys() == expected_update.keys()
    for key in expected_update:
        if key in ('species', 'delta_species', 'exchanges'):
            assert update[key].keys() == expected_update[key].keys()
            for species in expected_update[key]:
                assert abs(
                    update[key][species] - expected_update[key][species]
                ) <= 1e-4 * abs(expected_update[key][species])
        else:
            assert update[key] == expected_update[key]
