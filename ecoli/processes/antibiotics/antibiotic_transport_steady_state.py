import copy

import numpy as np
from scipy.constants import N_A
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from vivarium.core.process import Process
from vivarium.library.units import units, Quantity

from ecoli.parameters import param_store


AVOGADRO = N_A / units.mol

# NOTE: These must match the math in species_derivatives().
SPECIES = ('internal', 'external', 'external_delta', 'hydrolyzed')
SPECIES_TO_INDEX = {
    species: i
    for i, species in enumerate(SPECIES)
}
EXTERNAL_SPECIES = ('external', 'external_delta')
REACTIONS = {
    'diffusion': (('external_delta',), ('internal',)),  # Diffusion
    'export': (('internal',), ('external_delta',)),  # Efflux
    'hydrolysis': (('internal',), ('hydrolyzed',)),  # Hydrolysis
}
REACTIONS_TO_INDEX = {
    reaction: i
    for i, reaction in enumerate(REACTIONS)
}
# Build a stoichiometry matrix.
#
# For example, suppose we have the following reactions:

# .. code-block:: text

#     (rxn 1) A  -> B
#     (rxn 2) 2B -> A

# Then we can write a stoichiometry matrix **N**:

# .. code-block:: text

#          rxn 1    rxn 2
#         +-           -+
#       A | -1       +1 |
#       B | +1       -2 |
#         +-           -+

# Now if we have a derivatives vector **d** of the rates of change of
# each species, we can solve for a rate vector **v** representing the
# rates of each reaction needed to produce the rates of change in **d**:
# **d** = **N** · **v**.
STOICH = np.zeros((len(SPECIES), len(REACTIONS)))
for i_reaction, (reactants, products) in enumerate(REACTIONS.values()):
    for reactant in reactants:
        i_reactant = SPECIES_TO_INDEX[reactant]
        STOICH[i_reactant, i_reaction] -= 1
    for product in products:
        i_product = SPECIES_TO_INDEX[product]
        STOICH[i_product, i_reaction] += 1
UNITS = {
    'species': {
        species: units.mM
        for species in SPECIES
    },
    'reaction_parameters': {
        'diffusion': {
            'permeability': units.dm / units.sec,
            'area': units.dm**2,
            'volume': units.L,
            'charge': units.count,
        },
        'export': {
            'kcat': 1 / units.sec,
            'km': units.mM,
            'enzyme_conc': units.mM,
            'n': units.count,
        },
        'hydrolysis': {
            'kcat': 1 / units.sec,
            'km': units.mM,
            'enzyme_conc': units.mM,
            'n': units.count,
        },
    },
}


def species_derivatives(state_arr, reaction_params, internal_bias):
    # Parse state
    state = species_array_to_dict(state_arr, SPECIES_TO_INDEX)
    internal = state['internal']
    external = state['external']

    # Parse reaction parameters
    area = reaction_params['diffusion']['area']
    permeability = reaction_params['diffusion']['permeability']
    volume_p = reaction_params['diffusion']['volume']

    # TODO: Pull the Michaelis-Menten logic into a separate function for
    # reuse.
    kcat_export = reaction_params['export']['kcat']
    km_export = reaction_params['export']['km']
    pump_conc = reaction_params['export']['enzyme_conc']
    n_export = reaction_params['export']['n']
    kcat_hydrolysis = reaction_params['hydrolysis']['kcat']
    km_hydrolysis = reaction_params['hydrolysis']['km']
    hydrolase_conc = reaction_params['hydrolysis']['enzyme_conc']
    n_hydrolysis = reaction_params['hydrolysis']['n']

    diffusion_rate = area * permeability * (
        external * internal_bias - internal) / (volume_p)
    export_rate = (
        kcat_export * pump_conc * internal**n_export
    ) / (
        km_export + internal**n_export)
    hydrolysis_rate = (
        kcat_hydrolysis * hydrolase_conc * internal**n_hydrolysis
    ) / (
        km_hydrolysis + internal**n_hydrolysis)

    reaction_rates = {
        'diffusion': diffusion_rate,
        'export': export_rate,
        'hydrolysis': hydrolysis_rate,
    }
    reaction_rates_arr = species_dict_to_array(
        reaction_rates, REACTIONS_TO_INDEX)

    return STOICH @ reaction_rates_arr


def internal_derivative(internal, external, reaction_params,
        internal_bias):
    state = {
        'internal': internal,
        'external': external,
        'external_delta': 0,
        'hydrolyzed': 0,
    }
    state_arr = species_dict_to_array(state, SPECIES_TO_INDEX)
    derivatives = species_derivatives(
        state_arr, reaction_params, internal_bias)
    return derivatives[SPECIES_TO_INDEX['internal']]


def find_steady_state(external, reaction_params, internal_bias):
    args = (
        external,
        reaction_params,
        internal_bias
    )
    result = root_scalar(
        internal_derivative,
        args=args,
        bracket=[0, external * internal_bias],
    )
    assert result.converged
    return result.root


def update_from_steady_state(
        internal_steady_state, initial_state, reaction_params,
        internal_bias, timestep):
    assert set(SPECIES) == initial_state.keys()
    steady_state = initial_state.copy()
    # Assume that steady state is reached exclusively through diffusion
    steady_state['internal'] = internal_steady_state
    steady_state['external_delta'] = -(
        internal_steady_state - initial_state['internal'])
    steady_state_arr = species_dict_to_array(
        steady_state, SPECIES_TO_INDEX)

    args = (
        reaction_params,
        internal_bias,
    )
    result = solve_ivp(
        lambda t, state_arr, *args: species_derivatives(
            state_arr, *args),
        [0, timestep],
        steady_state_arr,
        args=args,
    )
    assert result.success
    final_state_arr = result.y[:, -1].T

    initial_state_arr = species_dict_to_array(
        initial_state, SPECIES_TO_INDEX)
    delta_arr = final_state_arr - initial_state_arr
    delta = species_array_to_dict(delta_arr, SPECIES_TO_INDEX)
    return {
        'species': delta,
        'delta_species': delta,
    }


def species_array_to_dict(array, species_to_index):
    '''Convert an array of values to a map from name to value index.

    >>> array = [1, 5, 2]
    >>> species_to_index = {
    ...     'C': 2,
    ...     'A': 0,
    ...     'B': 1,
    ... }
    >>> species_array_to_dict(array, species_to_index)
    {'C': 2, 'A': 1, 'B': 5}

    Args:
        array: The array of values. Must be subscriptable with indices.
        species_to_index: Dictionary mapping from names to the index in
            ``array`` of the value associated with each name. The values
            of the dictionary must be exactly the indices of ``array``
            with no duplicates.

    Returns:
        A dictionary mapping from names (the keys of
        ``species_to_index``) to the associated values in ``array``.
    '''
    return {
        species: array[index]
        for species, index in species_to_index.items()
    }


def species_dict_to_array(species_dict, species_to_index):
    '''Convert species dict to an array based on an index map.

    >>> d = {'C': 2, 'A': 1, 'B': 5}
    >>> species_to_index = {
    ...     'C': 2,
    ...     'A': 0,
    ...     'B': 1,
    ... }
    >>> species_dict_to_array(d, species_to_index)
    array([1., 5., 2.])

    Args:
        species_dict: The dictionary mapping species names to values.
        species_to_index: Dictionary mapping from names to the index in
            ``array`` of the value associated with each name.

    Returns:
        An array with every value from ``species_dict`` at the index
        specified by ``species_to_index``.

    Raises:
        AssertionError: When the keys of ``species_dict`` and the keys
            of ``species_to_index`` differ.
    '''
    assert species_dict.keys() == species_to_index.keys()
    array = np.zeros(len(species_dict))
    for species, value in species_dict.items():
        i = species_to_index[species]
        array[i] = value
    return array


class AntibioticTransportSteadyState(Process):
    name = 'antibiotic-transport-steady-state'
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

    # TODO: Make this a general utility function.
    def _remove_units(self, state, units_map=None):
        units_map = units_map or {}
        converted_state = {}
        saved_units = {}
        for key, value in state.items():
            if isinstance(value, dict):
                value_no_units, new_saved_units = self._remove_units(
                    value, units_map=units_map.get(key)
                )
                converted_state[key] = value_no_units
                saved_units[key] = new_saved_units
            elif isinstance(value, Quantity):
                saved_units[key] = value.units
                expected_units = units_map.get(key)
                if expected_units:
                    value_no_units = value.to(expected_units).magnitude
                else:
                    value_no_units = value.magnitude
                converted_state[key] = value_no_units
            else:
                assert not units_map.get(key), f'{key} does not have units'
                converted_state[key] = value

        return converted_state, saved_units

    # TODO: Make this a general utility function.
    def _add_units(self, state, saved_units, strict=True):
        """add units back in"""
        unit_state = state.copy()
        for key, value in saved_units.items():
            if key not in unit_state and not strict:
                continue
            before = unit_state[key]
            if isinstance(value, dict):
                unit_state[key] = self._add_units(before, value)
            else:
                unit_state[key] = before * value
        return unit_state

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

            prepared_state, saved_units = self._remove_units(
                prepared_state, units_map=UNITS)

            # Compute the update.
            charge = prepared_state['reaction_parameters']['diffusion']['charge']
            faraday = param_store.get(('faraday_constant',)).to(
                units.C / units.mol)
            potential = param_store.get(('membrane_potential',)).to(units.V)
            gas_constant = param_store.get(('gas_constant',)).to(
                units.J / units.mol / units.K)
            temperature = param_store.get(('temperature',)).to(units.K)
            internal_bias = np.exp(
                charge * faraday * potential / gas_constant / temperature)

            internal_steady_state = find_steady_state(
                prepared_state['species']['external'],
                prepared_state['reaction_parameters'],
                internal_bias,
            )
            if self.parameters['diffusion_only']:
                delta = (
                    internal_steady_state
                    - prepared_state['species']['internal'])
                antibiotic_update = {
                    'species': {
                        'internal': delta,
                    },
                    'delta_species': {
                        'internal': delta,
                    },
                }
            else:
                antibiotic_update = update_from_steady_state(
                    internal_steady_state,
                    prepared_state['species'],
                    prepared_state['reaction_parameters'],
                    internal_bias,
                    timestep,
                )

            # Make sure there are no NANs in the update.
            assert not np.any(np.isnan(list(antibiotic_update['species'].values())))

            # Add units back in
            antibiotic_update['species'] = self._add_units(
                antibiotic_update['species'],
                saved_units['species'],
                strict=not self.parameters['diffusion_only'])
            antibiotic_update['delta_species'] = self._add_units(
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


def test_antibiotic_transport_steady_state():
    proc = AntibioticTransportSteadyState({
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

    expected_update = {
        'species': {
            'internal': 2 * units.mM,
            'hydrolyzed': 1 * units.mM,
        },
        'delta_species': {
            'internal': 2 * units.mM,
            'hydrolyzed': 1 * units.mM,
        },
        'exchanges': {
            # Exchanges are in units of counts, but the species are in
            # units of mM with a volume of 1L.
            'external': (
                -3 * units.mM
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
                ) <= 1e-5 * abs(expected_update[key][species])
        else:
            assert update[key] == expected_update[key]


if __name__ == '__main__':
    test_exchange_aware_bioscrape()
