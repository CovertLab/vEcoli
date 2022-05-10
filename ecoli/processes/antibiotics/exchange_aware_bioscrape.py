'''A Bioscrape process that correctly handles environmental exchange.

When we need to move molecules from an environment into a cell, we do so
via "exchange", a count of the number of molecules to move. By
convention, exchange is positive for efflux from the cell. This approach
means that processes inside the cell do not have to know the volume of
the environment (or, more commonly, the volume of the environment bin
containing the cell).

The Vivarium Bioscrape process does not work with exchanges.
:py:class:`ExchangeAwareBioscrape` is a wrapper around Vivarium
Bioscrape. It largely works just like Vivarium Bioscrape, except that it
handles exchanges as follows:

* External molecules' concentrations are read from the ``external``
  port.
* External molecules' updates are sent to the ``exchanges`` port (after
  appropriate unit conversion).
'''

import copy
import tempfile

from biocrnpyler import (
    ChemicalReactionNetwork,
    Reaction,
    Species,
)
import numpy as np
from vivarium.library.units import units, Quantity
from vivarium_bioscrape.processes.bioscrape import Bioscrape


def _transform_dict_leaves(root, transform_func):
    if not isinstance(root, dict):
        # We are at a leaf node, so apply transformation function.
        return transform_func(root)
    # We are not at a leaf node, so recurse.
    transformed = {
        key: _transform_dict_leaves(value, transform_func)
        for key, value in root.items()
    }
    return transformed


def test_transform_dict_leaves():
    root = {
        'a': {
            'b': 1,
        },
        'c': 2,
    }
    transformed = _transform_dict_leaves(root, lambda x: x + 1)
    expected = {
        'a': {
            'b': 2,
        },
        'c': 3,
    }
    assert transformed == expected


class ExchangeAwareBioscrape(Bioscrape):
    name = 'exchange-aware-bioscrape'
    defaults = {
        **Bioscrape.defaults,
        'external_species': tuple(),
        'name_map': tuple(),
        'units_map': {},
        'species_units': units.mM
    }

    def __init__(self, parameters=None):
        '''Simulates a chemical reaction network (CRN) with Bioscrape.

        Supports all the parameters of Vivarium Bioscrape plus:

        * ``external_species``: Iterable of species names that exist in
          the external environment. These species will be expected in
          the ``external`` port, and their updates will be communicated
          through the ``exchanges`` port. Species should be listed by
          their bioscrape name, not their Vivarium name, if the two
          names are different (see ``name_map`` below).
        * ``name_map``: Iterable of tuples ``(vivarium, bioscrape)``
          where ``vivarium`` is the path of the species in the state
          provided to ``next_update()`` by Vivarium, and ``bioscrape``
          is the name of the molecule to be used internally for
          bioscrape. This conversion is needed because Vivarium supports
          a larger set of variable names than bioscrape. See
          https://github.com/biocircuits/bioscrape/wiki/BioSCRAPE-XML
          for the bioscrape naming rules.
        * ``units_map``: Dictionary reflecting the nested port
          structure, with variables mapped to the expected unit. The
          conversion to these units happens before units are stripped
          and the magnitudes are passed to Bioscrape.
        '''
        super().__init__(parameters)
        self.rename_vivarium_to_bioscrape = {}
        for path, new_name in self.parameters['name_map']:
            # Paths must be tuples so that they are hashable.
            path = tuple(path)
            self.rename_vivarium_to_bioscrape[path] = new_name
            if 'species' in path:
                delta_path = tuple(
                    item if item != 'species' else 'delta_species'
                    for item in path
                )
                self.rename_vivarium_to_bioscrape[
                    delta_path] = new_name
            if 'external' in path:
                external_path = tuple(
                    item if item != 'external' else 'exchanges'
                    for item in path
                )
                self.rename_vivarium_to_bioscrape[
                    external_path] = new_name

        self.rename_bioscrape_to_vivarium = self._invert_map(
            self.rename_vivarium_to_bioscrape)

        self.external_species_bioscrape = [
            self.rename_vivarium_to_bioscrape.get(
                ('external', species), species)
            for species in self.parameters['external_species']
        ]

    @staticmethod
    def _invert_map(rename_map):
        '''Invert a renaming map.

        For example:

        >>> renaming_map = {
        ...     ('a', 'b'): 'B',
        ...     ('a', 'c'): 'C'
        ... }
        >>> inverted_map = {
        ...     ('a', 'B'): 'b',
        ...     ('a', 'C'): 'c'
        ... }
        >>> inverted_map == ExchangeAwareBioscrape._invert_map(
        ...     renaming_map)
        True
        '''
        inverted_map = {
            path[:-1] + (new_name,): path[-1]
            for path, new_name in rename_map.items()
        }
        return inverted_map

    def initial_state(self, config=None):
        initial_state = super().initial_state()
        for k in initial_state['species'].keys():
            initial_state['species'][k] *= self.parameters['species_units']
        return initial_state

    def ports_schema(self):
        schema = super().ports_schema()
        schema['globals']['volume']['_default'] *= units.fL
        schema['globals']['volume'].pop('_updater')  # don't use default updater

        rename_schema_for_vivarium = {}
        for path, new_name in self.rename_bioscrape_to_vivarium.items():
            rename_schema_for_vivarium[path] = new_name
            if 'exchanges' in path:
                for store in ('delta_species', 'species'):
                    other_path = tuple(
                        item if item != 'exchanges' else store
                        for item in path
                    )
                    rename_schema_for_vivarium[other_path] = new_name

        assert 'exchanges' not in schema
        schema['exchanges'] = {
            species: {
                '_default': 0,
                '_emit': True,
            }
            for species in self.external_species_bioscrape
        }

        assert 'external' not in schema
        schema['external'] = {
            species: {
                '_default': 0 * self.parameters['species_units'],
                '_emit': True,
            }
            for species in self.external_species_bioscrape
        }

        for species in self.external_species_bioscrape:
            del schema['species'][species]
            del schema['delta_species'][species]

        # add units to species and delta_species
        for species_id in schema['species'].keys():
            schema['species'][species_id]['_default'] *= self.parameters['species_units']
            schema['delta_species'][species_id]['_default'] *= self.parameters['species_units']
            schema['delta_species'][species_id]['_emit'] = True

        assert 'mmol_to_counts' not in schema['globals']
        schema['globals']['mmol_to_counts'] = {
            '_default': 0,
        }

        # Apply units map.
        units_map_for_schema = _transform_dict_leaves(
            self.parameters['units_map'], lambda x: {'_default': x})
        schema = self._add_units(schema, units_map_for_schema)

        schema = self._rename_variables(
            schema,
            rename_schema_for_vivarium,
        )
        return schema

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

    def _add_units(self, state, saved_units):
        """add units back in"""
        unit_state = state.copy()
        for key, value in saved_units.items():
            before = unit_state[key]
            if isinstance(value, dict):
                unit_state[key] = self._add_units(before, value)
            else:
                unit_state[key] = before * value
        return unit_state

    @staticmethod
    def _rename_variables(state, rename_map, path=tuple()):
        '''Rename variables in a hierarchy dict per a renaming map.

        For example:

        >>> renaming_map = {
        ...     ('a', 'b'): 'B',
        ...     ('a', 'c'): 'C'
        ... }
        >>> state = {
        ...     'a': {
        ...         'b': 1,
        ...         'c': 2,
        ...     }
        ... }
        >>> renamed_state = {
        ...     'a': {
        ...         'B': 1,
        ...         'C': 2,
        ...     }
        ... }
        >>> renamed_state == ExchangeAwareBioscrape._rename_variables(
        ...     state, renaming_map)
        True

        Args:
            state: The hierarchy dict of variables to rename.
            rename_map: The renaming map.
            path: If ``state`` is a sub-dict, the path to the sub-dict.
                Only used for recursive calls.

        Returns:
            The renamed hierarchy dict.
        '''
        # Base Case
        if not isinstance(state, dict):
            return state

        # Recursive Case
        new_state = {}
        for key, value in state.items():
            cur_path = path + (key,)
            new_key = rename_map.get(cur_path, cur_path[-1])
            new_state[new_key] = (
                ExchangeAwareBioscrape._rename_variables(
                    value, rename_map, cur_path
                )
            )

        return new_state

    def next_update(self, timestep, state):
        # NOTE: We assume species are in mM.
        # Prepare the state for the bioscrape process by moving species
        # from 'external' to 'species' where the bioscrape process
        # expects them to be.
        prepared_state = copy.deepcopy(state)
        prepared_state = self._rename_variables(
            prepared_state, self.rename_vivarium_to_bioscrape)
        prepared_state, saved_units = self._remove_units(
            prepared_state,
            units_map=self.parameters['units_map'])

        for species in self.external_species_bioscrape:
            assert species not in prepared_state['species']
            prepared_state['species'][species] = prepared_state['external'].pop(species)

        # Compute the update using the bioscrape process.
        update = super().next_update(timestep, prepared_state)

        # Make sure there are no NANs in the update.
        assert not np.any(np.isnan(list(update['species'].values())))

        # Add units back in
        species_update = update['species']
        delta_species_update = update['delta_species']
        update['species'] = self._add_units(
            species_update, saved_units['species'])
        update['delta_species'] = self._add_units(
            delta_species_update, saved_units['species'])

        # Postprocess the update to convert changes to external
        # molecules into exchanges.
        update.setdefault('exchanges', {})
        mmol_to_counts = prepared_state['globals']['mmol_to_counts'] / units.mM
        for species in self.external_species_bioscrape:
            if species in update['species']:
                delta_internal_conc = (
                    update['delta_species'][species]
                    * self.parameters['species_units']
                )
                exchange = delta_internal_conc * mmol_to_counts
                assert species not in update['exchanges']
                # Exchanges flowing out of the cell are positive.
                update['exchanges'][species] = exchange.to(
                    units.counts).magnitude
                del update['species'][species]
                del update['delta_species'][species]

        # We don't want to change the rates nor the globals.
        if 'rates' in update:
            del update['rates']
        if 'globals' in update:
            del update['globals']

        update = self._rename_variables(
            update, self.rename_bioscrape_to_vivarium)

        # TODO (ERAN) -- this needs to be generalized
        # To correct the diffusion between compartments with different volumes
        if 'tetracycline_cytoplasm' in update['delta_species']:
            # Net tetracycline diffusion from cytoplasm to periplasm
            tet_c_to_p = - update['delta_species']['tetracycline_cytoplasm']
            # Undoing tetracycline diffusion from cytoplasm to periplasm
            update['delta_species']['tetracycline_periplasm'] += tet_c_to_p
            update['species']['tetracycline_periplasm'] += tet_c_to_p
            # Applying correct tetracycline concentration change in periplasm
            update['delta_species']['tetracycline_periplasm'] -= tet_c_to_p * (
                    prepared_state['rates']['volume_c'] / prepared_state['rates']['volume_p'])
            update['species']['tetracycline_periplasm'] -= tet_c_to_p * (
                    prepared_state['rates']['volume_c'] / prepared_state['rates']['volume_p'])
        return update


def test_exchange_aware_bioscrape():
    a = Species('A')
    b = Species('B')
    species = [a, b]

    initial_concentrations = {
        a: 1e5,
        b: 0,
    }

    reaction = Reaction.from_massaction([a], [b], k_forward=1e-10)
    crn = ChemicalReactionNetwork(
        species=species,
        reactions=[reaction],
        initial_concentration_dict=initial_concentrations,
    )
    with tempfile.NamedTemporaryFile() as temp_file:
        crn.write_sbml_file(
            file_name=temp_file.name,
            for_bioscrape=True,
            check_validity=True,
        )
        proc = ExchangeAwareBioscrape({
            'sbml_file': temp_file.name,
            'external_species': ('a',),
            'name_map': (
                (('external', 'a'), str(a)),
                (('species', 'b'), str(b)),
            ),
            'units_map': {
                'rates': {
                    'k_forward': 1 / units.sec,
                }
            }
        })

    schema = proc.get_schema()
    expected_schema = {
        'delta_species': {
            'b': {
                '_default': 0.0 * units.mM,
                '_emit': True,
                '_updater': 'set',
            },
        },
        'exchanges': {
            'a': {
                '_default': 0,
                '_emit': True,
            },
        },
        'external': {
            'a': {
                '_default': 0.0 * units.mM,
                '_emit': True,
            },
        },
        'globals': {
            'mmol_to_counts': {'_default': 0},
            'volume': {
                '_default': 1.0 * units.fL,
                '_emit': True,
            },
        },
        'rates': {
            'k_forward': {
                '_default': 1e-10 / units.sec,
                '_updater': 'set',
            },
        },
        'species': {
            'b': {
                '_default': 0.0 * units.mM,
                '_divider': 'set',
                '_emit': True,
                '_updater': 'accumulate',
            },
        },
    }
    assert schema == expected_schema

    initial_state = {
        'external': {
            'a': initial_concentrations[a],
        },
        'species': {
            'b': initial_concentrations[b],
        },
        'globals': {
            'mmol_to_counts': 10 / units.millimolar,
            'volume': 0.32e-12 * units.mL,
        },
        'rates': {
            'k_forward': 1e-10 / units.sec,
        },
    }
    update = proc.next_update(1, initial_state)

    expected_update = {
        'species': {
            'b': 1e-5,
        },
        'delta_species': {
            'b': 1e-5,
        },
        'exchanges': {
            # mmol_to_counts = AVOGADRO * volume, so the exchange is the
            # concentration * mmol_to_counts
            'a': -1e-4,
        },
    }
    assert update.keys() == expected_update.keys()
    for key in expected_update:
        if key in ('species', 'delta_species', 'exchanges'):
            assert update[key].keys() == expected_update[key].keys()
            for species in expected_update[key]:
                assert abs(
                    update[key][species] - expected_update[key][species]
                ) <= 1e-6
        else:
            assert update[key] == expected_update[key]


if __name__ == '__main__':
    test_exchange_aware_bioscrape()
