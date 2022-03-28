"""
=====
Shape
=====

``Shape`` is used to calculate shape properties using 3D capsule geometry.
Outputs `length and `surface_area` are determined from inputs `volume` and `width`.
These variables are required to plug into a `Lattice Environment
"""

import math

from scipy.constants import N_A
from vivarium.core.process import Step
from vivarium.library.units import units

PI = math.pi
AVOGADRO = N_A / units.mol


def length_from_volume(volume, width):
    """
    get cell length from volume, using the following equation for capsule volume, with V=volume, r=radius,
    a=length of cylinder without rounded caps, l=total length:

    :math:`V = (4/3)*PI*r^3 + PI*r^2*a`
    :math:`l = a + 2*r`
    """
    radius = width / 2
    cylinder_length = (volume - (4/3) * PI * radius**3) / (PI * radius**2)
    total_length = cylinder_length + 2 * radius
    return total_length


def volume_from_length(length, width):
    """
    get volume from length and width, using 3D capsule geometry
    """
    radius = width / 2
    cylinder_length = length - width
    volume = cylinder_length * (PI * radius**2) + (4 / 3) * PI * radius**3
    return volume


def surface_area_from_length(length, width):
    """
    get surface area from length and width, using 3D capsule geometry

    :math:`SA = 3*PI*r^2 + 2*PI*r*a`
    """
    radius = width / 2
    cylinder_length = length - width
    surface_area = 3 * PI * radius**2 + 2 * PI * radius * cylinder_length
    return surface_area


def mmol_to_counts_from_volume(volume):
    '''mmol_to_counts has units L/mmol'''
    return (volume * AVOGADRO).to(units.L/units.mmol)


class Shape(Step):
    """Shape Step

    Derives cell length and surface area from width and volume.

    Ports:

    * **cell_global**: Should be given the agent's boundary store.
      Contains variables: **volume**, **width**, **length**, and
      **surface_area**.
    * **periplasm_global**: Contains the **volume** variable for the
      volume of the periplasm.

    Arguments:
        parameters (dict): A dictionary that can contain the
            following configuration options:

            * **width** (:py:class:`float`): Initial width of the cell in
              microns
    """

    name = 'ecoli-shape'
    defaults = {
        'width': 1.0,  # um
        'periplasm_fraction': 0.3,
        'cytosol_fraction': 0.7,
        'initial_cell_volume': 1.2,  # fL
        'initial_mass': 1339,  # * units.fg
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

    def ports_schema(self):

        schema = {
            'cell_global': {
                'volume': {
                    '_default': 0,
                    '_updater': 'set',
                    '_emit': True,
                    '_divider': 'split',
                },
                'width': {
                    '_default': 0,
                    '_updater': 'set',
                    '_emit': True,
                    '_divider': 'set',
                },
                'length': {
                    '_default': 0,
                    '_updater': 'set',
                    '_emit': True,
                    '_divider': 'split',
                },
                'surface_area': {
                    '_default': 0,  # * units.um**2
                    '_updater': 'set',
                    '_emit': True,
                    '_divider': 'split',
                },
                'mmol_to_counts': {
                    '_default': 0 / units.millimolar,
                    '_emit': True,
                    '_divider': 'split',
                    '_updater': 'set',
                },
                'mass': {
                    '_default': 0,  # * units.fg
                    '_updater': 'set',
                    '_emit': True,
                    '_divider': 'split',
                },
            },
            'listener_cell_mass': {
                '_default': 0,
            },
            'periplasm_global': {
                'volume': {
                    '_default': 0,  # * units.fL
                    '_emit': True,
                    '_divider': 'split',
                    '_updater': 'set',
                },
                'mmol_to_counts': {
                    '_default': 0 / units.millimolar,
                    '_emit': True,
                    '_divider': 'split',
                    '_updater': 'set',
                },
            },
            'cytosol_global': {
                'volume': {
                    '_default': 0,  # * units.fL
                    '_emit': True,
                    '_divider': 'split',
                    '_updater': 'set',
                },
                'mmol_to_counts': {
                    '_default': 0 / units.millimolar,
                    '_emit': True,
                    '_divider': 'split',
                    '_updater': 'set',
                },
            },
        }
        return schema

    def initial_state(self, config=None):
        cell_volume = self.parameters['initial_cell_volume'] * units.fL
        width = self.parameters['width'] * units.um
        length = length_from_volume(cell_volume, width)
        surface_area = surface_area_from_length(length, width).magnitude

        assert self.parameters['periplasm_fraction'] + self.parameters['cytosol_fraction'] == 1
        periplasm_volume = cell_volume * self.parameters['periplasm_fraction']
        cytosol_volume = cell_volume * self.parameters['cytosol_fraction']

        mass = self.parameters['initial_mass']  # .to(units.fg)
        return {
            'cell_global': {
                'volume': cell_volume.magnitude,
                'width': width.magnitude,
                'length': length.magnitude,
                'surface_area': surface_area,
                'mmol_to_counts': mmol_to_counts_from_volume(
                    cell_volume),
                'mass': mass,
            },
            'listener_cell_mass': mass,  # .magnitude
            'periplasm_global': {
                'volume': periplasm_volume,
                'mmol_to_counts': mmol_to_counts_from_volume(
                    periplasm_volume),
            },
            'cytosol_global': {
                'volume': cytosol_volume,
                'mmol_to_counts': mmol_to_counts_from_volume(
                    cytosol_volume),
            },
        }

    def next_update(self, timestep, states):
        width = states['cell_global']['width'] * units.um
        cell_volume = states['cell_global']['volume'] * units.fL

        assert self.parameters['periplasm_fraction'] + self.parameters['cytosol_fraction'] == 1
        periplasm_volume = cell_volume * self.parameters['periplasm_fraction']
        cytosol_volume = cell_volume * self.parameters['cytosol_fraction']

        # calculate length and surface area
        length = length_from_volume(cell_volume, width)
        surface_area = surface_area_from_length(length, width).magnitude

        update = {
            'cell_global': {
                'length': length.magnitude,
                'surface_area': surface_area,
                'mmol_to_counts': mmol_to_counts_from_volume(
                    cell_volume),
                'mass': states['listener_cell_mass'],  # * units.fg,
            },
            'periplasm_global': {
                'volume': periplasm_volume.magnitude,
                'mmol_to_counts': mmol_to_counts_from_volume(
                    periplasm_volume),
            },
            'cytosol_global': {
                'volume': cytosol_volume.magnitude,
                'mmol_to_counts': mmol_to_counts_from_volume(
                    cytosol_volume),
            }
        }
        return update
