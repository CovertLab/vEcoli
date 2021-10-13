"""
=====
Shape
=====

``Shape`` is used to calculate shape properties using 3D capsule geometry.
Outputs `length and `surface_area` are determined from inputs `volume` and `width`.
These variables are required to plug into a `Lattice Environment
<https://github.com/vivarium-collective/vivarium-multibody/blob/master/vivarium_multibody/composites/lattice.py>`_
"""

import math

from vivarium.core.process import Deriver
from vivarium.library.units import units

PI = math.pi


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


class Shape(Deriver):
    """ Shape Deriver 
    
    Derives cell length and surface area from width and volume.

    Ports:
    * **global**: Should be given the agent's boundary store. Contains variables 
        **volume**, **width**, **length**, and **surface_area**.

    Arguments:
        parameters (dict): A dictionary that can contain the
            follwing configuration options:

            * **width** (:py:class:`float`): Initial width of the cell in
              microns
    """

    name = 'ecoli-shape'
    defaults = {
        'width': 1,  # um
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

    def ports_schema(self):
        """ includes **global** port """
        default_state = {
            'global': {
                'volume': 0.0 * units.fL,
                'width': self.parameters['width'],
                'length': 0.0,
                'surface_area': 0.0,
            }
        }

        schema = {
            'global': {
                variable: {
                    '_updater': 'set',
                    '_emit': True,
                    '_divider': (
                        'set' if variable == 'width' else 'split'
                    ),
                    '_default': default_state['global'][variable]
                }
                for variable in default_state['global']
            }
        }
        return schema

    def next_update(self, timestep, states):
        """
        Inputs:
        * ['global']['width']
        * ['global']['volume']

        Updates:
        * ['global']['length']
        * ['global']['surface_area']
        """
        width = states['global']['width']
        volume = states['global']['volume']

        length = length_from_volume(volume.magnitude, width)
        surface_area = surface_area_from_length(length, width)

        return {
            'global': {
                'length': length,
                'surface_area': surface_area,
            },
        }
