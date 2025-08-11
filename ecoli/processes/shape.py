"""
=====
Shape
=====

``Shape`` is used to calculate shape properties using 3D capsule geometry.
Outputs `length and `surface_area` are determined from inputs `volume` and `width`.
These variables are required to plug into a `Lattice Environment`
"""

import math

from scipy.constants import N_A
from vivarium.core.process import Step
from vivarium.library.units import units, Quantity

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
    cylinder_length = (volume - (4 / 3) * PI * radius**3) / (PI * radius**2)
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

    :math:`SA = 4*PI*r^2 + 2*PI*r*a`
    """
    radius = width / 2
    cylinder_length = length - width
    surface_area = 4 * PI * radius**2 + 2 * PI * radius * cylinder_length
    return surface_area


def mmol_to_counts_from_volume(volume):
    """mmol_to_counts has units L/mmol"""
    return (volume * AVOGADRO).to(units.L / units.mmol)


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

    name = "ecoli-shape"
    defaults = {
        "width": 1.0 * units.um,
        "periplasm_fraction": 0.2,
        "cytoplasm_fraction": 0.8,
        "initial_cell_volume": 1.2 * units.fL,
        "initial_mass": 1339 * units.fg,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.outer_to_inner_area = (
            math.pow(self.parameters["cytoplasm_fraction"], 1 / 3) ** 2
        )

    def ports_schema(self):
        schema = {
            "cell_global": {
                "volume": {
                    "_default": 0 * units.fL,
                    "_updater": "set",
                    "_emit": True,
                    "_divider": "split",
                },
                "width": {
                    "_default": 0 * units.um,
                    "_updater": "set",
                    "_emit": True,
                    "_divider": "set",
                },
                "length": {
                    "_default": 0 * units.um,
                    "_updater": "set",
                    "_emit": True,
                    "_divider": "split",
                },
                "outer_surface_area": {
                    "_default": 0 * units.um**2,
                    "_updater": "set",
                    "_emit": True,
                    "_divider": "split",
                },
                "inner_surface_area": {
                    "_default": 0 * units.um**2,
                    "_updater": "set",
                    "_emit": True,
                    "_divider": "split",
                },
                "mmol_to_counts": {
                    "_default": 0 / units.millimolar,
                    "_emit": True,
                    "_divider": "split",
                    "_updater": "set",
                },
                "mass": {
                    "_default": 0 * units.fg,
                    "_updater": "set",
                    "_emit": True,
                    "_divider": "split",
                },
            },
            "listener_cell_mass": {
                "_default": self.parameters["initial_mass"].magnitude,  # fg
            },
            "listener_cell_volume": {
                "_default": self.parameters["initial_cell_volume"].magnitude,  # fL
            },
            "periplasm_global": {
                "volume": {
                    "_default": self.parameters["initial_cell_volume"]
                    * self.parameters["periplasm_fraction"],  # fL
                    "_emit": True,
                    "_divider": "split",
                    "_updater": "set",
                },
                "mmol_to_counts": {
                    "_default": 0 / units.millimolar,
                    "_emit": True,
                    "_divider": "split",
                    "_updater": "set",
                },
            },
            "cytoplasm_global": {
                "volume": {
                    "_default": self.parameters["initial_cell_volume"]
                    * self.parameters["cytoplasm_fraction"],  # fL
                    "_emit": True,
                    "_divider": "split",
                    "_updater": "set",
                },
                "mmol_to_counts": {
                    "_default": 0 / units.millimolar,
                    "_emit": True,
                    "_divider": "split",
                    "_updater": "set",
                },
            },
        }
        return schema

    def initial_state(self, config=None):
        cell_volume = self.parameters["initial_cell_volume"]
        assert isinstance(cell_volume, Quantity)
        width = self.parameters["width"]
        assert isinstance(width, Quantity)
        length = length_from_volume(cell_volume, width)
        outer_surface_area = surface_area_from_length(length, width)
        inner_surface_area = self.outer_to_inner_area * outer_surface_area

        assert (
            self.parameters["periplasm_fraction"]
            + self.parameters["cytoplasm_fraction"]
            == 1
        )
        periplasm_volume = cell_volume * self.parameters["periplasm_fraction"]
        cytoplasm_volume = cell_volume * self.parameters["cytoplasm_fraction"]

        mass = self.parameters["initial_mass"]
        assert isinstance(mass, Quantity)
        return {
            "cell_global": {
                "volume": cell_volume,
                "width": width,
                "length": length,
                "outer_surface_area": outer_surface_area,
                "inner_surface_area": inner_surface_area,
                "mmol_to_counts": mmol_to_counts_from_volume(cell_volume),
                "mass": mass,
            },
            "listener_cell_mass": mass.magnitude,
            "listener_cell_volume": cell_volume.magnitude,
            "periplasm_global": {
                "volume": periplasm_volume,
                "mmol_to_counts": mmol_to_counts_from_volume(periplasm_volume),
            },
            "cytoplasm_global": {
                "volume": cytoplasm_volume,
                "mmol_to_counts": mmol_to_counts_from_volume(cytoplasm_volume),
            },
        }

    def next_update(self, timestep, states):
        for port in ("cell_global", "periplasm_global", "cytoplasm_global"):
            for variable, value in states[port].items():
                assert isinstance(value, Quantity), (
                    f"{variable}={value} is not a Quantity"
                )

        width = states["cell_global"]["width"]
        cell_volume = states["listener_cell_volume"] * units.fL

        assert (
            self.parameters["periplasm_fraction"]
            + self.parameters["cytoplasm_fraction"]
            == 1
        )
        periplasm_volume = cell_volume * self.parameters["periplasm_fraction"]
        cytoplasm_volume = cell_volume * self.parameters["cytoplasm_fraction"]

        # calculate length and surface area
        length = length_from_volume(cell_volume, width)
        outer_surface_area = surface_area_from_length(length, width)
        inner_surface_area = self.outer_to_inner_area * outer_surface_area

        update = {
            "cell_global": {
                "length": length,
                "outer_surface_area": outer_surface_area,
                "inner_surface_area": inner_surface_area,
                "mmol_to_counts": mmol_to_counts_from_volume(cell_volume),
                "mass": states["listener_cell_mass"] * units.fg,
                "volume": cell_volume,
            },
            "periplasm_global": {
                "volume": periplasm_volume,
                "mmol_to_counts": mmol_to_counts_from_volume(periplasm_volume),
            },
            "cytoplasm_global": {
                "volume": cytoplasm_volume,
                "mmol_to_counts": mmol_to_counts_from_volume(cytoplasm_volume),
            },
        }
        return update
