import copy
import math

from scipy import constants
import numpy as np

from vivarium.core.process import Step
from vivarium.library.units import units


PI = math.pi
AVOGADRO = constants.N_A * 1 / units.mol


def length_from_volume(volume, width):
    """
    get cell length from volume, using the following equation for capsule volume, with V=volume, r=radius,
    a=length of cylinder without rounded caps, l=total length:

    V = (4/3)*PI*r^3 + PI*r^2*a
    l = a + 2*r
    """
    radius = width / 2
    cylinder_length = (volume - (4 / 3) * PI * radius**3) / (PI * radius**2)
    total_length = cylinder_length + 2 * radius
    return total_length


def volume_from_length(length, width):
    """
    inverse of length_from_volume
    """
    radius = width / 2
    cylinder_length = length - width
    volume = cylinder_length * (PI * radius**2) + (4 / 3) * PI * radius**3
    return volume


def surface_area_from_length(length, width):
    """
    SA = 3*PI*r^2 + 2*PI*r*a
    """
    radius = width / 2
    cylinder_length = length - width
    surface_area = 3 * PI * radius**2 + 2 * PI * radius * cylinder_length
    return surface_area


class DeriveGlobals(Step):
    """
    Process for deriving volume, mmol_to_counts, and shape from the cell mass
    """

    name = "globals_deriver"
    defaults = {
        "width": 1 * units.um,
        "initial_mass": 1339 * units.fg,  # wet mass in fg
    }

    def ports_schema(self):
        set_states = [
            "volume",
            "mmol_to_counts",
            "length",
            "surface_area",
        ]
        split_divide = ["volume", "length", "surface_area"]
        emit = {
            "global": [
                "volume",
                "width",
                "length",
                "surface_area",
            ]
        }

        # default state
        mass = self.parameters["initial_mass"]
        width = self.parameters["width"]
        density = 1100 * units.g / units.L
        volume = mass / density
        mmol_to_counts = (AVOGADRO * volume).to("L/mmol")
        length = length_from_volume(volume, width)
        surface_area = surface_area_from_length(length, width)

        default_state = {
            "global": {
                "mass": mass,
                "volume": volume.to(units.fL),
                "mmol_to_counts": mmol_to_counts,
                "density": density,
                "width": width,
                "length": length,
                "surface_area": surface_area,
            },
        }

        schema = {}
        for port, states in default_state.items():
            schema[port] = {}
            for state_id, value in states.items():
                schema[port][state_id] = {}
                if state_id in set_states:
                    schema[port][state_id]["_updater"] = "set"
                if state_id in emit[port]:
                    schema[port][state_id]["_emit"] = True
                if state_id in split_divide:
                    schema[port][state_id]["_divider"] = "split"
                if state_id in default_state[port]:
                    schema[port][state_id]["_default"] = default_state[port][state_id]

        return schema

    def next_update(self, timestep, states):
        density = states["global"]["density"]
        mass = states["global"]["mass"].to("fg")
        width = states["global"]["width"]

        # get volume from mass, and more variables from volume
        volume = mass / density
        mmol_to_counts = (AVOGADRO * volume).to("L/mmol")
        length = length_from_volume(volume, width)
        surface_area = surface_area_from_length(length, width)

        return {
            "global": {
                "volume": volume,
                "mmol_to_counts": mmol_to_counts,
                "length": length,
                "surface_area": surface_area,
            },
        }


def get_default_global_state():
    mass = 1339 * units.fg  # wet mass in fg
    density = 1100 * units.g / units.L
    volume = mass / density
    mmol_to_counts = AVOGADRO * volume

    return {
        "global": {
            "volume": volume.to("fL"),
            "mmol_to_counts": mmol_to_counts.to("L/mmol"),
        }
    }


def test_deriver(total_time=10, return_data=False):
    growth_rate = 6e-3

    # configure process
    deriver = DeriveGlobals({})
    state = deriver.default_state()

    # initialize saved data
    saved_state = {}

    ## simulation
    timeline = [(total_time, {})]
    time = 0
    timestep = 1  # sec
    saved_state[time] = state
    while time < timeline[-1][0]:
        time += timestep
        for t, change_dict in timeline:
            if time >= t:
                for key, change in change_dict.items():
                    state[key].update(change)

        # set mass, counts
        mass = state["global"]["mass"]
        state["global"]["mass"] = mass * np.exp(growth_rate * timestep)

        # get update
        update = deriver.next_update(timestep, state)

        # set derived state
        state["global"].update(update["global"])

        # save state
        saved_state[time] = copy.deepcopy(state)

    if return_data:
        return saved_state


if __name__ == "__main__":
    saved_data = test_deriver(100, return_data=True)
    print(saved_data)
