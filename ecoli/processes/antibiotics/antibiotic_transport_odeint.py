import numpy as np
from scipy.constants import N_A
from scipy.integrate import solve_ivp
from vivarium.core.process import Process
from vivarium.library.units import units

from ecoli.library.units import add_units, remove_units
from ecoli.processes.antibiotics.antibiotic_transport_steady_state import (
    SPECIES,
    SPECIES_TO_INDEX,
    UNITS,
    GAS_CONSTANT,
    TEMPERATURE,
    FARADAY,
    INNER_POTENTIAL,
    OUTER_POTENTIAL,
    species_derivatives,
    species_array_to_dict,
    species_dict_to_array,
)


def update_from_odeint(
    initial_state, reaction_params, outer_internal_bias, inner_internal_bias, timestep
):
    """Compute an update given a steady-state solution.

    Beginning from the initial state, numerically integrates the
    species ODEs to find the final state.

    Args:
        initial_state: Initial state dictionary.
        reaction_params: Dictionary of reaction parameters.
        outer_internal_bias: See :py:func:`species_derivatives`.
        inner_internal_bias: See :py:func:`species_derivatives`.
        timestep: Timestep for update.

    Returns:
        Update dictionary.
    """
    assert set(SPECIES) == initial_state.keys()
    initial_state_arr = species_dict_to_array(initial_state, SPECIES_TO_INDEX)

    args = (reaction_params, outer_internal_bias, inner_internal_bias)
    result = solve_ivp(
        lambda t, state_arr, *args: species_derivatives(state_arr, *args),
        [0, timestep],
        initial_state_arr,
        args=args,
    )
    assert result.success
    final_state_arr = result.y[:, -1].T

    delta_arr = final_state_arr - initial_state_arr
    delta = species_array_to_dict(delta_arr, SPECIES_TO_INDEX)
    return {
        "species": delta,
    }


class AntibioticTransportOdeint(Process):
    name = "antibiotic-transport-odeint"
    defaults = {
        "initial_reaction_parameters": {},
        "diffusion_only": False,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.antibiotics = list(self.parameters["initial_reaction_parameters"].keys())

    def initial_state(self, config=None):
        state = {
            antibiotic: {
                "reaction_parameters": self.parameters["initial_reaction_parameters"][
                    antibiotic
                ]
            }
            for antibiotic in self.antibiotics
        }
        return state

    def ports_schema(self):
        schema = {
            antibiotic: {
                "species": {
                    species: {
                        "_default": 0.0 * units.mM,
                        "_updater": "accumulate",
                        "_emit": True,
                        "_divider": "set",
                    }
                    for species in SPECIES
                },
                "exchanges": {
                    "external": {
                        "_default": 0,
                        "_emit": True,
                    }
                },
                "reaction_parameters": {
                    reaction: {
                        parameter: {
                            "_default": 0 * unit,
                            "_emit": True,
                        }
                        for parameter, unit in reaction_params.items()
                    }
                    for reaction, reaction_params in self.parameters[
                        "initial_reaction_parameters"
                    ][antibiotic].items()
                },
            }
            for antibiotic in self.antibiotics
        }

        return schema

    def next_update(self, timestep, state):
        update = {}
        for antibiotic in self.antibiotics:
            antibiotic_state = state[antibiotic]
            # Prepare the state by doing unit conversions.
            prepared_state = {
                "species": antibiotic_state["species"],
                "reaction_parameters": antibiotic_state["reaction_parameters"],
            }
            prepared_state, saved_units = remove_units(prepared_state, UNITS)

            charge = prepared_state["reaction_parameters"]["diffusion"]["charge"]
            # Biases diffusion to favor higher internal concentrations
            # according to the Goldman-Hodgkin-Katz flux equation assuming
            # the outer membrane has a potential from the Donnan equilibrium.
            outer_internal_bias = (
                charge * FARADAY * OUTER_POTENTIAL / GAS_CONSTANT / TEMPERATURE
            ).magnitude
            inner_internal_bias = (
                charge * FARADAY * INNER_POTENTIAL / GAS_CONSTANT / TEMPERATURE
            ).magnitude

            # No export or hydrolysis if modelling diffusion only
            if self.parameters["diffusion_only"]:
                prepared_state["reaction_parameters"]["export"]["kcat"] = 0 / units.sec
                prepared_state["reaction_parameters"]["hydrolysis"]["kcat"] = (
                    0 / units.sec
                )

            # Compute the update.
            antibiotic_update = update_from_odeint(
                prepared_state["species"],
                prepared_state["reaction_parameters"],
                outer_internal_bias,
                inner_internal_bias,
                timestep,
            )

            # Make sure there are no NANs in the update.
            assert not np.any(np.isnan(list(antibiotic_update["species"].values())))

            # Change in external counts = -(Change in internal counts)
            # Divide concentrations by 1000 to convert mM to M
            delta_periplasm_counts = (
                (
                    antibiotic_update["species"]["periplasm"]
                    + antibiotic_update["species"]["hydrolyzed_periplasm"]
                )
                / 1000
                * (
                    N_A
                    * prepared_state["reaction_parameters"]["diffusion"][
                        "periplasm_volume"
                    ]
                )
            )
            delta_cytoplasm_counts = (
                (
                    antibiotic_update["species"]["cytoplasm"]
                    + antibiotic_update["species"]["hydrolyzed_cytoplasm"]
                )
                / 1000
                * (
                    N_A
                    * prepared_state["reaction_parameters"]["diffusion"][
                        "cytoplasm_volume"
                    ]
                )
            )
            delta_internal_counts = delta_periplasm_counts + delta_cytoplasm_counts

            # Add units back in
            antibiotic_update["species"] = add_units(
                antibiotic_update["species"],
                saved_units["species"],
                strict=not self.parameters["diffusion_only"],
            )

            antibiotic_update["exchanges"] = {"external": -delta_internal_counts}

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
    proc = AntibioticTransportOdeint(
        {
            "initial_reaction_parameters": {
                "antibiotic": {},
            },
        }
    )
    initial_state = {
        "antibiotic": {
            "species": {
                "periplasm": 0 * units.mM,
                "hydrolyzed_periplasm": 0 * units.mM,
                "cytoplasm": 0 * units.mM,
                "hydrolyzed_cytoplasm": 0 * units.mM,
                "external": 3 * units.mM,
            },
            "reaction_parameters": {
                "diffusion": {
                    "outer_permeability": 2 * units.dm / units.sec,
                    "outer_area": 3 * units.dm**2,
                    "periplasm_volume": 2 * units.L,
                    "charge": 0 * units.count,
                    "inner_permeability": 0 * units.dm / units.sec,
                    "inner_area": 3 * units.dm**2,
                    "cytoplasm_volume": 2 * units.L,
                },
                "export": {
                    "outer_kcat": 4 / units.sec,
                    "outer_km": 2 * units.mM,
                    "outer_enzyme_conc": 1 * units.mM,
                    "outer_n": 1 * units.count,
                    "inner_kcat": 0 / units.sec,
                    "inner_km": 2 * units.mM,
                    "inner_enzyme_conc": 1 * units.mM,
                    "inner_n": 1 * units.count,
                },
                "hydrolysis": {
                    "outer_kcat": 4 / units.sec,
                    "outer_km": 2 * units.mM,
                    "outer_enzyme_conc": 0.5 * units.mM,
                    "outer_n": 1 * units.count,
                    "inner_kcat": 0 / units.sec,
                    "inner_km": 2 * units.mM,
                    "inner_enzyme_conc": 0.5 * units.mM,
                    "inner_n": 1 * units.count,
                },
            },
        },
    }
    update = proc.next_update(1, initial_state)["antibiotic"]

    # Compute expected update
    initial_arr = np.array([3, 0, 0])
    result = solve_ivp(_dummy_derivative, (0, 1), initial_arr)
    assert result.success
    final_arr = result.y[:, -1]
    delta_arr = final_arr - initial_arr

    expected_update = {
        "species": {
            "periplasm": delta_arr[1] * units.mM,
            "hydrolyzed_periplasm": delta_arr[2] * units.mM,
            "cytoplasm": 0 * units.mM,
            "hydrolyzed_cytoplasm": 0 * units.mM,
            "external": 0 * units.mM,
        },
        "exchanges": {
            # Exchanges are in units of counts, but the species are in
            # units of mM with a volume of 1L.
            "external": (
                delta_arr[0]
                * units.mM
                * initial_state["antibiotic"]["reaction_parameters"]["diffusion"][
                    "periplasm_volume"
                ]
                * N_A
                / units.mol
            )
            .to(units.count)
            .magnitude,
        },
    }
    assert update.keys() == expected_update.keys()
    for key in expected_update:
        if key in ("species", "delta_species", "exchanges"):
            assert update[key].keys() == expected_update[key].keys()
            for species in expected_update[key]:
                assert abs(
                    update[key][species] - expected_update[key][species]
                ) <= 1e-4 * abs(expected_update[key][species])
        else:
            assert update[key] == expected_update[key]


if __name__ == "__main__":
    test_antibiotic_transport_odeint()
