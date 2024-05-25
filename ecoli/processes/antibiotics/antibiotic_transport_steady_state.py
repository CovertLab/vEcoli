import numpy as np
from scipy.constants import N_A
from scipy.integrate import solve_ivp
from scipy.optimize import root
from vivarium.core.process import Process
from vivarium.library.units import units

from ecoli.library.parameters import param_store
from ecoli.library.units import add_units, remove_units

# NOTE: These must match the math in species_derivatives().
SPECIES = (
    "periplasm",
    "cytoplasm",
    "external",
    "hydrolyzed_periplasm",
    "hydrolyzed_cytoplasm",
)
SPECIES_TO_INDEX = {species: i for i, species in enumerate(SPECIES)}
REACTIONS = {
    # Diffusion from environment into periplasm
    # Note that external concentration is not included in
    # stoichiometry because we assume it is constant for timestep
    "periplasm_diffusion": (tuple(), ("periplasm",)),
    # Diffusion from periplasm into cytoplasm
    "cytoplasm_diffusion": (("periplasm",), ("cytoplasm",)),
    # Active efflux from periplasm
    # External concentration is assumed to be constant for timestep
    "periplasm_export": (("periplasm",), tuple()),
    # Active efflux from cytoplasm
    "cytoplasm_export": (("cytoplasm",), ("periplasm",)),
    # Hydrolysis in periplasm
    "periplasm_hydrolysis": (("periplasm",), ("hydrolyzed_periplasm",)),
    # Hydrolysis in cytoplasm
    "cytoplasm_hydrolysis": (("cytoplasm",), ("hydrolyzed_cytoplasm",)),
}
REACTIONS_TO_INDEX = {reaction: i for i, reaction in enumerate(REACTIONS)}
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
#: Describes the expected units for each species and reaction parameter.
#: AntibioticTransportSteadyState uses this dictionary to do unit
#: conversions.
UNITS = {
    "species": {species: units.mM for species in SPECIES},
    # One set of parameters for outer membrane, another for inner
    "reaction_parameters": {
        "diffusion": {
            "outer_permeability": units.dm / units.sec,
            "outer_area": units.dm**2,
            "periplasm_volume": units.L,
            "charge": units.count,
            "inner_permeability": units.dm / units.sec,
            "inner_area": units.dm**2,
            "cytoplasm_volume": units.L,
        },
        "export": {
            "outer_kcat": 1 / units.sec,
            "outer_km": units.mM,
            "outer_enzyme_conc": units.mM,
            "outer_n": units.count,
            "inner_kcat": 1 / units.sec,
            "inner_km": units.mM,
            "inner_enzyme_conc": units.mM,
            "inner_n": units.count,
        },
        "hydrolysis": {
            "outer_kcat": 1 / units.sec,
            "outer_km": units.mM,
            "outer_enzyme_conc": units.mM,
            "outer_n": units.count,
            "inner_kcat": 1 / units.sec,
            "inner_km": units.mM,
            "inner_enzyme_conc": units.mM,
            "inner_n": units.count,
        },
    },
}

# Cache these values so no additional units conversions are necessary
FARADAY = param_store.get(("faraday_constant",)).to(units.C / units.mol)
OUTER_POTENTIAL = param_store.get(("outer_potential",)).to(units.V)
INNER_POTENTIAL = param_store.get(("inner_potential",)).to(units.V)
GAS_CONSTANT = param_store.get(("gas_constant",)).to(units.J / units.mol / units.K)
TEMPERATURE = param_store.get(("temperature",)).to(units.K)


def species_derivatives(
    state_arr, reaction_params, outer_internal_bias, inner_internal_bias
):
    """Compute the derivatives for each species.

    Args:
        state_arr: Array of molecule counts in periplasm, cytoplasm,
            and environment at which to evaluate the derivative. Assumes that
            environment concentration is relatively constant over the timestep
            (reasonable if environment volume >> internal volume or if timestep
            is short enough).
        reaction_params: Dictionary of reaction parameters.
        outer_internal_bias: A positive bias that, when greater than one,
            favors influx into the periplasm over efflux diffusion. This can
            be used to model the effect of the membrane potential on diffusion
            equilibrium.
        inner_internal_bias: A positive bias that, when greater than one,
            favors influx into the cytoplasm over efflux diffusion. This can
            be used to model the effect of the membrane potential on diffusion
            equilibrium.

    Returns:
        Derivatives of each species as an array.
    """
    # Parse state
    state = species_array_to_dict(state_arr, SPECIES_TO_INDEX)
    periplasm = state["periplasm"]
    cytoplasm = state["cytoplasm"]
    external = state["external"]

    # Parse reaction parameters
    outer_area = reaction_params["diffusion"]["outer_area"]
    outer_permeability = reaction_params["diffusion"]["outer_permeability"]
    periplasm_volume = reaction_params["diffusion"]["periplasm_volume"]
    inner_area = reaction_params["diffusion"]["inner_area"]
    inner_permeability = reaction_params["diffusion"]["inner_permeability"]
    cytoplasm_volume = reaction_params["diffusion"]["cytoplasm_volume"]

    # TODO: Pull the Michaelis-Menten logic into a separate function for
    # reuse.
    outer_kcat_export = reaction_params["export"]["outer_kcat"]
    outer_km_export = reaction_params["export"]["outer_km"]
    outer_pump_conc = reaction_params["export"]["outer_enzyme_conc"]
    outer_n_export = reaction_params["export"]["outer_n"]
    outer_kcat_hydrolysis = reaction_params["hydrolysis"]["outer_kcat"]
    outer_km_hydrolysis = reaction_params["hydrolysis"]["outer_km"]
    outer_hydrolase_conc = reaction_params["hydrolysis"]["outer_enzyme_conc"]
    outer_n_hydrolysis = reaction_params["hydrolysis"]["outer_n"]
    inner_kcat_export = reaction_params["export"]["inner_kcat"]
    inner_km_export = reaction_params["export"]["inner_km"]
    inner_pump_conc = reaction_params["export"]["inner_enzyme_conc"]
    inner_n_export = reaction_params["export"]["inner_n"]
    inner_kcat_hydrolysis = reaction_params["hydrolysis"]["inner_kcat"]
    inner_km_hydrolysis = reaction_params["hydrolysis"]["inner_km"]
    inner_hydrolase_conc = reaction_params["hydrolysis"]["inner_enzyme_conc"]
    inner_n_hydrolysis = reaction_params["hydrolysis"]["inner_n"]

    if outer_internal_bias != 0:
        periplasm_diffusion_rate = (
            outer_area
            * outer_permeability
            / (periplasm_volume)
            * (
                outer_internal_bias
                * (external - periplasm * np.exp(outer_internal_bias))
                / (np.exp(outer_internal_bias) - 1)
            )
        )
    # Goldman-Hodgkin-Katz equation simplfies to Fick's law when bias = 0
    else:
        periplasm_diffusion_rate = (
            outer_area
            * outer_permeability
            / (periplasm_volume)
            * (external - periplasm)
        )

    if inner_internal_bias != 0:
        cytoplasm_diffusion_rate = (
            inner_area
            * inner_permeability
            / (cytoplasm_volume)
            * (
                inner_internal_bias
                * (periplasm - cytoplasm * np.exp(inner_internal_bias))
                / (np.exp(inner_internal_bias) - 1)
            )
        )
    # Goldman-Hodgkin-Katz equation simplfies to Fick's law when bias = 0
    else:
        cytoplasm_diffusion_rate = (
            inner_area
            * inner_permeability
            / (cytoplasm_volume)
            * (periplasm - cytoplasm)
        )

    periplasm_export_rate = (
        outer_kcat_export * outer_pump_conc * periplasm**outer_n_export
    ) / (outer_km_export + periplasm**outer_n_export)
    periplasm_hydrolysis_rate = (
        outer_kcat_hydrolysis * outer_hydrolase_conc * periplasm**outer_n_hydrolysis
    ) / (outer_km_hydrolysis + periplasm**outer_n_hydrolysis)
    cytoplasm_export_rate = (
        inner_kcat_export * inner_pump_conc * cytoplasm**inner_n_export
    ) / (inner_km_export + cytoplasm**inner_n_export)
    cytoplasm_hydrolysis_rate = (
        inner_kcat_hydrolysis * inner_hydrolase_conc * cytoplasm**inner_n_hydrolysis
    ) / (inner_km_hydrolysis + cytoplasm**inner_n_hydrolysis)
    reaction_rates = {
        "periplasm_diffusion": periplasm_diffusion_rate,
        "cytoplasm_diffusion": cytoplasm_diffusion_rate,
        "periplasm_export": periplasm_export_rate,
        "cytoplasm_export": cytoplasm_export_rate,
        "periplasm_hydrolysis": periplasm_hydrolysis_rate,
        "cytoplasm_hydrolysis": cytoplasm_hydrolysis_rate,
    }
    reaction_rates_arr = species_dict_to_array(reaction_rates, REACTIONS_TO_INDEX)

    return STOICH @ reaction_rates_arr


def internal_derivative(
    internal, external, reaction_params, outer_internal_bias, inner_internal_bias
):
    """Compute the derivatives of the ``periplasm`` and ``cytoplasm``
    concentrations.

    Args:
        internal: Vector of the form [``periplasm``, ``cytoplasm``].
        external: Current concentration of ``external`` species.
        reaction_params: See :py:func:`species_derivatives`.
        inner_internal_bias: See :py:func:`species_derivatives`.
        outer_internal_bias: See :py:func:`species_derivatives`.

    Returns:
        Derivative of the internal species, as a float.
    """
    state = {
        "periplasm": internal[0],
        "cytoplasm": internal[1],
        "external": external,
        "hydrolyzed_periplasm": 0,
        "hydrolyzed_cytoplasm": 0,
    }
    state_arr = species_dict_to_array(state, SPECIES_TO_INDEX)
    derivatives = species_derivatives(
        state_arr, reaction_params, outer_internal_bias, inner_internal_bias
    )
    return [
        derivatives[SPECIES_TO_INDEX["periplasm"]],
        derivatives[SPECIES_TO_INDEX["cytoplasm"]],
    ]


def find_steady_state(
    external, reaction_params, outer_internal_bias, inner_internal_bias
):
    """Find steady-state concentration of species in periplasm and cytoplasm.

    Args:
        external: Current concentration of ``external`` species.
        reaction_params: See :py:func:`species_derivatives`.
        outer_internal_bias: See :py:func:`species_derivatives`.
        inner_internal_bias: See :py:func:`species_derivatives`.

    Returns:
        Steady-state concentrations of the internal species, as a
        vector of the form [``periplasm``, ``cytoplasm``].
    """
    args = (external, reaction_params, outer_internal_bias, inner_internal_bias)
    result = root(
        internal_derivative,
        [
            external * outer_internal_bias,
            external * outer_internal_bias * inner_internal_bias,
        ],
        args=args,
    )
    assert result.success
    return result.x


def update_from_steady_state(
    internal_steady_state,
    initial_state,
    reaction_params,
    outer_internal_bias,
    inner_internal_bias,
    timestep,
):
    """Compute an update given a steady-state solution.

    Beginning from the steady-state solution, numerically integrates the
    species ODEs to find the final state.

    Args:
        internal_steady_state: Vector of steady-state concentrations
            [``periplasm``, ``cytoplasm``].
        initial_state: Initial state dictionary.
        reaction_params: Dictionary of reaction parameters.
        outer_internal_bias: See :py:func:`species_derivatives`.
        inner_internal_bias: See :py:func:`species_derivatives`.
        timestep: Timestep for update.

    Returns:
        Update dictionary.
    """
    assert set(SPECIES) == initial_state.keys()
    steady_state = initial_state.copy()
    # Assume that steady state is reached exclusively through diffusion
    steady_state["periplasm"] = internal_steady_state[0]
    steady_state["cytoplasm"] = internal_steady_state[1]
    steady_state_arr = species_dict_to_array(steady_state, SPECIES_TO_INDEX)

    args = (reaction_params, outer_internal_bias, inner_internal_bias)
    result = solve_ivp(
        lambda t, state_arr, *args: species_derivatives(state_arr, *args),
        [0, timestep],
        steady_state_arr,
        args=args,
    )
    assert result.success
    final_state_arr = result.y[:, -1].T

    initial_state_arr = species_dict_to_array(initial_state, SPECIES_TO_INDEX)
    delta_arr = final_state_arr - initial_state_arr
    delta = species_array_to_dict(delta_arr, SPECIES_TO_INDEX)
    return {
        "species": delta,
    }


def species_array_to_dict(array, species_to_index):
    """Convert an array of values to a map from name to value index.

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
    """
    return {species: array[index] for species, index in species_to_index.items()}


def species_dict_to_array(species_dict, species_to_index):
    """Convert species dict to an array based on an index map.

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
    """
    assert species_dict.keys() == species_to_index.keys()
    array = np.zeros(len(species_dict))
    for species, value in species_dict.items():
        i = species_to_index[species]
        array[i] = value
    return array


class AntibioticTransportSteadyState(Process):
    name = "antibiotic-transport-steady-state"
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
            # Prepare the state for the bioscrape process by moving
            # species from 'external' to 'species' where the bioscrape
            # process expects them to be, renaming variables, and doing
            # unit conversions.
            # NOTE: This is a relic from when we used a bioscrape process
            prepared_state = {
                "species": antibiotic_state["species"],
                "reaction_parameters": antibiotic_state["reaction_parameters"],
            }
            prepared_state, saved_units = remove_units(prepared_state, UNITS)

            # Save initial total count
            periplasm_counts = (
                (
                    prepared_state["species"]["periplasm"]
                    + prepared_state["species"]["hydrolyzed_periplasm"]
                )
                / 1000
                * (
                    N_A
                    * prepared_state["reaction_parameters"]["diffusion"][
                        "periplasm_volume"
                    ]
                )
            )
            cytoplasm_counts = (
                (
                    prepared_state["species"]["cytoplasm"]
                    + prepared_state["species"]["hydrolyzed_cytoplasm"]
                )
                / 1000
                * (
                    N_A
                    * prepared_state["reaction_parameters"]["diffusion"][
                        "cytoplasm_volume"
                    ]
                )
            )
            initial_internal_counts = periplasm_counts + cytoplasm_counts

            # Compute the update.
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
            internal_steady_state = find_steady_state(
                prepared_state["species"]["external"],
                prepared_state["reaction_parameters"],
                outer_internal_bias,
                inner_internal_bias,
            )
            antibiotic_update = update_from_steady_state(
                internal_steady_state,
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
            periplasm_counts = (
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
            cytoplasm_counts = (
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
            final_internal_counts = periplasm_counts + cytoplasm_counts

            # Add units back in
            antibiotic_update["species"] = add_units(
                antibiotic_update["species"],
                saved_units["species"],
                strict=not self.parameters["diffusion_only"],
            )

            antibiotic_update["exchanges"] = {
                "external": -(final_internal_counts - initial_internal_counts)
            }

            update[antibiotic] = antibiotic_update
        return update


def test_antibiotic_transport_steady_state():
    proc = AntibioticTransportSteadyState(
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

    expected_update = {
        "species": {
            "periplasm": 2 * units.mM,
            "hydrolyzed_periplasm": 1 * units.mM,
            "external": 0 * units.mM,
            "cytoplasm": 0 * units.mM,
            "hydrolyzed_cytoplasm": 0 * units.mM,
        },
        "exchanges": {
            # Exchanges are in units of counts, but the species are in
            # units of mM with a volume of 1L.
            "external": (
                -3
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
                ) <= 1e-5 * abs(expected_update[key][species])
        else:
            assert update[key] == expected_update[key]


if __name__ == "__main__":
    test_antibiotic_transport_steady_state()
