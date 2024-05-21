"""
==================
Membrane Potential
==================

`MembranePotential` calculates membrane potential and the proton motive force by
taking into account all of the ions that are permeant through the membrane.

Goldman equation for membrane potential:
 :math:`d_{V} = \\frac{RT}{F} * \\ln{\\frac{\\sum_{i}^{n} P_{M^{+}_{i}}[M^{+}_{i}]_{out} +
 \\sum_{j}^{m} P_{A^{-}_{j}}[A^{-}_{j}]_{in}}{\\sum_{i}^{n} P_{M^{+}_{i}}[M^{+}_{i}]_{in} +
 \\sum_{j}^{m} P_{A^{-}_{j}}[A^{-}_{j}]_{out} }}`

  * :math:`D`: Diffusion constant
  * :math:`M` monovalent positive ionic species
  * :math:`A` negative ionic species

transmembrane pH difference (currently fixed at -50):
 :math:`d_{pH} = -2.3 * k * T / e`

  * :math:`k`: Boltzmann constant
  * :math:`T`: temperature

proton motive force (``PMF``):
 :math:`PMF = d_{V} + d_{pH}`

.. note::
    * expected d_{V} = -120 mV
    * expected d_{pH} = -50 mV for cells grown at pH 7. (Berg, H. "E. coli in motion", pg 105)
    * PMF ~170mV at pH 7. ~140mV at pH 7.7 (Berg)
    * *E. coli* internal pH in range 7.6-7.8 (Berg)
    * (mmol) http://book.bionumbers.org/what-are-the-concentrations-of-different-ions-in-cells/
    * Schultz, Stanley G., and A. K. Solomon. "Cation Transport in Escherichia coli" (1961)
    * Zilberstein, Dan, et al. "Escherichia coli intracellular pH, membrane potential, and cell growth."
      Journal of bacteriology 158.1 (1984): 246-252.
    * TODO -- add Mg2+, Ca2+

"""

import os

import numpy as np
import scipy.constants as constants

from vivarium.core.process import Process
from vivarium.core.composition import (
    simulate_process,
    PROCESS_OUT_DIR,
)
from vivarium.plots.simulation_output import plot_simulation_output


NAME = "membrane_potential"


class NoChargeError(Exception):
    pass


class MembranePotential(Process):
    """Membrane Potential

    :term:`Ports`:
        * ``internal``: holds the concentrations of internal ions
        * ``external``: holds the concentrations of external ions
        * ``membrane``: holds the cross-membrane properties 'PMF', 'd_{V}', 'd_{pH}'
    """

    name = NAME
    defaults = {
        "initial_state": {
            "internal": {
                "K": 300,  # (mmol) 30-300
                "Na": 10,  # (mmol) 10
                "Cl": 10,
                "PROTON": 0,
            },  # (mmol) 10-200 media-dependent
            "external": {
                "K": 5,
                "Na": 145,
                "Cl": 110,  # (mmol)
                "PROTON": 0,
                "T": 310.15,
            },
        },
        "permeability_map": {
            "K": "p_K",
            "Na": "p_Na",
            "Cl": "p_Cl",
            "PROTON": "p_PROTON",
        },
        # cation is positively charged, anion is negatively charged
        "charge_map": {
            "K": "cation",
            "Na": "cation",
            "Cl": "anion",
            "PROTON": "cation",
        },
        # parameters
        "p_K": 1,  # unitless, relative membrane permeability of K
        "p_Na": 0.05,  # unitless, relative membrane permeability of Na
        "p_Cl": 0.05,  # unitless, relative membrane permeability of Cl
        "p_PROTON": 1,  # should be more?
        # physical constants
        "R": constants.gas_constant,  # (J * K^-1 * mol^-1) gas constant
        "F": constants.physical_constants["Faraday constant"][
            0
        ],  # (C * mol^-1) Faraday constant
        "k": constants.Boltzmann,  # (J * K^-1) Boltzmann constant
        "e": 1,  # proton charge
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

    def ports_schema(self):
        """
        declare schema for ports ``internal``, ``external`` and ``membrane``.
        """
        ports = [
            "internal",
            "membrane",
            "external",
        ]
        schema = {port: {} for port in ports}

        ## internal
        # internal ions and charge (c_in)
        for state_id, value in self.parameters["initial_state"]["internal"].items():
            schema["internal"][state_id] = {
                "_default": value,
                "_emit": True,
            }

        ## external
        # external ions, charge (c_out) and temperature (T)
        for state_id, value in self.parameters["initial_state"]["external"].items():
            schema["external"][state_id] = {
                "_default": value,
                "_emit": True,
            }

        ## membrane
        # proton motive force (PMF), electrical difference (d_V), pH difference (d_pH)
        for state in ["PMF", "d_V", "d_pH"]:
            schema["membrane"][state] = {
                "_default": 0.0,
                "_updater": "set",
                "_emit": True,
            }
        return schema

    def next_update(self, timestep, states):
        internal_state = states["internal"]
        external_state = states["external"]

        # parameters
        R = self.parameters["R"]
        F = self.parameters["F"]
        # k = self.parameters["k"]
        # e = self.parameters["e"]

        # state
        T = external_state["T"]  # temperature

        # Membrane potential.
        numerator = 0
        denominator = 0
        for ion_id, p_ion_id in self.parameters["permeability_map"].items():
            charge = self.parameters["charge_map"][ion_id]
            p_ion = self.parameters[p_ion_id]

            # ions states
            internal = internal_state[ion_id]
            external = external_state[ion_id]

            if charge == "cation":
                numerator += p_ion * external
                denominator += p_ion * internal
            elif charge == "anion":
                numerator += p_ion * internal
                denominator += p_ion * external
            else:
                raise NoChargeError("No charge given for {}".format(ion_id))

        # Goldman equation for membrane potential
        # expected d_V = -120 mV
        d_V = (
            (R * T) / F * np.log(numerator / denominator) * 1e3
        )  # (mV). 1e3 factor converts from V

        # transmembrane pH difference
        # -2.3 * k * T / e  # -2.3 Boltzmann constant * temperature
        # expected d_pH = -50 mV
        d_pH = (
            -50
        )  # (mV) for cells grown at pH 7. (Berg, H. "E. coli in motion", pg 105)

        # proton motive force
        PMF = d_V + d_pH

        return {"membrane": {"d_V": d_V, "d_pH": d_pH, "PMF": PMF}}


def test_mem_potential(return_data=False):
    """
    test :module: ecoli.processes.membrane.membrane_potential.MembranePotential
    by running it with changing external Na concentrations.
    """
    parameters = {}
    mp = MembranePotential(parameters)
    timeline = [(0, {("external", "Na"): 1}), (50, {("external", "Na"): 10}), (100, {})]

    settings = {"timeline": {"timeline": timeline}}
    timeseries = simulate_process(mp, settings)

    PMF_timeseries = timeseries["membrane"]["PMF"]
    assert PMF_timeseries[-1] > PMF_timeseries[2]

    if return_data:
        return timeseries


def main():
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    timeseries = test_mem_potential(return_data=True)
    settings = {"remove_first_timestep": True}
    plot_simulation_output(timeseries, settings, out_dir)


if __name__ == "__main__":
    main()
