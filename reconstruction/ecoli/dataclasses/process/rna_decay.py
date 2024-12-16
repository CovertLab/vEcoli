"""
SimulationData for rna decay process
"""

from wholecell.utils import units

import autograd.numpy as anp
from autograd import jacobian
import numpy as np


class RnaDecay(object):
    """RnaDecay"""

    def __init__(self, raw_data, sim_data):
        self._buildRnaDecayData(raw_data, sim_data)

    def _buildRnaDecayData(self, raw_data, sim_data):
        _ = sim_data  # unused
        self.endoRNase_ids = [
            x["endoRnase"] + sim_data.getter.get_compartment_tag(x["endoRnase"])
            for x in raw_data.endoRNases
        ]
        self.kcats = (1 / units.s) * np.array(
            [x["kcat"].asNumber(1 / units.s) for x in raw_data.endoRNases]
        )
        self.stats_fit = {
            "LossKm": 0.0,
            "LossKmOpt": 0.0,
            "RnegKmOpt": 0.0,
            "ResKm": 0.0,
            "ResKmOpt": 0.0,
            "ResEndoRNKm": 0.0,
            "ResEndoRNKmOpt": 0.0,
            "ResScaledKm": 0.0,
            "ResScaledKmOpt": 0.0,
        }

        # store Residuals re-scaled (sensitivity analysis "alpha")
        self.sensitivity_analysis_alpha_residual = {}
        self.sensitivity_analysis_alpha_regular_i_neg = {}

        # store Km's and Residuals re-scaled (sensitivity analysis "kcat EndoRNases")
        self.sensitivity_analysis_kcat = {}
        self.sensitivity_analysis_kcat_res_ini = {}
        self.sensitivity_analysis_kcat_res_opt = {}

        # store Km's from first-order RNA decay
        self.Km_first_order_decay = []

        # store convergence of non-linear Km's (G(km))
        self.Km_convergence = []

    def km_loss_function(self, vMax, rnaConc, kDeg, isEndoRnase, alpha):
        """
        Generates the functions used for estimating the per-RNA affinities (Michaelis-Menten
        constants) to the endoRNAses.

        The goal is to find a set of Michaelis-Menten constants such that the
        endoRNAse-mediated degradation under basal concentrations is consistent
        with the experimentally observed half-lives.

        If ``nonlinear`` is the rate of degradation predicted by Michaelis-Menten kinetics
        and ``linear`` is the rate of degradation from observed half-lives, we want::

                nonlinear - linear = 0

        In reality, there will be residuals ``R_aux = ``nonlinear - linear``. We care about
        the residuals after normalizing by the linear rate:: ``R = nonlinear / linear - 1``.

        In order to turn this into a minimization problem, we define the loss function
        as the squared sum of the residuals. Additionally, to ensure that all Km values
        are positive, the loss function accepts as input the logarithm of the final Km
        values and exponentiates them before calculating the residuals.

        The third-party package Autograd uses autodiff to calculate Jacobians for our loss
        function that can be used during minimization.

        Parameters
        ----------
        vMax (float): The total endoRNase capacity, in units of amount per volume per time.
        rnaConc (np.ndarray): Concentrations of RNAs (amount per volume).
        kDeg (np.ndarray): Experimental degradation rates (per unit time).
        isEndoRnase (np.ndarray): Boolean array indicating endoRNase RNAs.

        Returns
        -------
        tuple:
            A tuple containing the following functions:

            - **L** (*function*): The loss function to minimize.
            - **Lp** (*function*): The Jacobian of the loss function `L`.
            - **residual_f** (*function*): The residual error function.
            - **residual_aux_f** (*function*): The unnormalized residual error function.
        """

        def residual_f(km):
            return vMax / km / kDeg / (1 + anp.sum(rnaConc / km)) - 1

        def residual_aux_f(km):
            return vMax * rnaConc / km / (1 + anp.sum(rnaConc / km)) - kDeg * rnaConc

        def L(log_km):
            km = anp.exp(log_km)
            residual_squared = residual_f(km) ** 2

            # Loss function
            return anp.sum(residual_squared)

        def Lp(km):
            return jacobian(L)(km)

        return (
            L,
            Lp,
            residual_f,
            residual_aux_f,
        )
