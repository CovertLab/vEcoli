"""FBA via gradient descent."""
from typing import Any, Iterable, Mapping, Optional

import jax
import jax.numpy as jnp

import numpy as np
import scipy.optimize


class GradientDescentFba:
    def __init__(self,
                 reactionStoich: Mapping[str, Mapping[str, float]],
                 exchangeMolecules: Iterable[str],
                 objective: Mapping[str, float],
                 objectiveType: Optional[str] = None,
                 objectiveParameters: Optional[Mapping[str, Any]] = None):
        # Implementation Note: this is a prototype implementation with an API roughly following
        # modular_fba.FluxBalanceAnalysis, focusing on objectiveTypes currently used in metabolism.py.
        # The details could be refactored quite a lot, depending on what is learned from the prototype.

        # First pass maps between string id and numerical array index, for reactions and molecules.
        self._reactionIDs = []
        self._reaction_index = {}
        self._moleculeIDs = []
        self._molecule_index = {}
        for reactionID, stoichiometry in reactionStoich.items():
            self._reaction_index[reactionID] = len(self._reactionIDs)
            self._reactionIDs.append(reactionID)
            for moleculeID in stoichiometry:
                if moleculeID not in self._molecule_index:
                    self._molecule_index[moleculeID] = len(self._moleculeIDs)
                    self._moleculeIDs.append(moleculeID)

        # Second pass builds the S matrix, as a standard (dense) 2D numpy array
        s_matrix = np.zeros((len(self._moleculeIDs), len(self._reactionIDs)))
        for reactionID, stoichiometry in reactionStoich.items():
            for moleculeID, stoichCoeff in stoichiometry.items():
               s_matrix[self._molecule_index[moleculeID], self._reaction_index[reactionID]] += stoichCoeff
        self._s_matrix = s_matrix

        # Identify intermediates, whose rate of change should be 0 at steady state.
        self._exchanges = set(exchangeMolecules)
        self._objective_components = set(objective.keys())
        indices = []
        for moleculeID in self._moleculeIDs:
            if moleculeID not in self._exchanges and moleculeID not in self._objective_components:
                indices.append(self._molecule_index[moleculeID])
        self._intermediate_indices = np.array(indices)

        self.objectiveType = objectiveType or "homeostatic"
        if self.objectiveType == "homeostatic":
            self._homeostatic_objective = True
            self._kinetic_objective = False
        elif self.objectiveType == "kinetic_only":
            self._homeostatic_objective = False
            self._kinetic_objective = True
        elif self.objectiveType == "homeostatic_kinetics_mixed":
            self._homeostatic_objective = False
            self._kinetic_objective = True
        else:
            raise ValueError(f"Unrecognized self.objectiveType: {self.objectiveType}")

        if self._homeostatic_objective:
            indices = []
            targets = []
            for moleculeID, coeff in objective.items():
                indices.append(self._molecule_index[moleculeID])
                targets.append(coeff)
            self._homeostatic_indices = np.array(indices)
            self._homeostatic_targets = np.array(targets)

        if self._kinetic_objective:
            indices = []
            targets = []
            for reactionID, rate in objectiveParameters["reactionRateTargets"].items():
                indices.append(self._reaction_index[reactionID])
                targets.append(rate)
            self._kinetic_indices = np.array(indices)
            self._kinetic_targets = np.array(targets)

    def steady_state_residual(self, rates_of_change: jnp.ndarray) -> jnp.ndarray:
        """Residual vector penalizing intermediates not at steady state."""
        return rates_of_change[self._intermediate_indices]

    def homeostatic_residual(self, rates_of_change: jnp.ndarray) -> jnp.ndarray:
        """Residual vector penalizing deviation from homeostatic targets."""
        return rates_of_change[self._homeostatic_indices] - self._homeostatic_targets

    def kinetic_residual(self, reaction_flux: jnp.ndarray) -> jnp.ndarray:
        """Residual vector penalizing deviation of reaction fluxes from kinetic targets."""
        return reaction_flux[self._kinetic_indices] - self._kinetic_targets

    def residual(self, reaction_flux: jnp.ndarray) -> jnp.ndarray:
        """Combines loss components into a single residual vector."""
        rates_of_change = self._s_matrix @ reaction_flux
        residuals = [self.steady_state_residual(rates_of_change)]
        if self._homeostatic_objective:
            residuals.append(self.homeostatic_residual(rates_of_change))
        if self._kinetic_targets:
            residuals.append(self.kinetic_residual(reaction_flux))
        return jnp.concatenate(residuals)

    def solve(self):
        bounds = [(0, np.inf)] * self._s_matrix.shape[1]
        fn = jax.jit(self.residual)
        jac = jax.jit(jax.jacfwd(self.residual))

        prng = jax.random.PRNGKey(0)
        x0 = jax.random.uniform(prng, (self._s_matrix.shape[1],))
        soln = scipy.optimize.least_squares(fn, x0, jac=jac, bounds=bounds)
