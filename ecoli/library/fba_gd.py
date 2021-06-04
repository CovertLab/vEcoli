"""FBA via gradient descent."""
from typing import Any, Iterable, Iterator, Mapping, Optional, Tuple, Union

import jax
import jax.numpy as jnp

import numpy as np
import scipy.optimize


class GradientDescentFba:
    def __init__(self,
                 reactions: Mapping[str, Mapping[str, float]],
                 exchanges: Iterable[str],
                 objective: Mapping[str, float],
                 objectiveType: Optional[str] = None,
                 objectiveParameters: Optional[Mapping[str, Any]] = None):
        # Implementation Note: this is a prototype implementation with an API roughly following
        # modular_fba.FluxBalanceAnalysis, focusing on objectiveTypes currently used in metabolism.py.
        # The details could be refactored quite a lot, depending on what is learned from the prototype.

        self.network = ReactionNetwork(reactions)

        # Identify intermediates, whose rate of change should be 0 at steady state.
        self._exchanges = set(exchanges)
        self._objective_components = set(objective.keys())
        indices = []
        for moleculeID in self.network.moleculeIDs():
            if moleculeID not in self._exchanges and moleculeID not in self._objective_components:
                indices.append(self.network.molecule_index(moleculeID))
        self._intermediate_indices = np.array(indices)

        self.objectiveType = objectiveType or "homeostatic"
        if self.objectiveType == "homeostatic":
            self._homeostatic_objective = True
            self._kinetic_objective = False
        elif self.objectiveType == "kinetic_only":
            self._homeostatic_objective = False
            self._kinetic_objective = True
        elif self.objectiveType == "homeostatic_kinetics_mixed":
            self._homeostatic_objective = True
            self._kinetic_objective = True
        else:
            raise ValueError(f"Unrecognized self.objectiveType: {self.objectiveType}")

        if self._homeostatic_objective:
            self._homeostatic_indices = np.array([self.network.molecule_index(moleculeID) for moleculeID in objective])
        if self._kinetic_objective:
            self._kinetic_indices = np.array(
                [self.network.reaction_index(reactionID)
                 for reactionID in objectiveParameters["reactionRateTargets"]])

    def steady_state_residual(self, dm_dt: jnp.ndarray) -> jnp.ndarray:
        """Residual vector penalizing intermediates not at steady state."""
        return dm_dt[self._intermediate_indices]

    def homeostatic_residual(self, dm_dt: jnp.ndarray, production_targets: jnp.ndarray) -> jnp.ndarray:
        """Residual vector penalizing deviation from homeostatic targets."""
        return dm_dt[self._homeostatic_indices] - production_targets

    def kinetic_residual(self, velocities: jnp.ndarray, kinetic_targets: jnp.ndarray) -> jnp.ndarray:
        """Residual vector penalizing deviation of reaction fluxes from kinetic targets."""
        return velocities[self._kinetic_indices] - kinetic_targets

    def residual(self, velocities: jnp.ndarray, params: Mapping[str, Any]) -> jnp.ndarray:
        """Combines loss components into a single residual vector."""
        dm_dt = self.network.dm_dt(velocities)
        residuals = [self.steady_state_residual(dm_dt)]
        if self._homeostatic_objective:
            residuals.append(self.homeostatic_residual(dm_dt, params["objective"]))
        if self._kinetic_objective:
            residuals.append(self.kinetic_residual(velocities, params["kinetic_targets"]))
        return jnp.concatenate(residuals)

    def solve(self,
              objective: Mapping[str, float],
              params: Mapping[str, Any],
              initial: Optional[Mapping[str, float]] = None,
              variance: Optional[float] = None,
              seed: int = 0) -> Tuple[Mapping[str, float], Mapping[str, float]]:
        processed_params = {}
        if self._homeostatic_objective:
            processed_params["objective"] = self.network.molecule_vector(objective)[self._homeostatic_indices]
        if self._kinetic_objective:
            processed_params["kinetic_targets"] = self.network.reaction_vector(params["kinetic_targets"])[
                self._kinetic_indices]

        if initial is not None:
            x0 = jnp.asarray(self.network.reaction_vector(initial))
        else:
            # Random starting point, ensuring we have some variation.
            x0 = jnp.zeros(self.network.shape[1])
            if variance is None:
                variance = 1.0

        if variance is not None:
            x0 += variance * jax.random.normal(jax.random.PRNGKey(seed), (self.network.shape[1],))

        bounds = [(0, np.inf)] * self.network.shape[1]
        fn = jax.jit(lambda v: self.residual(v, processed_params))
        jac = jax.jacfwd(fn)

        soln = scipy.optimize.least_squares(fn, x0, jac=jac, bounds=bounds)
        velocities = self.network.reaction_values(soln.x)
        dm_dt = self.network.molecule_values(self.network.dm_dt(soln.x))
        return velocities, dm_dt


class ReactionNetwork:
    """General representation of a network of stoichiometric reactions."""

    def __init__(self, reactions: Mapping[str: Mapping[str, float]]):
        # First pass maps between string id and numerical array index, for reactions and molecules.
        self._reactionIDs = []
        self._reaction_index = {}
        self._moleculeIDs = []
        self._molecule_index = {}
        for reactionID, stoichiometry in reactions.items():
            self._reaction_index[reactionID] = len(self._reactionIDs)
            self._reactionIDs.append(reactionID)
            for moleculeID in stoichiometry:
                if moleculeID not in self._molecule_index:
                    self._molecule_index[moleculeID] = len(self._moleculeIDs)
                    self._moleculeIDs.append(moleculeID)

        # Second pass builds the S matrix, as a standard (dense) 2D numpy array
        s_matrix = np.zeros((len(self._moleculeIDs), len(self._reactionIDs)))
        for reactionID, stoichiometry in reactions.items():
            for moleculeID, coeff in stoichiometry.items():
                s_matrix[self._molecule_index[moleculeID], self._reaction_index[reactionID]] += coeff
        self._s_matrix = jnp.asarray(s_matrix)

    @property
    def shape(self):
        """The 2D shape of this network, (#molecules, #reactions)."""
        return self._s_matrix.shape

    def reactionIDs(self) -> Iterator[str]:
        """Iterates through reactionIDs in their indexed order."""
        for reactionID in self._reactionIDs:
            yield reactionID

    def reaction_index(self, reactionID: str) -> Optional[int]:
        """The index of the reactionID."""
        return self._reaction_index.get(reactionID, None)

    def reaction_vector(self, data: Mapping[str, float], default: float = 0) -> np.ndarray:
        """Converts a dict of {reactionID: value} to a 1D vector for numpy ops."""
        return np.array([data.get(reactionID, default) for reactionID in self._reactionIDs])

    def reaction_values(self, values: Iterable[float]) -> Mapping[str, float]:
        """Converts an array of values to a {reactionID: value} dict."""
        return {reactionID: value for reactionID, value in zip(self._reactionIDs, values)}

    def moleculeIDs(self) -> Iterator[str]:
        """Iterates through moleculeIDs in their indexed order."""
        for moleculeID in self._moleculeIDs:
            yield moleculeID

    def molecule_index(self, moleculeID: str) -> Optional[int]:
        """The index of the moleculeID."""
        return self._molecule_index.get(moleculeID, None)

    def molecule_vector(self, data: Mapping[str, float], default: float = 0) -> np.ndarray:
        """Converts a dict of {moleculeID: value} to a 1D vector for numpy ops."""
        return np.array([data.get(moleculeID, default) for moleculeID in self._moleculeIDs])

    def molecule_values(self, values: Iterable[float]) -> Mapping[str, float]:
        """Converts an array of values to a {moleculeID: value} dict."""
        return {moleculeID: value for moleculeID, value in zip(self._moleculeIDs, values)}

    def dm_dt(self, velocities: jnp.ndarray) -> jnp.ndarray:
        """Net rate of change for each molecule, given a vector of reaction velocities."""
        return self._s_matrix @ velocities
