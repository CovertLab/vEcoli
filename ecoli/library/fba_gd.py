"""FBA via gradient descent."""
from dataclasses import dataclass
import time
from typing import Any, Iterable, Iterator, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize


class ReactionNetwork:
    """General representation of a network of stoichiometric reactions."""

    def __init__(self, reactions: Mapping[str, Mapping[str, float]]):
        # Defer construction of the stoichiometry matrix until it is needed.
        self._s_matrix = None

        # Build maps between string id and numerical array index, for reactions and molecules.
        self._reactions = {}
        self._reactionIDs = []
        self._reaction_index = {}
        self._moleculeIDs = []
        self._molecule_index = {}
        for reactionID, stoichiometry in reactions.items():
            self.add_reaction(reactionID, stoichiometry)

    def add_reaction(self, reactionID: str, stoichiometry: Mapping[str, float]):
        """Adds a reaction to the network."""
        self._reactions[reactionID] = stoichiometry
        self._reaction_index[reactionID] = len(self._reactionIDs)
        self._reactionIDs.append(reactionID)
        for moleculeID in stoichiometry:
            if moleculeID not in self._molecule_index:
                self._molecule_index[moleculeID] = len(self._moleculeIDs)
                self._moleculeIDs.append(moleculeID)
        # Force reconstruction of the stoichiometry matrix.
        self._s_matrix = None

    @property
    def s_matrix(self) -> np.ndarray:
        """The 2D stoichiometry matrix describing this reaction network."""
        if self._s_matrix is None:
            s_matrix = np.zeros(self.shape)
            for reactionID, stoichiometry in self._reactions.items():
                for moleculeID, coeff in stoichiometry.items():
                    s_matrix[self._molecule_index[moleculeID], self._reaction_index[reactionID]] += coeff
            self._s_matrix = s_matrix
        return self._s_matrix

    @property
    def shape(self) -> Tuple[int, int]:
        """The 2D shape of this network, (#molecules, #reactions)."""
        return len(self._moleculeIDs), len(self._reactionIDs)

    def reactionIDs(self) -> Iterator[str]:
        """Iterates through reactionIDs in their indexed order."""
        for reactionID in self._reactionIDs:
            yield reactionID

    def reaction_index(self, reactionID: str) -> Optional[int]:
        """The index of the reactionID, or None if it is not part of the network."""
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
        """The index of the moleculeID, or None if it is not part of the network."""
        return self._molecule_index.get(moleculeID, None)

    def molecule_vector(self, data: Mapping[str, float], default: float = 0) -> np.ndarray:
        """Converts a dict of {moleculeID: value} to a 1D vector for numpy ops."""
        return np.array([data.get(moleculeID, default) for moleculeID in self._moleculeIDs])

    def molecule_values(self, values: Iterable[float]) -> Mapping[str, float]:
        """Converts an array of values to a {moleculeID: value} dict."""
        return {moleculeID: value for moleculeID, value in zip(self._moleculeIDs, values)}


class SteadyStateResidual:
    """Calculates the deviation of the system from steady state, for network intermediates."""

    def __init__(self, network: ReactionNetwork, exchanges: Iterable[str]):
        self.network = network
        self.param_key = None
        exchanges = set(exchanges)
        indices = []
        for moleculeID in self.network.moleculeIDs():
            if moleculeID not in exchanges:
                indices.append(self.network.molecule_index(moleculeID))
        self._intermediate_indices = np.array(indices)

    def __call__(self, velocities: jnp.ndarray, dm_dt: jnp.ndarray, params: Mapping[str, Any]) -> jnp.ndarray:
        del (velocities)
        del (params)
        return dm_dt[self._intermediate_indices]


class DmdtTargetResidual:
    """Calculates the deviation from target rates of change (dm/dt) for specified molecules."""

    def __init__(self, network: ReactionNetwork, moleculeIDs: Iterable[str], param_key: str):
        self.network = network
        self.param_key = param_key
        self._molecule_indices = np.array([self.network.molecule_index(moleculeID) for moleculeID in moleculeIDs])

    def __call__(self, velocities: jnp.ndarray, dm_dt: jnp.ndarray, params: Mapping[str, Any]) -> jnp.ndarray:
        del (velocities)
        targets = jnp.asarray(params[self.param_key])
        return (dm_dt - targets)[self._molecule_indices]


class KineticTargetResidual:
    """Calculates the deviation from target velocities for specified reactions."""

    def __init__(self, network: ReactionNetwork, reactionIDs: Iterable[str], param_key: str):
        self.network = network
        self.param_key = param_key
        self._reaction_indices = np.array([self.network.reaction_index(reactionID) for reactionID in reactionIDs])

    def __call__(self, velocities: jnp.ndarray, dm_dt: jnp.ndarray, params: Mapping[str, Any]) -> jnp.ndarray:
        del (dm_dt)
        targets = jnp.asarray(params[self.param_key])
        return (velocities - targets)[self._reaction_indices]


@dataclass
class FbaResult:
    """Reaction velocities and dm/dt for an FBA solution, with metrics."""
    seed: int
    velocities: Mapping[str, float]
    dm_dt: Mapping[str, float]
    ss_residual: np.ndarray


class GradientDescentFba:
    """Solves an FBA problem with kinetic and/or homeostatic objectives, by gradient descent."""

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
        self._exchanges = set(exchanges)
        self._objective_components = set(objective)

        # Declared exchanges and objective components are excluded from steady state requirement.
        self._steady_state_objective = SteadyStateResidual(
            self.network, self._exchanges | self._objective_components)

        self.objectiveType = objectiveType or "homeostatic"
        if self.objectiveType == "homeostatic":
            self._homeostatic_objective = DmdtTargetResidual(self.network, self._objective_components, "objective")
            self._kinetic_objective = None
        elif self.objectiveType == "kinetic_only":
            self._homeostatic_objective = None
            self._kinetic_objective = KineticTargetResidual(
                self.network, objectiveParameters["reactionRateTargets"], "kinetic_targets")
        elif self.objectiveType == "homeostatic_kinetics_mixed":
            self._homeostatic_objective = DmdtTargetResidual(self.network, self._objective_components, "objective")
            self._kinetic_objective = KineticTargetResidual(
                self.network, objectiveParameters["reactionRateTargets"], "kinetic_targets")
        else:
            raise ValueError(f"Unrecognized self.objectiveType: {self.objectiveType}")

    def residual(self, velocities: jnp.ndarray, params: Mapping[str, Any]) -> jnp.ndarray:
        """Combines objective residuals into a single residual vector."""
        dm_dt = self.network.s_matrix @ velocities
        residuals = []
        for objective in (self._steady_state_objective, self._homeostatic_objective, self._kinetic_objective):
            if objective:
                residuals.append(objective(velocities, dm_dt, params))
        return jnp.concatenate(residuals)

    def solve(self,
              objective: Mapping[str, float],
              params: Mapping[str, Any],
              initial: Optional[Mapping[str, float]] = None,
              variance: Optional[float] = None,
              seed: int = None) -> FbaResult:
        """Performs the optimization to solve the specified FBA problem."""
        ready_params = {}
        if self._homeostatic_objective:
            key = self._homeostatic_objective.param_key
            ready_params[key] = jnp.asarray(self.network.molecule_vector(objective))
        if self._kinetic_objective:
            key = self._kinetic_objective.param_key
            ready_params[key] = jnp.asarray(self.network.reaction_vector(params[key]))

        if initial is not None:
            x0 = jnp.asarray(self.network.reaction_vector(initial))
        else:
            # Random starting point, ensuring we have some variation.
            x0 = jnp.zeros(self.network.shape[1])
            if variance is None:
                variance = 1.0

        if variance is not None:
            if seed is None:
                seed = int(time.time())
            x0 += variance * jax.random.normal(jax.random.PRNGKey(seed), (self.network.shape[1],))

        # TODO(fdrusso): Bounds depending on reversibility. Requires we keep track of reversibility.
        fn = jax.jit(lambda v: self.residual(v, ready_params))
        jac = jax.jacfwd(fn)

        soln = scipy.optimize.least_squares(fn, x0, jac=jac)
        dm_dt = dm_dt = self.network.s_matrix @ soln.x
        ss_residual = self._steady_state_objective(soln.x, dm_dt, ready_params)
        return FbaResult(seed=seed,
                         velocities=self.network.reaction_values(soln.x),
                         dm_dt=self.network.molecule_values(dm_dt),
                         ss_residual=ss_residual)
