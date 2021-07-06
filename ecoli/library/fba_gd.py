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

    def __init__(self, reactions: Optional[Iterable[dict]] = None):
        """Initialize this reaction network.

        Args:
            reactions: a list of reaction dicts following the knowledge base structure. Expected keys are "reaction id",
                "stoichiometry", "is reversible".
        """

        # Defer construction of the stoichiometry matrix until it is needed.
        self._s_matrix = None

        # Build maps between string id and numerical array index, for reactions and molecules.
        self._reactions = {}
        self._reaction_ids = []
        self._reaction_index = {}
        self._molecule_ids = []
        self._molecule_index = {}
        for reaction in reactions:
            self.add_reaction(reaction)

    def add_reaction(self, reaction: dict):
        """Adds a reaction to the network.

        Args:
            reaction: a reaction dict following the knowledge base structure. Expected keys are "reaction id",
                "stoichiometry", "is reversible".
        """
        reaction_id = reaction["reaction id"]
        stoichiometry = reaction["stoichiometry"]

        self._reactions[reaction_id] = stoichiometry
        self._reaction_index[reaction_id] = len(self._reaction_ids)
        self._reaction_ids.append(reaction_id)
        for molecule_id in stoichiometry:
            if molecule_id not in self._molecule_index:
                self._molecule_index[molecule_id] = len(self._molecule_ids)
                self._molecule_ids.append(molecule_id)
        # Force reconstruction of the stoichiometry matrix.
        self._s_matrix = None

    @property
    def s_matrix(self) -> np.ndarray:
        """The 2D stoichiometry matrix describing this reaction network."""
        if self._s_matrix is None:
            s_matrix = np.zeros(self.shape)
            for reaction_id, stoichiometry in self._reactions.items():
                for molecule_id, coeff in stoichiometry.items():
                    s_matrix[self._molecule_index[molecule_id], self._reaction_index[reaction_id]] += coeff
            self._s_matrix = s_matrix
        return self._s_matrix

    @property
    def shape(self) -> Tuple[int, int]:
        """The 2D shape of this network, (#molecules, #reactions)."""
        return len(self._molecule_ids), len(self._reaction_ids)

    def reaction_ids(self) -> Iterator[str]:
        """Iterates through reaction_ids in their indexed order."""
        for reaction_id in self._reaction_ids:
            yield reaction_id

    def reaction_index(self, reaction_id: str) -> Optional[int]:
        """The index of the reaction_id, or None if it is not part of the network."""
        return self._reaction_index.get(reaction_id, None)

    def reaction_vector(self, data: Mapping[str, float], default: float = 0) -> np.ndarray:
        """Converts a dict of {reaction_id: value} to a 1D vector for numpy ops."""
        return np.array([data.get(reaction_id, default) for reaction_id in self._reaction_ids])

    def reaction_values(self, values: Iterable[float]) -> Mapping[str, float]:
        """Converts an array of values to a {reaction_id: value} dict."""
        return {reaction_id: value for reaction_id, value in zip(self._reaction_ids, values)}

    def molecule_ids(self) -> Iterator[str]:
        """Iterates through molecule_ids in their indexed order."""
        for molecule_id in self._molecule_ids:
            yield molecule_id

    def molecule_index(self, molecule_id: str) -> Optional[int]:
        """The index of the molecule_id, or None if it is not part of the network."""
        return self._molecule_index.get(molecule_id, None)

    def molecule_vector(self, data: Mapping[str, float], default: float = 0) -> np.ndarray:
        """Converts a dict of {molecule_id: value} to a 1D vector for numpy ops."""
        return np.array([data.get(molecule_id, default) for molecule_id in self._molecule_ids])

    def molecule_values(self, values: Iterable[float]) -> Mapping[str, float]:
        """Converts an array of values to a {molecule_id: value} dict."""
        return {molecule_id: value for molecule_id, value in zip(self._molecule_ids, values)}


class SteadyStateResidual:
    """Calculates the deviation of the system from steady state, for network intermediates."""

    def __init__(self, network: ReactionNetwork, exchanges: Iterable[str]):
        self.network = network
        self.param_key = None
        exchanges = set(exchanges)
        indices = []
        for molecule_id in self.network.molecule_ids():
            if molecule_id not in exchanges:
                indices.append(self.network.molecule_index(molecule_id))
        self._intermediate_indices = np.array(indices)

    def __call__(self, velocities: jnp.ndarray, dm_dt: jnp.ndarray, params: Mapping[str, Any]) -> jnp.ndarray:
        del (velocities)
        del (params)
        return dm_dt[self._intermediate_indices]


class DmdtTargetResidual:
    """Calculates the deviation from target rates of change (dm/dt) for specified molecules."""

    def __init__(self, network: ReactionNetwork, molecule_ids: Iterable[str], param_key: str):
        self.network = network
        self.param_key = param_key
        self._molecule_indices = np.array([self.network.molecule_index(molecule_id) for molecule_id in molecule_ids])

    def __call__(self, velocities: jnp.ndarray, dm_dt: jnp.ndarray, params: Mapping[str, Any]) -> jnp.ndarray:
        del (velocities)
        targets = jnp.asarray(params[self.param_key])
        return (dm_dt - targets)[self._molecule_indices]


class KineticTargetResidual:
    """Calculates the deviation from target velocities for specified reactions."""

    def __init__(self, network: ReactionNetwork, reaction_ids: Iterable[str], param_key: str):
        self.network = network
        self.param_key = param_key
        self._reaction_indices = np.array([self.network.reaction_index(reaction_id) for reaction_id in reaction_ids])

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
                 reactions: Iterable[dict],
                 exchanges: Iterable[str],
                 objective: Iterable[str],
                 objectiveType: Optional[str] = None,
                 objectiveParameters: Optional[Mapping[str, Any]] = None):
        """Initialize this FBA solver.

        Args:
            reactions: a list of reaction dicts following the knowledge base structure. Expected keys are "reaction id",
                "stoichiometry", "is reversible".
            exchanges: ids of molecules on the boundary, which may flow in or out of the system.
            objective: ids of molecules included in the (homeostatic) objective.
            objectiveType: "homeostatic", "kinetic_only", or "homeostatic_kinetics_mixed".
            objectiveParameters: additional objective-specific parameters. Currently supported is "reactionRateTargets",
                for kinetic objectives.
        """
        # Implementation Note: this is a prototype implementation with an API roughly following
        # modular_fba.FluxBalanceAnalysis, focusing on objectiveTypes currently used in metabolism.py.
        # The details could be refactored quite a lot, depending on what is learned from the prototype.

        self.network = ReactionNetwork(reactions)
        lb = []
        ub = []
        for reaction in reactions:
            if reaction["is reversible"]:
                lb.append(-np.inf)
                ub.append(np.inf)
            else:
                lb.append(0)
                ub.append(np.inf)
        self._bounds = (np.array(lb), np.array(ub))

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
              reaction_flux_bounds: Optional[Mapping[str, float]] = None,
              initial: Optional[Mapping[str, float]] = None,
              rng_seed: int = None) -> FbaResult:
        """Performs the optimization to solve the specified FBA problem.

        Args:
            objective: dict mapping metabolite_id -> homeostatic objective, in units of concentration per time. Values
                should be included for all objective keys used to initialize this solver.
            params: dict with additional parameter values, depending on the problem to be solved.
                'kinetic_targets': dict mapping reaction_id -> velocity, as current kinetic targets per reaction.
            reaction_flux_bounds: (optional) dict mapping reaction_id -> (lb, ub) providing explicit bounds on reaction
                velocity (flux). Bounds default to (-inf, +inf) for reversible or (0, +inf) for irreversible reactions.
                Bounds provided here will override these defaults, including for instance allowing reversibility.
            initial: (optional) dict mapping reaction_id -> velocity as a starting point for optimization. For
                repeated solutions with evolving objective targets, starting from the previous solution can improve
                performance. If None, a random starting point is used.
            rng_seed: (optional) seed for the random number generator, when randomizing the starting point. Provided
                as an arg to support reproducibility; if None then a suitable seed is chosen.

        Returns:
            FbaResult containing optimized reaction velocities, and resulting rate of change per metabolite (dm/dt).
        """
        # Set up x0 with or without random variation, and truncate to bounds.
        if initial is not None:
            x0 = jnp.asarray(self.network.reaction_vector(initial))
        else:
            # Random starting point.
            if rng_seed is None:
                rng_seed = int(time.time() * 1000)
            num_reactions = self.network.shape[1]
            x0 = jax.random.uniform(jax.random.PRNGKey(rng_seed), (num_reactions,))

        # Update bounds, and enforce on x0.
        lb = np.array(self._bounds[0])
        ub = np.array(self._bounds[1])
        if reaction_flux_bounds is not None:
            for reaction_id,  (reaction_lb, reaction_ub) in reaction_flux_bounds.items():
                i = self.network.reaction_index(reaction_id)
                lb[i] = reaction_lb
                ub[i] = reaction_ub
        x0 = np.maximum(lb, np.minimum(ub, x0))

        # Put all parameters into jax arrays, passed as side arguments to the final residual function.
        ready_params = {}
        if self._homeostatic_objective:
            key = self._homeostatic_objective.param_key
            ready_params[key] = jnp.asarray(self.network.molecule_vector(objective))
        if self._kinetic_objective:
            key = self._kinetic_objective.param_key
            ready_params[key] = jnp.asarray(self.network.reaction_vector(params[key]))

        fn = jax.jit(lambda v: self.residual(v, ready_params))
        jac = jax.jacfwd(fn)

        # Perform the actual gradient descent, and extract the result.
        soln = scipy.optimize.least_squares(fn, x0, jac=jac, bounds=(lb, ub))
        dm_dt = self.network.s_matrix @ soln.x
        ss_residual = self._steady_state_objective(soln.x, dm_dt, ready_params)
        return FbaResult(seed=rng_seed,
                         velocities=self.network.reaction_values(soln.x),
                         dm_dt=self.network.molecule_values(dm_dt),
                         ss_residual=ss_residual)
