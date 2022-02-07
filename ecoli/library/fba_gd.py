"""FBA via gradient descent."""
import abc
import time
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Optional, Tuple, Union

import jax
import jax.ops
import jax.numpy as jnp
import numpy as np
import scipy.optimize
from scipy.sparse import csr_matrix

ArrayT = Union[np.ndarray, jnp.ndarray]


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
        if reactions is not None:
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

    def reaction_vector(self, data: Mapping[str, Any], default: Any = 0) -> np.ndarray:
        """Converts a dict of {reaction_id: value} to a 1D vector for numpy ops."""
        return np.array([data.get(reaction_id, default) for reaction_id in self._reaction_ids])

    def reaction_values(self, values: Iterable[Any]) -> Mapping[str, Any]:
        """Converts an array of values to a {reaction_id: value} dict."""
        return {reaction_id: value for reaction_id, value in zip(self._reaction_ids, values)}

    def molecule_ids(self) -> Iterator[str]:
        """Iterates through molecule_ids in their indexed order."""
        for molecule_id in self._molecule_ids:
            yield molecule_id

    def molecule_index(self, molecule_id: str) -> Optional[int]:
        """The index of the molecule_id, or None if it is not part of the network."""
        return self._molecule_index.get(molecule_id, None)

    def molecule_vector(self, data: Mapping[str, Any], default: Any = 0) -> np.ndarray:
        """Converts a dict of {molecule_id: value} to a 1D vector for numpy ops."""
        return np.array([data.get(molecule_id, default) for molecule_id in self._molecule_ids])

    def molecule_values(self, values: Iterable[Any]) -> Mapping[str, Any]:
        """Converts an array of values to a {molecule_id: value} dict."""
        return {molecule_id: value for molecule_id, value in zip(self._molecule_ids, values)}


class ObjectiveComponent(abc.ABC):
    """Abstract base class for components of an objective function to be optimized."""

    @abc.abstractmethod
    def prepare_targets(self, target_values: Mapping[str, Any]) -> Optional[ArrayT]:
        """Converts a dict of target values into an array, suitable to be passed to residual()."""
        raise NotImplementedError()

    @abc.abstractmethod
    def residual(self, velocities: ArrayT, dm_dt: ArrayT, targets: ArrayT) -> ArrayT:
        """Returns a vector of singular residual values, all to be minimized to achieve the objective."""
        raise NotImplementedError()


class SteadyStateObjective(ObjectiveComponent):
    """Calculates the deviation of the system from steady state, for network intermediates."""

    def __init__(self, network: ReactionNetwork, intermediates: Iterable[str], weight: float = 1.0):
        self.indices = np.array([network.molecule_index(m) for m in intermediates])
        self.weight = weight

    def prepare_targets(self, target_values: Optional[Mapping[str, Any]] = None) -> Optional[ArrayT]:
        """SteadyStateObjective does not use solve-time target values; always returns None."""
        return None

    def residual(self, velocities: ArrayT, dm_dt: ArrayT, targets: Optional[ArrayT] = None) -> ArrayT:
        """Returns the subset of dm/dt affecting intermediates, which should all be zero."""
        return dm_dt[self.indices] * self.weight


class VelocityBoundsObjective(ObjectiveComponent):
    """Penalizes reaction velocities outside of specified bounds."""

    def __init__(self, network: ReactionNetwork, bounds: Mapping[str, Tuple[float, float]], weight: float = 1.0):
        """Initializes the objective with defined upper and lower bounds."""
        self.network = network
        self.indices = np.array([network.reaction_index(r) for r in bounds])
        self.bounds = {reaction_id: (lb, ub) for reaction_id, (lb, ub) in bounds.items()}
        self.weight = weight

    def prepare_targets(self, target_values: Optional[Mapping[str, Any]] = None) -> Optional[ArrayT]:
        """Prepares an array of upper and lower bounds.

        Args:
            target_values: {reaction_id: (lb, ub)} _overriding_ any bounds specified at initialization.

        Returns:
            2D numpy array with shape (2, #targets). Any reaction missing from target_values (including if
            target_values is None) defaults to the bounds specified on initialization.
        """
        if target_values is not None:
            # Copy initialized bounds and update with target values as specified.
            bounds = dict(self.bounds)
            bounds.update(target_values)
        else:
            # Safe to use initialized bounds without copying
            bounds = self.bounds

        return self.network.reaction_vector(bounds, (-np.inf, np.inf))[self.indices].T

    def residual(self, velocities: ArrayT, dm_dt: ArrayT, targets: ArrayT) -> ArrayT:
        """Returns a vector of numbers, zero within bounds, negative for below lb, or positive for above ub."""
        lb, ub = targets
        shortfall = jnp.minimum(0, velocities[self.indices] - lb)
        excess = jnp.maximum(0, velocities[self.indices] - ub)
        return shortfall + excess


class TargetDmdtObjective(ObjectiveComponent):
    """Calculates the deviation from target rates of change (dm/dt) for specified molecules."""

    def __init__(self, network: ReactionNetwork, target_molecules: Iterable[str], weight: float = 1.0):
        self.network = network
        self.indices = np.array([network.molecule_index(m) for m in target_molecules])
        self.weight = weight

    def prepare_targets(self, target_values: Mapping[str, Any]) -> Optional[ArrayT]:
        """Converts a dict {molecule_id: dmdt} into a vector of target values."""
        return self.network.molecule_vector(target_values)[self.indices]

    def residual(self, velocities: ArrayT, dm_dt: ArrayT, targets: ArrayT) -> ArrayT:
        """Returns the excess or shortfall of the actual dm/dt vs the target, for all target molecules."""
        return (dm_dt[self.indices] - targets) * self.weight


class TargetVelocityObjective(ObjectiveComponent):
    """Calculates the deviation from target velocities for specified reactions."""

    def __init__(self, network: ReactionNetwork, target_reactions: Iterable[str], weight: float = 1.0):
        self.network = network
        self.indices = np.array([network.reaction_index(r) for r in target_reactions])
        self.weight = weight

    def prepare_targets(self, target_values: Mapping[str, Any]) -> Optional[ArrayT]:
        """Converts a dict {reaction_id: velocity} into a vector of target values."""
        return self.network.reaction_vector(target_values)[self.indices]

    def residual(self, velocities: ArrayT, dm_dt: ArrayT, targets: ArrayT) -> ArrayT:
        """Returns the excess or shortfall of the actual velocity vs the target, for all target reactions."""
        return (velocities[self.indices] - targets) * self.weight


@dataclass
class CooSparseMatrix:
    """Basic implementation of Coordinate-Format (COO) sparse matrix-vector multiplication with JAX primitives.

    COO format uses three vectors containing the value, row index, and column index respectively for all non-zero
    elements in a matrix. All three have length NNZ (Number of Non-Zeros), sorted first by row then by column.

    Attrs:
      shape: (#rows, #cols)
      data: vector of non-zero values
      rows: row index for each value in data
      cols: column index for each value in data
    """
    shape: Tuple[int, int]
    data: jnp.ndarray
    rows: jnp.ndarray
    cols: jnp.ndarray

    def __matmul__(self, other: jnp.ndarray) -> jnp.ndarray:
        """Implements the @ operator."""
        return self.dot(other)

    def dot(self, other: jnp.ndarray) -> jnp.ndarray:
        """Sparse matrix by dense vector multiplication."""
        terms = jnp.multiply(self.data, other[self.cols])
        return jax.ops.segment_sum(terms, self.rows, num_segments=self.shape[0], indices_are_sorted=True)


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
                 target_metabolites: Iterable[str]):
        """Initialize this FBA solver.

        Args:
            reactions: a list of reaction dicts following the knowledge base structure. Expected keys are "reaction id",
                "stoichiometry", "is reversible".
            exchanges: ids of molecules on the boundary, which may flow in or out of the system.
            target_metabolites: ids of molecules with production targets.
        """
        exchanges = set(exchanges)
        target_metabolites = set(target_metabolites)

        # Iterate once through the list of reactions
        network = ReactionNetwork()
        irreversible_reactions = []
        for reaction in reactions:
            network.add_reaction(reaction)
            if not reaction["is reversible"]:
                irreversible_reactions.append(reaction["reaction id"])

        self.network = network
        # Build a sparse copy of the S matrix for optimized matrix-vector multiplication.
        rows, cols = jnp.nonzero(network.s_matrix)
        self._s_sparse = CooSparseMatrix(network.s_matrix.shape, jnp.asarray(network.s_matrix[rows, cols]), rows, cols)

        # All FBA problems have a steady-state objective, for all intermediates.
        self._objectives = {}
        self.add_objective("steady-state",
                           SteadyStateObjective(network,
                                                (m for m in network.molecule_ids()
                                                 if m not in exchanges and m not in target_metabolites)))
        # Apply any reversibility constraints with a bounds objective.
        if irreversible_reactions:
            self.add_objective("irreversibility",
                               VelocityBoundsObjective(network,
                                                       {reaction_id: (0, np.inf)
                                                        for reaction_id in irreversible_reactions}))

    def add_objective(self, objective_id: str, objective: ObjectiveComponent):
        self._objectives[objective_id] = objective

    def residuals(self, velocities: ArrayT, objective_targets: Mapping[str, ArrayT]) -> Mapping[str, ArrayT]:
        """Calculates the residual for each component of the overall objective function.

        Args:
            velocities: vector of velocities (rates) for all reactions in the network.
            objective_targets: dict of target value vectors for each objective component. The shape and values of these
                targets depend on the individual objectives. Missing are permitted, if the individual objective accepts
                None.

        Returns:
            A dict of residual vectors, supplied by each objective component.
        """
        dm_dt = self._s_sparse @ velocities

        residuals = {}
        for objective_id, objective in self._objectives.items():
            targets = objective_targets.get(objective_id, None)
            residuals[objective_id] = objective.residual(velocities, dm_dt, targets)
        return residuals

    def solve(self,
              objective_targets: Mapping[str, Mapping[str, float]],
              initial_velocities: Optional[Mapping[str, float]] = None,
              rng_seed: int = None,
              **kwargs) -> FbaResult:
        """Performs the optimization to solve the specified FBA problem.

        Args:
            objective_targets: {objective_id: {key: value}} for each objective component. The details of these targets
                depend on the individual objectives. Missing targets are permitted, if the individual objective accepts
                None.
            initial_velocities: (optional) {reaction_id: velocity} as a starting point for optimization. For repeated
                solutions with evolving objective targets, starting from the previous solution can improve performance.
                If None, a random starting point is used.
            rng_seed: (optional) seed for the random number generator, when randomizing the starting point. Provided
                as an arg to support reproducibility; if None then a suitable seed is chosen.
            kwargs: Any additional keyword arguments will be passed through to scipy.optimize.least_squares.

        Returns:
            FbaResult containing optimized reaction velocities, and resulting rate of change per metabolite (dm/dt).
        """
        # Set up x0 with or without random variation, and truncate to bounds.
        if initial_velocities is not None:
            x0 = jnp.asarray(self.network.reaction_vector(initial_velocities))
        else:
            # Random starting point.
            if rng_seed is None:
                rng_seed = int(time.time() * 1000)
            num_reactions = self.network.shape[1]
            x0 = jax.random.uniform(jax.random.PRNGKey(rng_seed), (num_reactions,))

        target_values = {}
        for objective_id, objective in self._objectives.items():
            targets = objective.prepare_targets(objective_targets.get(objective_id))
            if targets is not None:
                target_values[objective_id] = jnp.asarray(targets)

        # Overall residual is a flattened vector of the (weighted) residuals of individual objectives.
        def loss(v):
            return jnp.concatenate(list(self.residuals(v, target_values).values()))

        jac = jax.jit(jax.jacfwd(loss))
        jac_wrap = lambda x: csr_matrix(jac(x))

        # Perform the actual gradient descent, and extract the result.
        soln = scipy.optimize.least_squares(jax.jit(loss), x0, jac=jac_wrap, **kwargs)

        # Perform the actual gradient descent, and extract the result.
        #soln = scipy.optimize.least_squares(jax.jit(loss), x0, jac=jax.jit(jax.jacfwd(loss)), **kwargs)
        dm_dt = self._s_sparse @ soln.x
        ss_residual = self._objectives["steady-state"].residual(soln.x, dm_dt, None)
        return FbaResult(seed=rng_seed,
                         velocities=self.network.reaction_values(soln.x),
                         dm_dt=self.network.molecule_values(dm_dt),
                         ss_residual=ss_residual)
