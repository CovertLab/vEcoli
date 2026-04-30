
"""
Predicates used by :py:class:`.XarrayTransducer` to decide which *simulation
steps* will produce *emit steps*. Predicates are represented as formulas in
`conjunctive normal form`_ (CNF), where literals are atomic predicates
parametrized by the JSON configuration.

.. _conjunctive normal form: https://en.wikipedia.org/wiki/Conjunctive_normal_form
"""


from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Self, final

from .utils import emitter_arg_error


# ==============================================================================
# abstract atomic predicate
# ==============================================================================


class AtomicEmitPredicate(ABC):
    """
    An atomic predicate type over simulation steps, which is parametrized by a
    JSON configuration.

    Example JSON configuration::

      {"subsample": {"interval": 10}}

    Here, the single top-level key determines the predicate subtype, and the
    single top-level value constitutes its parameters.
    """

    @classmethod
    def build(cls, config: dict[str, Any]) -> AtomicEmitPredicate:
        """
        Instantiate the subclass.
        """
        match config:
            case dict() if len(config) == 1:
                (ty, param), = config.items()
                match ty:
                    case "fixed":
                        return FixedSteps(**param)
                    case "subsample":
                        return SubsampleSteps(**param)
                    case _:
                        raise TypeError(emitter_arg_error(
                            cls, "Unsupported predicate type",
                            f"\"predicate\": [[{config}]]"))
            case _:
                raise TypeError(emitter_arg_error(
                    cls, "Invalid argument", f"\"predicate\": [[{config}]]"))

    @abstractmethod
    def __call__(self, sim_tix: int, t: float, data: dict[str, Any], /) -> bool:
        """
        Evaluate the predicate for a simulation step.

        Args:
          sim_tix: :py:attr:`.XarrayTransducer.sim_tix`.
          t:       Simulation time stamp.
          data:    Input received from :py:meth:`!Engine._emit_store_data`.
        """
        ...


# ==============================================================================
# composite predicates
# ==============================================================================


@dataclass(slots=True, frozen=True)
class DisjunctiveEmitPredicate:
    """
    A disjunctive clause whose literals are atomic predicates.

    Example JSON configuration::

      [...]

    Here, each entry in the JSON array is parsed by
    :py:class:`.AtomicEmitPredicate`.
    """

    atoms: list[AtomicEmitPredicate]

    def __post_init__(self) -> None:
        assert isinstance(self.atoms, list)
        assert all(isinstance(p, AtomicEmitPredicate) for p in self.atoms)

    @classmethod
    def build(cls, config: list[dict[str, Any]]) -> Self:
        if not isinstance(config, list):
            raise TypeError(emitter_arg_error(
                cls, "Invalid argument", f"\"predicate\": [{config}]"))
        return cls(list(map(AtomicEmitPredicate.build, config)))

    @abstractmethod
    def __call__(self, sim_tix: int, t: float, data: dict[str, Any], /) -> bool:
        """
        Evaluate the predicate for a simulation step.

        Calls: :py:meth:`.AtomicEmitPredicate.__call__`.

        Args:
          sim_tix: :py:attr:`.XarrayTransducer.sim_tix`.
          t:       Simulation time stamp.
          data:    Input received from :py:meth:`!Engine._emit_store_data`.
        """
        return any(p(sim_tix, t, data) for p in self.atoms)


# ------------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ConjunctiveEmitPredicate:
    """
    A conjuctive formula over disjunctive clauses.

    Example JSON configuration::

      [...]

    Here, each entry in the JSON array is parsed by
    :py:class:`.DisjunctiveEmitPredicate`.
    """

    clauses: list[DisjunctiveEmitPredicate]

    def __post_init__(self) -> None:
        assert isinstance(self.clauses, list)
        assert all(isinstance(c, DisjunctiveEmitPredicate) for c in self.clauses)

    @classmethod
    def build(cls, config: list[list[dict[str, Any]]]) -> Self:
        if not isinstance(config, list):
            raise TypeError(emitter_arg_error(
                cls, "Invalid argument", f"\"predicate\": {config}"))
        return cls(list(map(DisjunctiveEmitPredicate.build, config)))

    @abstractmethod
    def __call__(self, sim_tix: int, t: float, data: dict[str, Any], /) -> bool:
        """
        Evaluate the predicate for a simulation step.

        Calls: :py:meth:`.DisjunctiveEmitPredicate.__call__`.

        Args:
          sim_tix: :py:attr:`.XarrayTransducer.sim_tix`.
          t:       Simulation time stamp.
          data:    Input received from :py:meth:`!Engine._emit_store_data`.
        """
        return all(c(sim_tix, t, data) for c in self.clauses)


# ==============================================================================
# concrete predicates
# ==============================================================================


@final
@dataclass(kw_only=True, slots=True)
class FixedSteps(AtomicEmitPredicate):
    """
    An atomic predicate which selects a statically known set of simulation
    steps.

    Example JSON configuration::

      {"fixed": {"steps": [0]}}

    Here, ``steps`` is a list of integer-valued simulation steps.
    """

    steps: list[int]

    def __post_init__(self) -> None:
        match self.steps:
            case list(steps) if all(isinstance(s, int) and s >= 0 for s in steps):
                self.steps = sorted(self.steps)
            case steps:
                raise ValueError(emitter_arg_error(
                    self, "Invalid argument",
                    f"\"predicate\": [[{{\"fixed\": {{\"steps\": {steps}}}}}]]"))

    def __call__(self, sim_tix: int, t: float, data: dict[str, Any], /) -> bool:
        if self.steps and (sim_tix == self.steps[0]):
            self.steps.pop(0)
            return True
        else:
            return False

@final
@dataclass(kw_only=True, slots=True, frozen=True)
class SubsampleSteps(AtomicEmitPredicate):
    """
    An atomic predicate which selects a regular time grid.

    Example JSON configuration::

      {"subsample": {"interval": 10}}

    Here, ``interval`` is an integer number of simulation steps.
    """

    interval: int

    def __post_init__(self) -> None:
        match self.interval:
            case int(i) if i >= 1:
                pass
            case i:
                raise ValueError(emitter_arg_error(
                    self, "Invalid argument",
                    f"\"predicate\": [[{{\"subsample\": {{\"interval\": {i}}}}}]]"))

    def __call__(self, sim_tix: int, t: float, data: dict[str, Any], /) -> bool:
        return sim_tix % self.interval == 0
