
"""
Basic building blocks for `map-reduce`_ pipelines.

This module defines `closures`_ for parallel/asynchronous *maps* (with hashable
keys) and abstract/concrete *reduce* operations. Backend-specific pipeline
implementations, which use these building blocks, should be placed into separate
modules, e.g., :py:mod:`~ecoli.analysis.xarray_emitter.zarr_mapreduce`.

.. _map-reduce: https://en.wikipedia.org/wiki/MapReduce
.. _closures: https://en.wikipedia.org/wiki/Closure_(computer_programming)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from asyncio import Future, TaskGroup
from collections.abc import (
    AsyncGenerator,
    Callable,
    Coroutine,
    Generator,
    Iterable,
    Mapping,
)
from dataclasses import astuple, dataclass
from inspect import isasyncgen, isgenerator
from itertools import chain
from multiprocessing.pool import Pool
from typing import Any, Self, cast

from bottleneck import nanmax, nanmin
from numpy import ndarray, vstack
from numpy.typing import NDArray

# ==============================================================================
# concurrency utils
# ==============================================================================


@dataclass(slots=True, frozen=True)
class PoolDictItem[KeyT, InputT, ResultT]:
    """
    A constructor of `closures`_ intended for *parallel execution* by
    :py:meth:`~multiprocessing.pool.Pool.imap_unordered`.

    This object holds the definition of the computation and of the (unordered)
    output key, while :py:func:`.pool_dict` provides the execution context.
    """

    #: Output key as a function of the input value.
    key: Callable[[InputT], KeyT]
    #: Output value as a function of the input value.
    closure: Callable[[InputT], ResultT]

    def __call__(self, arg: InputT) -> tuple[KeyT, ResultT]:
        """
        Execute the closure for a given input value.
        """
        return (self.key(arg), self.closure(arg))


def pool_dict[KeyT, InputT, ResultT](
    worker: PoolDictItem[KeyT, InputT, ResultT],
    args: Iterable[InputT],
    *,
    n_procs: int, n_items: int
) -> dict[KeyT, ResultT]:
    """
    Create :py:class:`.PoolDictItem` closures for a concrete set of input
    values, and execute them *in parallel* using
    :py:meth:`~multiprocessing.pool.Pool.imap_unordered`.
    """
    assert isinstance(worker, PoolDictItem)
    assert isinstance(n_procs, int) and n_procs > 0
    if n_procs == 1:
        return dict(map(worker, args))
    else:
        with Pool(processes=n_procs) as pool:
            return dict(pool.imap_unordered(worker, args,
                                            chunksize=n_items // n_procs))


# ------------------------------------------------------------------------------


type AnyGenerator[ResultT] = Generator[ResultT] | AsyncGenerator[ResultT]


@dataclass(slots=True, frozen=True)
class AsyncDictItem[KeyT, ResultT]:
    """
    A constructor of `closures`_ intended for *asynchronous execution* by a
    :py:class:`~asyncio.TaskGroup`.

    This object holds the definition of the computation and of the (unordered)
    output key, while :py:func:`.async_dict` provides the execution context.
    """

    #: Output key.
    key: KeyT
    #: :py:class:`asyncio.Task` name.
    name: str
    #: Coroutine producing the output value.
    closure: Coroutine[None, None, ResultT]

    async def item(self) -> tuple[KeyT, ResultT]:
        """
        Associate the closure with its output key.
        """
        return (self.key, await self.closure)

    def create_task(self, g: TaskGroup) -> Future:
        """
        Assign the closure to a :py:class:`asyncio.TaskGroup`.
        """
        return g.create_task(self.item(), name=self.name)


async def async_dict[KeyT, ResultT](
    items: AnyGenerator[AsyncDictItem[KeyT, ResultT]]
) -> dict[KeyT, ResultT]:
    r"""
    Execute a set of :py:class:`.AsyncDictItem`\ s in a
    :py:class:`~asyncio.TaskGroup`.
    """
    async with TaskGroup() as g:
        if isgenerator(items):
            futures = [i.create_task(g) for i in items]
        elif isasyncgen(items):
            futures = [i.create_task(g) async for i in items]
        else:
            raise TypeError(str(type(items)))
    return dict(f.result() for f in futures)


# ==============================================================================
# reduce operators
# ==============================================================================


@dataclass(slots=True, frozen=True)
class Reducer(ABC):
    """
    A wrapper type for reduce operations inside a `map-reduce`_ pipeline.
    """

    @classmethod
    @abstractmethod
    def reduce(cls, args: list[Self], /) -> Self:
        """
        Apply the reduce operation to a set of instances of this type.
        """
        ...

    @abstractmethod
    def extract(self) -> Any:
        """
        Extract the reduced value from the wrapper type.
        """
        ...


# ------------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ReducerList(Reducer):
    """
    A composite reducer, with component reducers indexed by an integer.
    """

    #: Component reducers.
    reducers: list[Reducer]

    def __post_init__(self) -> None:
        assert isinstance(self.reducers, list)
        assert all(isinstance(r, Reducer) for r in self.reducers)

    @classmethod
    def reduce(cls, args: list[Self], /) -> Self:
        reducers = [
            [a.reducers[i] for a in args]
            for i in range(len(args[0].reducers))]
        return cls([type(r[0]).reduce(r) for r in reducers])

    def extract(self) -> list[Any]:
        return [r.extract() for r in self.reducers]


@dataclass(slots=True, frozen=True)
class ReducerMapping[KeyT](Reducer):
    """
    A composite reducer, with component reducers indexed by a hashable key.
    """

    #: Component reducers.
    reducers: Mapping[KeyT, Reducer]

    def __post_init__(self) -> None:
        assert isinstance(self.reducers, dict)
        assert all(isinstance(r, Reducer) for r in self.reducers.values())

    @classmethod
    def reduce(cls, args: list[Self], /) -> Self:
        reducers: list[list[Reducer]] = [[a.reducers[k] for a in args]
                                         for k in args[0].reducers]
        return cls(dict(zip(args[0].reducers.keys(),
                            (type(r[0]).reduce(r) for r in reducers))))

    def extract(self) -> dict[KeyT, Any]:
        return {k: r.extract() for (k, r) in self.reducers.items()}


@dataclass(slots=True, frozen=True)
class ReducerDistribute[KeyT1, KeyT2](ReducerMapping):
    """
    A composite reducer, with two nested levels of :py:class:`ReducerMapping`
    that can be `distributed`_/exchanged.

    This is a useful operation for explicitly translating between different data
    hierarchies, e.g., when the input hierarchy is optimised for parallel
    computation, but is transposed relative to the desired output hierarchy.

    .. _distributed: https://en.wikipedia.org/wiki/Distributive_property
    """

    #: Component reducers.
    reducers: Mapping[KeyT1, ReducerMapping[KeyT2]]

    def __post_init__(self) -> None:
        assert isinstance(self.reducers, dict)
        assert all(isinstance(r, ReducerMapping) for r in self.reducers.values())

    @classmethod
    def reduce(cls, args: list[Self], /) -> Self:
        return cls(cast(
            Mapping[KeyT1, ReducerMapping[KeyT2]],
            ReducerMapping.reduce(
                [ReducerMapping(a.reducers) for a in args]
            ).reducers))

    def distribute(self) -> ReducerDistribute[KeyT2, KeyT1]:
        """
        Exchange the two nested levels of :py:class:`ReducerMapping`, and apply
        :py:meth:`Reducer.reduce` to the output's inner level.
        """
        transpose: dict[KeyT2, dict[KeyT1, list[Reducer]]] = {}
        for (k1, r1) in self.reducers.items():
            for (k2, r2) in r1.reducers.items():
                transpose.setdefault(k2, {}).setdefault(k1, []).append(r2)
        result: dict[KeyT2, dict[KeyT1, Reducer]] = {}
        for (k2, x2) in transpose.items():
            for (k1, x1) in x2.items():
                result.setdefault(k2, {})[k1] = type(x1[0]).reduce(x1)
        return ReducerDistribute[KeyT2, KeyT1]({k2: ReducerMapping[KeyT1](m2)
                                                for (k2, m2) in result.items()})


# ------------------------------------------------------------------------------


@dataclass(kw_only=True, slots=True, frozen=True)
class MinMax(Reducer):
    """
    A primitive reducer that computes the minimum and maximum values across its
    inputs, which are assumed to be 1-D arrays with identical shapes.
    """

    mins: NDArray
    maxs: NDArray

    def __post_init__(self) -> None:
        assert isinstance(self.mins, ndarray)
        assert isinstance(self.maxs, ndarray)
        assert self.mins.ndim == 1
        assert self.maxs.ndim == 1
        assert self.mins.shape == self.maxs.shape

    @classmethod
    def reduce(cls, args: list[Self], /) -> Self:
        return cls(mins=nanmin(vstack([a.mins for a in args]), axis=0),
                   maxs=nanmax(vstack([a.maxs for a in args]), axis=0))

    def extract(self) -> tuple[NDArray, NDArray]:
        return astuple(self)


@dataclass(slots=True, frozen=True)
class Chain[ElemT](Reducer):
    """
    A primitive reducer that simply accumulates its inputs into a list.
    """

    string: list[ElemT]

    def __post_init__(self) -> None:
        assert isinstance(self.string, list)

    @classmethod
    def reduce(cls, args: list[Self], /) -> Self:
        return cls(list(chain.from_iterable(a.string for a in args)))

    def extract(self) -> list[ElemT]:
        return self.string


@dataclass(slots=True, frozen=True)
class Unique[ElemT](Chain):
    """
    A primitive reducer that checks whether all of its inputs are identical.
    """

    def extract(self) -> list[ElemT]:
        """
        Return the unique input element as a singleton list.
        """
        fst = self.string[0]
        assert all(el == fst for el in self.string[1:])
        return [fst]
