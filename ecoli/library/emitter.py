
"""
Extensions to the :py:class:`~vivarium.core.emitter.Emitter` interface, as used
by :py:class:`.ParquetEmitter` and :py:class:`.XarrayEmitter`.
"""


from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import Future, Executor
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Self
from urllib import parse
from warnings import warn

from vivarium.core.types import HierarchyPath
from vivarium.core.engine import Engine
from vivarium.core.emitter import Emitter


# ==============================================================================


class BlockingExecutor(Executor):

    def __init__(self, *args) -> None:
        assert not len(args)
        super().__init__()

    def submit(self, fn: Callable, /, *args, **kwargs) -> Future:
        """
        Run a function in the current thread, and return a
        :py:class:`~concurrent.futures.Future` that is already done.
        """
        future: Future = Future()
        try:
            result = fn(*args, **kwargs)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        return future

    def shutdown(self, wait=True, *, cancel_futures=False) -> None:
        pass


# ==============================================================================


@dataclass(eq=True, kw_only=True, slots=True)
class StoragePartition:
    """
    Metadata determining the relative storage location for the simulation
    outputs of a single-generation :py:class:`.EcoliSim`, inside a hive
    partition or hierarchical store (see :ref:`parquet_emitter`).
    """

    experiment_id: str
    variant: int
    lineage_seed: int
    generation: int = field(init=False)
    agent_id: str

    def __post_init__(self) -> None:
        assert isinstance(self.experiment_id, str)
        assert isinstance(self.variant, int)
        assert isinstance(self.lineage_seed, int)
        assert isinstance(self.agent_id, str)
        self.generation = len(self.agent_id)
        assert self.generation > 0

    @property
    def parent(self) -> Self:
        """
        Metadata of the mother cell in the same cell lineage.
        """
        return replace(self, agent_id=self.agent_id[:-1])


# ==============================================================================


class BufferedEmitter(Emitter, ABC):
    """
    An extension to the :py:class:`~vivarium.core.emitter.Emitter` interface
    that buffers emitted simulation data before writing it to persistent
    storage. In particular, this interface is used by
    :py:meth:`.EcoliSim.update_experiment` and
    :py:meth:`.EngineProcess.next_update`.

    .. warning::
        :py:meth:`~.finalize` must be explicitly called in a
        ``try ... finally ...`` block around the call to
        :py:meth:`vivarium.core.engine.Engine.update`, in order to ensure that
        all buffered emits are written out when the simulation terminates for
        any reason.
    """

    def __init__(self) -> None:
        """
        .. warning::
            This method should be called **at the end** of a subclass
            ``__init__()``.
        """
        self.finalized: bool = False
        """
        Flag set by :py:meth:`.finalize` after writing the last buffer.
        """

    @abstractmethod
    def reset_emit_flags(
        self, *,
        engine: Engine, agent: HierarchyPath, emit_paths: tuple[HierarchyPath]
    ) -> None:
        """
        Reconfigure the simulation engine to avoid futile data marshalling, by
        suppressing all default emissions and enabling only stores that were
        explicitly requested by this emitter's configuration.

        Called by: :py:meth:`.EcoliSim.run` or
        :py:meth:`.EngineProcess.create_emitter`.
        """
        ...

    def extract_partition(self, metadata: dict[str, Any], /) -> StoragePartition:
        """
        Define the current :py:class:`StoragePartition` from the simulation
        metadata received via :py:meth:`!Engine._emit_configuration`.
        """
        return StoragePartition(
            experiment_id=parse.quote_plus(
                metadata.get("experiment_id", "default")),
            variant=int(metadata.get("variant", 0)),
            lineage_seed=int(metadata.get("lineage_seed", 0)),
            agent_id=metadata.get("agent_id", "1"))

    def finalize(self, *, success: bool = False) -> None:
        """
        Emit the partially filled buffer at the end of a single-generation
        simulation.

        Args:
          success: Indicates whether the simulation reached a
                   :py:exc:`.DivisionDetected` event.
        """
        if self.finalized:
            raise RuntimeError(
                f"`{type(self).__name__}.finalize()` was already called.")
        assert isinstance(success, bool)
        self._finalize(success=success)
        self.finalized = True

    @abstractmethod
    def _finalize(self, *, success: bool) -> None:
        """
        Called by: :py:meth:`.finalize`.
        """
        ...

    def __del__(self) -> None:
        """
        When a successfully initialised :py:class:`.BufferedEmitter` instance is
        destroyed, check that its last batch has been flushed by the simulation
        loop.
        """
        if not getattr(self, "finalized", True):
            warn(f"\n  `{type(self).__name__}.finalize()` was never called.")
