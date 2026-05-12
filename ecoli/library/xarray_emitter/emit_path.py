
"""
Structural assumptions made by :py:class:`.XarrayBuffer` about schema paths of
emitted variables.

.. note::
  Adding support for a new metadata provider or a new kind of
  non-:py:class:`~vivarium.core.store.Store` emit path will involve extending
  :py:class:`.EmitPath` and :py:class:`.TreeView`.
"""


from __future__ import annotations

from dataclasses import dataclass, field
from enum import Flag, auto

from vivarium.core.types import HierarchyPath


# ==============================================================================
# emit paths
# ==============================================================================


class EmitPathType(Flag):
    r"""
    Type information for :py:data:`~vivarium.core.types.HierarchyPath`\ s, as
    determined in the constructor of :py:class:`.EmitPath`.
    """

    agent = auto()
    """
    Absolute path prefix until ``agent_id``. Other paths wrapped by
    :py:class:`.EmitPath` are relative to an agent.
    """
    listener = auto()
    """
    Relative path to a ``("listeners", ...)``
    :py:class:`~vivarium.core.store.Store` created by
    :py:func:`~ecoli.library.schema.listener_schema`.
    """
    update = auto()
    r"""
    Relative path to a ``("log_update", ...)``
    :py:class:`~vivarium.core.store.Store` created by
    :py:func:`~ecoli.library.logging_tools.make_logging_process`. The output
    tree from such a store does not map onto a source tree of
    :py:class:`~vivarium.core.store.Store`\ s, and hence does not support
    internal ``emit`` flags.
    """

    # ~~~~~~~~~~~~~~~~~ #

    @property
    def is_agent(self) -> bool:
        return self.agent in self  # type: ignore[operator]

    @property
    def is_listener(self) -> bool:
        return self.listener in self  # type: ignore[operator]

    @property
    def is_update(self) -> bool:
        return self.update in self  # type: ignore[operator]

    @property
    def is_update_listener(self) -> bool:
        return (self.listener | self.update) in self  # type: ignore[operator]


# ------------------------------------------------------------------------------


@dataclass
class EmitPath:
    """
    Wrapper data class for an absolute or relative
    :py:data:`~vivarium.core.types.HierarchyPath` that locates a variable
    emitted by :py:meth:`!vivarium.core.engine.Engine._emit_store_data`. Such a
    path is composed of:

      1. *always* a path to a :py:class:`~vivarium.core.store.Store`,
      2. and *possibly* a suffix path inside a schema dictionary that is emitted
         from (1).
    """

    type: EmitPathType = field(init=False)
    path: HierarchyPath

    def __post_init__(self) -> None:
        assert all(isinstance(p, str) for p in self.path)
        self.path = tuple(self.path)
        self.type = EmitPathType(0)
        if "agents" in self.path:
            assert self.path[0] == "agents"
            assert "agents" not in self.path[1:]
            self.type |= EmitPathType.agent
        if "listeners" in self.path:
            self.type |= EmitPathType.listener
        if "log_update" in self.path:
            assert self.path[0] == "log_update"
            assert "log_update" not in self.path[1:]
            self.type |= EmitPathType.update
        if self.type.is_agent:
            assert not (self.type.is_listener or self.type.is_update)

    # ~~~~~~~~~~~~~~~~~ #

    @property
    def emitting_path(self) -> HierarchyPath:
        """
        Path to the responsible :py:class:`~vivarium.core.store.Store`. This is
        always a prefix of :py:attr:`.path`, and identical to :py:attr:`.path`
        if :py:attr:`.path` points to a :py:class:`~vivarium.core.store.Store`.
        """
        assert not self.type.is_agent
        if self.type.is_update_listener:
            return self.path[:self.path.index("listeners")]
        else:
            return self.path

    @property
    def metadata_path(self) -> HierarchyPath:
        """
        Corresponding path within the result of
        :py:meth:`ecoli.experiments.ecoli_master_sim.EcoliSim.output_metadata`.
        Currently, this is either identical to :py:attr:`.path`, or is the
        suffix of :py:attr:`.path` starting with ``"listeners"``.
        """
        assert not self.type.is_agent
        if self.type.is_listener:
            # access listener metadata
            return self.path[self.path.index("listeners"):]
        else:
            # access process metadata
            return self.path
