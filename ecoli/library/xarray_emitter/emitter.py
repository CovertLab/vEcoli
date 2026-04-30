
"""
See :py:mod:`.xarray_emitter`.
"""


from __future__ import annotations

from datetime import datetime
from pathlib import Path
from pprint import pp
from typing import Any, final

from vivarium.core.types import HierarchyPath
from vivarium.core.engine import Engine
from vivarium.core.store import Store

from ..emitter import BufferedEmitter
from .transducer import XarrayTransducer
from .storage import XarrayStoragePartition
from .writer import AsyncBufferWriter
from .utils import emitter_arg_error


# ==============================================================================


@final
class XarrayEmitter(BufferedEmitter):
    """
    Entry point for :py:mod:`.xarray_emitter`.

    This is a wrapper around :py:class:`.XarrayTransducer`, and is mainly
    responsible for:

      - Propagating configurations.
      - Coupling to :py:class:`~vivarium.core.engine.Engine` via the
        :py:class:`.BufferedEmitter` interface.
      - Connecting :py:class:`.XarrayTransducer` to
        :py:class:`.AsyncBufferWriter`.

    Example JSON configuration::

      {
        "emitter": "xarray",
        "emitter_arg": {
          "transducer": {...},
          "view": [...],
          "writer": {...}
        }
      }

    Here,

      - ``transducer`` is parsed by :py:class:`.XarrayTransducer`,
      - ``view`` is parsed by :py:class:`.ForestView`,
      - and ``writer`` is parsed by :py:class:`.AsyncBufferWriter`.

    For a complete example, see
    ``configs/test_configs/test_xarray_emitter.json``.
    """

    __slots__ = ("transducer", "writer", "finalized", "debug")

    metadata_keys = [
        'experiment_id', 'description', 'sim_data_path', 'time',
        'suffix_time', 'time_step', 'initial_global_time',
        'max_duration', 'fail_at_max_duration',
        'lineage_seed', 'seed', 'variants', 'n_init_sims', 'generations',
        'agent_id', 'parallel',
        'skip_baseline', 'log_updates',
        'single_daughters', 'daughter_outdir',
        'fixed_media', 'condition',
        'parca_options',
        'mar_regulon', 'amp_lysis',
        'divide', 'd_period', 'division_threshold', 'division_variable',
        'chromosome_path',
    ]
    """
    The subset of metadata selected by :py:meth:`.extract_metadata`.
    """

    def __init__(self, config: dict[str, Any], /) -> None:
        self.validate_config(config)
        self.debug: bool = config.get("debug", False)
        """ Flag for debug-level printing. Defaults to ``False``. """
        self.transducer: XarrayTransducer = XarrayTransducer(config, debug=self.debug)
        """ Presentation layer. """
        self.writer: AsyncBufferWriter = AsyncBufferWriter.dispatch(config["writer"])
        """ Session layer. """
        super().__init__()

    @classmethod
    def validate_config(cls, config: dict[str, Any], /) -> None:
        """
        Check assumptions about static emitter configuration.
        """
        for key in ["transducer", "view", "writer"]:
            if key not in config:
                raise KeyError(emitter_arg_error(
                    cls, "Missing argument", f"\"{key}\": ..."))
        match config.get("debug", False):
            case bool():
                pass
            case debug:
                raise TypeError(emitter_arg_error(
                    cls, "Invalid argument", f"\"debug\": {debug}"))

    # ~~~~~~~~~~~~~~~~~ #

    @classmethod
    def validate_metadata(cls, metadata: dict[str, Any], /) -> None:
        """
        Check assumptions about static simulator configuration.
        """
        expected = {
            "single_daughters": True,
            "save": False,
            "save_times": False,
            "emit_config": False,
            "emit_topology": False,
            "emit_processes": False,
            "emit_unique": False,
        }
        for (k, v) in expected.items():
            if bool(w := metadata[k]) != v:
                raise ValueError(
                    f"\n  Config argument unsupported by {cls.__name__}:"
                    f"\n    {{\"{k}\": {w}}}")

    def extract_partition(self, metadata: dict[str, Any], /) -> XarrayStoragePartition:
        return XarrayStoragePartition.cast(super().extract_partition(metadata))

    def extract_metadata(self, metadata: dict[str, Any], /) -> dict[str, Any]:
        """
        While executing :py:meth:`!Engine._emit_configuration` during
        :py:meth:`!Engine.__init__`, select and transform the subset of
        simulation metadata that will be stored by :py:class:`.XarrayEmitter`,
        starting from the :py:attr:`!Engine.metadata` that have been populated
        by :py:meth:`.EcoliSim.run`.
        """
        _metadata = {k: metadata[k] for k in self.metadata_keys}
        # reduce to basic JSON types
        for (k, v) in _metadata.items():
            match v:
                case Path():
                    _metadata[k] = str(v)
                case datetime():
                    # store the timestamp created by `EcoliSim.get_metadata()`,
                    # rather than the one created by `Engine.__init__()`
                    _metadata[k] = str(v.astimezone())
        if self.debug:
            hline = "-" * 79
            print(f"\nMetadata:\n{hline}")
            pp(_metadata)
            print(hline)
        return _metadata

    @staticmethod
    def extract_coords(metadata: dict[str, Any], /) -> dict[str, Any]:
        """
        While executing :py:meth:`!Engine._emit_configuration` during
        :py:meth:`!Engine.__init__`, extract the port schemas that have been
        populated by :py:meth:`.EcoliSim.output_metadata`.
        """
        return metadata["output_metadata"]

    @property
    def partition(self) -> XarrayStoragePartition:
        """
        Reference to :py:attr:`.XarrayBuffer.partition`.
        """
        return self.transducer.buffer.partition

    def flush(self, *, final=False) -> None:
        """
        Calls: :py:meth:`.AsyncBufferWriter.write`.
        """
        self.writer.write(self.transducer, final=final)

    # ~~~~~~~~~~~~~~~~~ #

    def reset_emit_flags(
        self, *,
        engine: Engine, agent: HierarchyPath, emit_paths: tuple[HierarchyPath]
    ) -> None:
        """
        In this subclass, ``agent`` is required and ``emit_paths`` is expected
        to be empty.
        """
        assert engine.emitter is self
        if emit_paths:
            raise KeyError(
                "For {\"emitter\": \"xarray\"}, please provide:\n"
                "  {\"emitter_arg\": {\"view\": ...}}\n"
                "  instead of\n"
                "  {\"emit_paths\": ...}")
        engine.state.set_emit_value(
            emit=False, path=tuple())
        assert isinstance(agent_state := engine.state.get_path(agent), Store)
        agent_state.set_emit_values(
            emit=True, paths=self.transducer.buffer.view.emitting_paths)

    def emit(self, data: dict[str, Any], /):
        """
        Main method.

        Calls: :py:meth:`.XarrayTransducer.step` and possibly :py:meth:`.flush`.
        """
        header, payload = data["table"], data["data"]
        match header:
            # sender: `Engine._emit_configuration()`
            case "configuration":
                self.validate_metadata(metadata := payload["metadata"])
                self.transducer.alloc(
                    partition=self.extract_partition(metadata),
                    metadata=self.extract_metadata(metadata),
                    coords=self.extract_coords(metadata))
                self.writer.open_store(self.transducer.buffer)
            # sender: `Engine._emit_store_data()`
            case "history":
                if not self.transducer.step(payload):
                    self.flush()
                    self.transducer.shift()
                    assert self.transducer.step(payload)
            case _:
                raise ValueError(f"Unexpected emit type: {header}")

    def _finalize(self, *, success: bool) -> None:
        self.flush(final=True)
        if success:
            self.writer.mark_success()
        self.writer.close()
