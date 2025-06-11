import zarr
import zipfile
import os
import shutil
from fsspec import filesystem
from urllib import parse
from typing import Any, Optional
from ecoli.library.parquet_emitter import (
    flatten_dict,
    USE_UINT16,
    USE_UINT32,
)
from vivarium.core.emitter import Emitter
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np
import warnings


def get_encoding(
    val: Any, field_name: str, use_uint16: bool = False, use_uint32: bool = False
) -> tuple[Any, tuple[int, ...]]:
    """
    Get optimal Zarr type for input value.
    """
    if isinstance(val, float):
        return np.float64, ()
    elif isinstance(val, bool):
        return np.bool_, ()
    elif isinstance(val, str):
        return np.dtypes.StringDType, ()
    elif isinstance(val, int):
        # Optimize memory usage for select integer fields
        if use_uint16:
            return np.uint16, ()
        elif use_uint32:
            return np.uint32, ()
        else:
            return np.int64, ()
    elif isinstance(val, np.ndarray):
        return val.dtype, val.shape
    elif isinstance(val, list):
        if len(val) > 0:
            for inner_val in val:
                np_type, dims = get_encoding(
                    inner_val, field_name, use_uint16, use_uint32
                )
                if np_type is None:
                    continue
                return np_type, (len(val),) + dims
        return None, ()
    elif val is None:
        return None, ()
    raise TypeError(f"{field_name} has unsupported type {type(val)}.")


def write_to_zarr(
    buffered_emits: dict[str, np.ndarray],
    root_store: zarr.Group,
    start_index: int,
    end_index: int,
) -> None:
    """
    Write data to a Zarr array at the specified coordinates.
    """
    for k, v in buffered_emits.items():
        array = root_store[k]
        array[start_index:end_index] = v
        buffered_emits[k] = None


class ZarrEmitter(Emitter):
    """
    Emit data to a Zarr dataset.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Configure emitter.

        Args:
            config: Should be a dictionary as follows::

                {
                    'type': 'zarr',
                    'emits_to_batch': Number of emits to batch before writing (default: 400),
                    'max_time': Estimated maximum number of emits (automatically doubled
                        every time exceeded, default: 10800),
                    # One of the following is REQUIRED
                    'out_dir': local output directory (absolute/relative),
                    'out_uri': Google Cloud storage bucket URI
                }

        """
        if "out_uri" not in config:
            self.out_uri = os.path.abspath(config["out_dir"])
            self.filesystem = filesystem("file")
        else:
            self.out_uri = config["out_uri"]
            self.filesystem = filesystem(parse.urlparse(self.out_uri).scheme)
        self.batch_size = config.get("batch_size", 400)
        self.max_time = config.get("max_time", 10800)
        self.np_types = {}
        self.executor = ThreadPoolExecutor(1)
        self.last_batch_future: Optional[Future] = None
        self.root_store = None
        self.num_emits = 0
        self.total_num_emits = 0
        self.partitioning_keys = {}
        self.buffered_emits = {}
        self.store_dir = None
        self.success = False
        # Silence warnings about strings not being officially part of Zarr v3
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="The codec `vlen-utf8`"
        )
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="The dtype `StringDType"
        )

    def _finalize(self):
        """Convert remaining batched emits to Parquet at sim shutdown
        and mark sim as successful if ``success`` flag was set. In vEcoli,
        this is done by :py:class:`~ecoli.experiments.ecoli_master_sim.EcoliSim`
        upon reaching division.
        """
        if self.last_batch_future is not None:
            self.last_batch_future.result()
        # If we have any buffered emits left, write them to Zarr
        for k, v in self.buffered_emits.items():
            if v is not None:
                self.buffered_emits[k] = v[: self.num_emits]
        write_to_zarr(
            self.buffered_emits,
            self.root_store,
            self.total_num_emits - self.num_emits,
            self.total_num_emits,
        )
        # Include success flag for easy filtering in analyses
        self.root_store.create_array(
            name="success",
            shape=(self.total_num_emits,),
            dtype="bool",
            chunks=(self.total_num_emits,),
            dimension_names=[
                "agent_time",
            ],
            config={"write_empty_chunks": False},
        )
        self.root_store["success"][:] = self.success
        for v in self.root_store.array_values():
            v.resize((self.total_num_emits,) + v.shape[1:])
        zarr.consolidate_metadata(self.root_store.store)
        # Zip Zarr store to reduce file count
        with zipfile.ZipFile(self.store_dir + ".zip", "w", zipfile.ZIP_STORED) as zf:
            for root, _, files in os.walk(self.store_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self.store_dir)
                    zf.write(file_path, arcname)
        shutil.rmtree(self.store_dir)

    def emit(self, data: dict[str, Any]):
        """
        Flattens emit dictionary by concatenating nested key names with double
        underscores (:py:func:`~.flatten_dict`), serializes flattened emit with
        ``orjson``, and writes each listener to a Zarr array.
        """
        # Config will always be first emit
        if data["table"] == "configuration":
            data = {**data["data"].pop("metadata"), **data["data"]}
            data["time"] = data.get("initial_global_time", 0.0)
            # Manually create filepaths with hive partitioning
            # Start agent ID with 1 to avoid leading zeros
            agent_id = data.get("agent_id", "1")
            quoted_experiment_id = parse.quote_plus(
                data.get("experiment_id", "default")
            )
            self.partitioning_keys = {
                "variant": data.get("variant", 0),
                "lineage_seed": data.get("lineage_seed", 0),
                "generation": len(agent_id),
                "agent_id": agent_id,
            }
            self.store_dir = os.path.join(
                self.out_uri,
                quoted_experiment_id,
                "history",
                *(f"{k}={v}" for k, v in self.partitioning_keys.items()),
            )
            self.partitioning_keys["experiment_id"] = quoted_experiment_id
            self.filesystem.makedirs(self.store_dir, exist_ok=True)
            self.root_store = zarr.create_group(store=self.store_dir, overwrite=True)
            return
        # Each Engine that uses this emitter should only simulate a single cell
        # In lineage simulations, StopAfterDivision Step will terminate
        # Engine in timestep immediately after division (first with 2 cells)
        # In colony simulations, EngineProcess will terminate simulation
        # immediately upon division (following branch is never invoked)
        if len(data["data"]["agents"]) > 1:
            return
        for agent_data in data["data"]["agents"].values():
            agent_data["time"] = float(data["data"]["time"])
            agent_data = flatten_dict(agent_data)
            serialized_agent = agent_data
            for k, v in self.partitioning_keys.items():
                serialized_agent[k] = v
            # If we encounter columns that have, up until this point,
            # been NULL, serialize/deserialize them and update their
            # type in our cached Parquet schema
            for k, v in serialized_agent.items():
                if k not in self.buffered_emits:
                    if k not in self.np_types:
                        np_type, dims = get_encoding(
                            v, k, k in USE_UINT16, k in USE_UINT32
                        )
                        if np_type is None:
                            continue
                        self.np_types[k] = (np_type, dims)
                        if k not in self.root_store:
                            if (
                                np_type == np.dtype("str")
                                or np_type == np.dtype("object")
                                or np_type == np.dtypes.StringDType()
                            ):
                                np_type = "str"
                            self.root_store.create_array(
                                name=k,
                                shape=(self.max_time,) + dims,
                                dtype=np_type,
                                dimension_names=[
                                    "data_id",
                                ]
                                + [f"{k}_{i}" for i in range(len(dims))],
                                config={"write_empty_chunks": False},
                            )
                    np_type, dims = self.np_types[k]
                    self.buffered_emits[k] = np.zeros(
                        (self.batch_size,) + dims, dtype=np_type
                    )
                self.buffered_emits[k][self.num_emits] = v
        self.num_emits += 1
        self.total_num_emits += 1
        if self.total_num_emits > self.max_time:
            # Resize arrays to accommodate more than 10800 emits
            self.max_time *= 2
            for v in self.root_store.array_values():
                v.resize((self.max_time,) + v.shape[1:])
        if self.num_emits % self.batch_size == 0:
            # If last batch of emits failed, exception should be raised here
            if self.last_batch_future is not None:
                self.last_batch_future.result()
            self.last_batch_future = self.executor.submit(
                write_to_zarr,
                self.buffered_emits,
                self.root_store,
                self.total_num_emits - self.num_emits,
                self.total_num_emits,
            )
            self.buffered_emits = {}
            self.num_emits = 0
