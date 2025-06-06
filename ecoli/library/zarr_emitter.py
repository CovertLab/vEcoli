import zarr
import os
import orjson
from pyarrow import json as pj
from urllib import parse
from typing import Any, Optional
from ecoli.library.parquet_emitter import (
    flatten_dict,
    USE_UINT16,
    USE_UINT32,
    ndlist_to_ndarray,
)
from vivarium.core.emitter import Emitter, make_fallback_serializer_function
from concurrent.futures import ThreadPoolExecutor, Future
import tempfile

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})


def get_encoding(
    val: Any, field_name: str, use_uint16: bool = False, use_uint32: bool = False
) -> tuple[str, bool, tuple[int, ...]]:
    """
    Get optimal Zarr type for input value.
    """
    if isinstance(val, float):
        return "float64", False, ()
    elif isinstance(val, bool):
        return "bool", False, ()
    elif isinstance(val, str):
        return "str", False, ()
    elif isinstance(val, int):
        # Optimize memory usage for select integer fields
        if use_uint16:
            zarr_type = "uint16"
        elif use_uint32:
            zarr_type = "uint32"
        else:
            zarr_type = "int64"
        return zarr_type, False, ()
    elif isinstance(val, list):
        if len(val) > 0:
            for inner_val in val:
                inner_type, is_null, dims = get_encoding(
                    inner_val, field_name, use_uint16, use_uint32
                )
                if is_null:
                    continue
                return inner_type, is_null, (len(val),) + dims
        return "null", True, ()
    elif val is None:
        return "null", True, ()
    raise TypeError(f"{field_name} has unsupported type {type(val)}.")


def write_to_zarr(
    temp_file: str,
    root_store: zarr.Group,
    non_null_keys: set[str],
    time_range: tuple[int, int],
) -> None:
    """
    Write data to a Zarr array at the specified coordinates.
    """
    t = pj.read_json(temp_file, read_options=pj.ReadOptions(block_size=1e9))
    for k in sorted(non_null_keys):
        if k in t.column_names:
            array = root_store[k]
            try:
                array.append(ndlist_to_ndarray(t[k]))
            except ValueError as e:
                raise ValueError(
                    f"Failed to write data for key '{k}' at time range {time_range}: {e}"
                ) from e


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
                    'out_dir': local output directory (absolute/relative),
                }

        """
        self.out_dir = os.path.abspath(config["out_dir"])
        self.root_store = None
        self.fallback_serializer = make_fallback_serializer_function()
        self.non_null_keys = set()
        self.zarr_types = {}
        self.executor = ThreadPoolExecutor(2)
        self.last_batch_future: Optional[Future] = None
        self.temp_data = tempfile.NamedTemporaryFile(delete=False)
        self.last_time = 0
        self.num_emits = 0
        self.partitioning_keys = {}

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
                "experiment_id": quoted_experiment_id,
                "variant": data.get("variant", 0),
                "lineage_seed": data.get("lineage_seed", 0),
                "generation": len(agent_id),
                "agent_id": agent_id,
            }
            self.root_store = zarr.create_group(
                store=os.path.join(self.out_dir, quoted_experiment_id), overwrite=True
            )
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
            # If we encounter columns that have, up until this point,
            # been NULL, serialize/deserialize them and update their
            # type in our cached Parquet schema
            new_keys = set(agent_data) - set(self.non_null_keys)
            if len(new_keys) > 0:
                new_key_data = orjson.loads(
                    orjson.dumps(
                        {k: agent_data[k] for k in new_keys},
                        option=orjson.OPT_SERIALIZE_NUMPY,
                        default=self.fallback_serializer,
                    )
                )
                for k, v in new_key_data.items():
                    zarr_type, is_null, dims = get_encoding(
                        v, k, k in USE_UINT16, k in USE_UINT32
                    )
                    if not is_null:
                        self.zarr_types[k] = zarr_type
                        self.non_null_keys.add(k)
                        if k not in self.root_store:
                            self.root_store.create_array(
                                name=k,
                                shape=(0,) + dims,
                                dtype=zarr_type,
                                chunks=(100,) + dims,
                                shards=(4000,) + dims,
                                dimension_names=[
                                    "time",
                                ]
                                + [f"{k}_{i}" for i in range(len(dims))],
                                config={"write_empty_chunks": False},
                            )
            for k, v in self.partitioning_keys.items():
                agent_data[k] = v
            self.temp_data.write(
                orjson.dumps(
                    agent_data,
                    option=orjson.OPT_SERIALIZE_NUMPY,
                    default=self.fallback_serializer,
                )
            )
            self.num_emits += 1
        if self.num_emits % 100 == 0:
            # If last batch of emits failed, exception should be raised here
            self.temp_data.close()
            if self.last_batch_future is not None:
                self.last_batch_future.result()
            self.last_batch_future = self.executor.submit(
                write_to_zarr,
                self.temp_data.name,
                self.root_store,
                self.non_null_keys,
                (self.last_time, int(agent_data["time"]) - 1),
            )
            self.last_time = int(agent_data["time"]) + 1
            self.temp_data = tempfile.NamedTemporaryFile(delete=False)
