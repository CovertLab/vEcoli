import atexit
import os
import pathlib
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, BinaryIO, Mapping, Union

import orjson
import pyarrow
from pyarrow import dataset as ds
from pyarrow import fs
from pyarrow import json as pj
from pyarrow import parquet as pq
from vivarium.core.emitter import Emitter
from vivarium.core.serialize import make_fallback_serializer_function


def get_datasets(outdir: Union[str, pathlib.Path]
                 ) -> tuple[ds.Dataset, ds.Dataset]:
    """
    PyArrow does not currently support schema evolution: the changing
    of fields/types in Parquet files that are part of the same dataset.
    Since this is a common occurence in Vivarium-based models (e.g.
    creating/deleting stores), we provide this convenience function
    to automatically unify all schemas in the specified outdir and
    return PyArrow datasets for sim configurations and sim outputs with
    these unified schemas.

    Args:
        outdir: Directory containing ``history`` and ``configuration``
            dataset folders
    
    Returns:
        Tuple ``(configuration dataset, history dataset)``.
    """
    history = ds.dataset(os.path.join(outdir, 'history'),
                         partitioning='hive')
    cell_dirs = set(os.path.dirname(f) for f in history.files)
    history_schema = pyarrow.unify_schemas((pq.read_schema(
        os.path.join(f, '_common_metadata')) for f in cell_dirs),
        promote_options='permissive')
    history_schema = pyarrow.unify_schemas((history.schema, history_schema))
    history = ds.dataset(os.path.join(outdir, 'history'), history_schema,
                         partitioning='hive')
    config = ds.dataset(os.path.join(outdir, 'configuration'),
                        partitioning='hive')
    config_schema = pyarrow.unify_schemas(
        (pq.read_schema(f) for f in config.files),
        promote_options='permissive')
    config_schema = pyarrow.unify_schemas((config.schema, config_schema))
    config = ds.dataset(os.path.join(outdir, 'configuration'), config_schema,
                        partitioning='hive')
    return config, history


def get_encoding(val: Any) -> str:
    """
    Get optimal Parquet encoding for input value. Returns None if the default
    dictionary encoding is the best option.
    """
    if isinstance(val, float):
        return 'BYTE_STREAM_SPLIT'
    elif isinstance(val, bool):
        return
    elif isinstance(val, int):
        return 'DELTA_BINARY_PACKED'
    elif isinstance(val, str):
        return 'DELTA_BYTE_ARRAY'
    elif isinstance(val, list):
        return get_encoding(val[0])


def write_parquet(tempfile: BinaryIO, outfile: str,
                  filesystem: fs.FileSystem, encodings: dict[str, str]=None):
    """
    Read newline-delimited JSON of simulation output and write Parquet file.

    Args:
        tempfile: Newline-delimited JSON file object (each emit on new line)
        outfile: Path and name of output Parquet file
        filesystem: FileSystem object inferred from ``config['outdir']`` or
            for local output or ``config['outuri']`` for S3, GCS, etc.
        encodings: Mapping of field names to non-default encodings (e.g.
            from calling :py:func:`~.get_encoding`)
    """
    tempfile.seek(0)
    table = pj.read_json(tempfile,
        read_options=pj.ReadOptions(block_size=int(1e7)))
    use_dictionary = encodings is None
    sorting_columns = pq.SortingColumn.from_ordering(
        table.schema, [('time', 'ascending')])
    pq.write_table(table, outfile, filesystem=filesystem,
        use_dictionary=use_dictionary, column_encoding=encodings,
        compression='zstd', sorting_columns=sorting_columns)
    tempfile.close()


_FLAG_FIRST = object()

def flatten_dict(d: dict):
    """
    Flatten nested dictionary down to key-value pairs where each key
    concatenates all the keys needed to reach the corresponding value
    in the input. Prunes empty dicts and lists. Allows each field in
    emits to be written to, compressed, and encoded as its own column
    in a Parquet file for efficient storage and retrieval.
    """
    results = []

    def visit_key(subdict, results, partialKey):
        for k, v in subdict.items():
            newKey = k if partialKey==_FLAG_FIRST else f'{partialKey}__{k}'
            if isinstance(v, Mapping):
                visit_key(v, results, newKey)
            elif isinstance(v, list) and len(v) == 0:
                continue
            elif v is None:
                continue
            else:
                results.append((newKey, v))

    visit_key(d, results, _FLAG_FIRST)
    return dict(results)


class ParquetEmitter(Emitter):
    """
    Emit data to a Parquet dataset.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Partitioning keys will be extracted from configuration emit
        and used to fully construct outdir paths in first run of
        :py:meth:`~.ParquetEmitter.emit`.
        """
        emitter_config = config.get('config', {})
        self.filesystem, outdir = fs.FileSystem.from_uri(
            emitter_config.get('outdir', 'out'))
        self.history_outdir = pathlib.Path(outdir) / 'history'
        self.config_outdir = pathlib.Path(outdir) / 'configuration'
        self.fallback_serializer = make_fallback_serializer_function()
        # Write emits to temp file and convert to Parquet in batches
        self.temp_file = tempfile.TemporaryFile()
        self.batched_emits = 0
        self.emits_to_batch = config.get('emits_to_batch', 50)
        # PyArrow uses efficient code that can release the GIL and can
        # be I/O bound. Call in separate thread to minimize blocking.
        self.executor = ThreadPoolExecutor()
        # Keep a cache of field encodings and fields encountered
        self.encodings = {}
        self.accounted_fields = set()
        # Convert all remaining emits upon program shutdown
        atexit.register(self._shutdown)

    def _shutdown(self):
        """
        Called upon program shutdown to ensure all remaining emits are
        written to a Parquet file. Also unifies all schemas written
        during this experiment into a ``_common_metadata`` file to
        reduce the amount of disk I/O required when unifying schemas
        for many experiments in :py:func:`~.get_datasets`.
        """
        write_parquet(self.temp_file, str(self.history_outdir /
            f'{self.batched_emits}.parquet'), self.filesystem, self.encodings)
        history = ds.dataset(self.history_outdir)
        history_schema = pyarrow.unify_schemas((
            pq.read_schema(f) for f in history.files))
        pq.write_metadata(history_schema, self.history_outdir /
            '_common_metadata')
        self.temp_file.close()

    def emit(self, data: dict[str, Any]):
        """
        Serializes emit data with ``orjson`` and writes newline-delimited
        JSONs in a temporary file. Users can specify ``emits_to_batch``
        in their emitter config to control how many such JSONs are written
        before being converted into a Parquet file in ``outdir`` (also from
        emitter config). The output directory will contain two subdirectories:
        ``configuration`` (for metadata emitted once per simulation) and 
        ``history`` (for simulation output data). Each of these directories is
        a Parquet dataset with hive partitioning. Users are encouraged to
        use a single ``outdir`` for a given model, calling
        :py:func:`~.get_datasets` and filtering as desired.
        """
        data = orjson.loads(orjson.dumps(
            data, option=orjson.OPT_SERIALIZE_NUMPY,
            default=self.fallback_serializer))
        # Config will always be first emit
        if data['table'] == 'configuration':
            metadata = data['data'].pop('metadata')
            data['data'] = {**metadata, **data['data']}
            data['time'] = data['data'].get('initial_global_time', 0.0)
            agent_id = data['data'].get('agent_id', '0')
            partitioning_keys = {
                'experiment_id': data['data'].get('experiment_id', 'default'),
                'variant': data['data'].get('variant', 'default'),
                'seed': data['data'].get('seed', 0),
                'generation': len(agent_id),
                'agent_id': agent_id
            }
            data = flatten_dict(data)
            self.temp_file.write(orjson.dumps(
                data, option=orjson.OPT_SERIALIZE_NUMPY,
                default=self.fallback_serializer))
            encodings = {}
            for k, v in data.items():
                encoding = get_encoding(v)
                if encoding is not None:
                    encodings[k] = encoding
            for k, v in partitioning_keys.items():
                self.history_outdir = self.history_outdir / f'{k}={v}'
                self.config_outdir = self.config_outdir / f'{k}={v}'
            self.filesystem.create_dir(str(self.history_outdir))
            self.filesystem.create_dir(str(self.config_outdir))
            write_parquet(self.temp_file, str(self.config_outdir /
                'config.parquet'), self.filesystem, encodings)
            self.temp_file = tempfile.TemporaryFile()
            return
        assert len(data['data']['agents']) == 1
        for agent_data in data['data']['agents'].values():
            agent_data['time'] = float(data['data']['time'])
            agent_data = flatten_dict(agent_data)
            new_keys = set(agent_data) - self.accounted_fields
            if len(new_keys) > 0:
                for k in new_keys:
                    encoding = get_encoding(agent_data[k])
                    if encoding is not None:
                        self.encodings[k] = encoding
                self.accounted_fields.update(new_keys)
            json_str = orjson.dumps(agent_data)
            self.temp_file.write(json_str)
            self.temp_file.write('\n'.encode('utf-8'))
        self.batched_emits += 1
        if self.batched_emits % self.emits_to_batch == 0:
            self.executor.submit(write_parquet, self.temp_file,
                str(self.history_outdir / f'{self.batched_emits}.parquet'),
                self.filesystem, self.encodings)
            self.temp_file = tempfile.TemporaryFile()
