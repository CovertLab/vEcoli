import atexit
import os
import pathlib
import tempfile
from typing import Any, Mapping, Union

import orjson
import pyarrow
from pyarrow import dataset as ds
from pyarrow import fs
from pyarrow import json as pj
from pyarrow import parquet as pq
from vivarium.core.emitter import Emitter
from vivarium.core.serialize import make_fallback_serializer_function


def json_to_parquet(ndjson, schema_file, other):
    with open(other, 'rb') as f:
        out_uri = f.readline().split(b'\n')[0].decode('utf-8')
        encodings = orjson.loads(f.readline())
    schema = pq.read_schema(schema_file)
    filesystem, outdir = fs.FileSystem.from_uri(out_uri)
    parse_options = pj.ParseOptions(explicit_schema=schema)
    read_options = pj.ReadOptions(use_threads=False, block_size=int(1e7))
    filesystem.create_dir(outdir)
    writer = pq.ParquetWriter(os.path.join(outdir, 'data.parquet'), 
        schema, use_dictionary=False, compression='zstd',
        column_encoding=encodings, filesystem=filesystem)
    with open(ndjson, 'rb') as f:
        temp_file = tempfile.NamedTemporaryFile()
        for i, line in enumerate(f):
            temp_file.write(line)
            temp_file.write('\n'.encode('utf-8'))
            if i % 200 == 0 and i != 0:
                t = pj.read_json(temp_file.name, read_options=read_options,
                                 parse_options=parse_options)
                writer.write_table(t)
                temp_file.close()
                del t
                temp_file = tempfile.NamedTemporaryFile()
        t = pj.read_json(temp_file.name, read_options=read_options,
                         parse_options=parse_options)
        writer.write_table(t)
        temp_file.close()


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
    filesystem, outdir = fs.FileSystem.from_uri(outdir)
    history = ds.dataset(os.path.join(outdir, 'history'),
                         partitioning='hive', filesystem=filesystem)
    cell_dirs = set(os.path.dirname(f) for f in history.files)
    history_schema = pyarrow.unify_schemas((pq.read_schema(
        os.path.join(f, '_common_metadata'), filesystem=filesystem)
        for f in cell_dirs), promote_options='permissive')
    history_schema = pyarrow.unify_schemas((history.schema, history_schema))
    history = ds.dataset(os.path.join(outdir, 'history'), history_schema,
                         partitioning='hive', filesystem=filesystem)
    config = ds.dataset(os.path.join(outdir, 'configuration'),
                        partitioning='hive', filesystem=filesystem)
    config_schema = pyarrow.unify_schemas(
        (pq.read_schema(f, filesystem=filesystem) for f in config.files),
        promote_options='permissive')
    config_schema = pyarrow.unify_schemas((config.schema, config_schema))
    config = ds.dataset(os.path.join(outdir, 'configuration'), config_schema,
                        partitioning='hive', filesystem=filesystem)
    return config, history


def get_encoding(val: Any) -> str:
    """
    Get optimal Parquet encoding for input value. Returns None if the default
    dictionary encoding is the best option.
    """
    if isinstance(val, float):
        return pyarrow.float64(), 'BYTE_STREAM_SPLIT'
    elif isinstance(val, bool):
        return pyarrow.bool_(), None
    elif isinstance(val, int):
        return pyarrow.int64(), 'DELTA_BINARY_PACKED'
    elif isinstance(val, str):
        return pyarrow.string(), 'DELTA_BYTE_ARRAY'
    elif isinstance(val, list):
        inner_type, encoding = get_encoding(val[0])
        return pyarrow.list_(inner_type), encoding


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
        self.experiment_id = config.get('experiment_id')
        self.outdir = pathlib.Path(config.get('config', {}).get('outdir', 'out'))
        self.fallback_serializer = make_fallback_serializer_function()
        # Write emits as newline-delimited JSON into temporary file
        # then read/write them to Parquet at the end with unified schema
        self.temp_data = tempfile.NamedTemporaryFile(delete=False)
        self.temp_schema = tempfile.NamedTemporaryFile(delete=False)
        self.temp_other = tempfile.NamedTemporaryFile(delete=False)
        # Keep a cache of field encodings and fields encountered
        self.encodings = {}
        self.schema = pyarrow.schema([])
        # Convert emits to Parquet on shutdown
        atexit.register(lambda : json_to_parquet(
            self.temp_data.name, self.temp_schema.name, self.temp_other.name))

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
            self.partitioning_path = os.path.join(*(
                f'{k}={v}' for k, v in partitioning_keys.items()))
            data = flatten_dict(data)
            self.temp_data.write(orjson.dumps(
                data, option=orjson.OPT_SERIALIZE_NUMPY,
                default=self.fallback_serializer))
            encodings = {}
            schema = []
            for k, v in data.items():
                pa_type, encoding = get_encoding(v)
                if encoding is not None:
                    encodings[k] = encoding
                schema.append((k, pa_type))
            outdir = self.outdir / data['table'] / self.partitioning_path
            self.temp_other.write(outdir.resolve().as_uri().encode('utf-8'))
            self.temp_other.write('\n'.encode('utf-8'))
            self.temp_other.write(orjson.dumps(encodings))
            pq.write_metadata(pyarrow.schema(schema), self.temp_schema.name)
            json_to_parquet(self.temp_data.name, self.temp_schema.name,
                            self.temp_other.name)
            self.temp_data = open(self.temp_data.name, 'w+b')
            self.temp_schema = open(self.temp_schema.name, 'w+b')
            self.temp_other = open(self.temp_other.name, 'w+b')
            return
        assert len(data['data']['agents']) == 1
        for agent_data in data['data']['agents'].values():
            agent_data['time'] = float(data['data']['time'])
            agent_data = flatten_dict(agent_data)
            self.temp_data.write(orjson.dumps(agent_data))
            self.temp_data.write('\n'.encode('utf-8'))
            new_keys = set(agent_data) - set(self.schema.names)
            if len(new_keys) > 0:
                for k in new_keys:
                    pa_type, encoding = get_encoding(agent_data[k])
                    if encoding is not None:
                        self.encodings[k] = encoding
                    self.schema = self.schema.append(pyarrow.field(k, pa_type))
                outdir = self.outdir / data['table'] / self.partitioning_path
                self.temp_schema.close()
                self.temp_other.close()
                self.temp_schema = open(self.temp_schema.name, 'w+b')
                self.temp_other = open(self.temp_other.name, 'w+b')
                self.temp_other.write(outdir.resolve().as_uri().encode('utf-8'))
                self.temp_other.write('\n'.encode('utf-8'))
                self.temp_other.write(orjson.dumps(self.encodings))
                pq.write_metadata(self.schema, self.temp_schema.name)
