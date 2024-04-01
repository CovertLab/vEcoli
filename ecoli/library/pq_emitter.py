import atexit
from typing import Any, Mapping
from concurrent.futures import ThreadPoolExecutor
import os

import orjson
import pathlib
import pyarrow
import tempfile
from pyarrow import json as pj
from pyarrow import parquet as pq
from pyarrow import dataset as ds
from vivarium.core.emitter import Emitter
from vivarium.core.serialize import make_fallback_serializer_function

def get_datasets(outdir):
    history = ds.dataset(os.path.join(outdir, 'history'))
    cell_dirs = set(os.path.dirname(f) for f in history.files)
    history_schema = pyarrow.unify_schemas((pq.read_schema(
        os.path.join(f, '_common_metadata')) for f in cell_dirs))
    history = ds.dataset(os.path.join(outdir, 'history'), history_schema)
    config = ds.dataset(os.path.join(outdir, 'configuration'))
    config_schema = pyarrow.unify_schemas(
        (pq.read_schema(f) for f in config.files))
    config = ds.dataset(os.path.join(outdir, 'configuration'), config_schema)
    return config, history


def get_encoding(val):
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

def write_parquet(tempfile, outfile, encodings=None):
    tempfile.seek(0)
    table = pj.read_json(tempfile,
        read_options=pj.ReadOptions(block_size=int(1e7)))
    use_dictionary = encodings is None
    pq.write_table(table, outfile, use_dictionary=use_dictionary, 
                   column_encoding=encodings, compression='zstd')
    tempfile.close()

_FLAG_FIRST = object()

def flatten_dict(d: dict):
    """
    Flatten nested dictionary down to key-value pairs where each key
    concatenates all the keys needed to reach the
    corresponding value in the input. Prunes empty dicts and lists.
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


class PQEmitter(Emitter):
    """
    Emit data to a Parquet dataset.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.experiment_id = config.get('experiment_id', 'default')
        self.emits_to_batch = config.get('emits_to_batch', 50)
        agent_id = config.get('agent_id', '0')
        partitioning_keys = {
            'experiment_id': config.get('experiment_id', 'default'),
            'variant': config.get('variant', 'default'),
            'seed': config.get('seed', '0'),
            'generation': len(agent_id),
            'agent_id': agent_id
        }
        # Get correct Parquet dataset dir
        outdir = pathlib.Path(config.get('outdir', 'out'))
        self.history_outdir = outdir / 'history'
        self.config_outdir = outdir / 'configuration'
        for k, v in partitioning_keys.items():
            self.history_outdir = self.history_outdir / f'{k}={v}'
            self.config_outdir = self.config_outdir / f'{k}={v}'
        os.makedirs(self.history_outdir, exist_ok=True)
        os.makedirs(self.config_outdir, exist_ok=True)
        self.fallback_serializer = make_fallback_serializer_function()
        self.temp_file = tempfile.TemporaryFile()
        self.batched_emits = 0
        self.executor = ThreadPoolExecutor()
        self.encodings = {}
        self.accounted_fields = set()
        atexit.register(self._write_parquet)

    def _write_parquet(self):
        write_parquet(self.temp_file,
            self.history_outdir / f'{self.batched_emits}.parquet',
            self.encodings)
        history = ds.dataset(self.history_outdir)
        history_schema = pyarrow.unify_schemas((
            pq.read_schema(f) for f in history.files))
        pq.write_metadata(history_schema,
            self.history_outdir / '_common_metadata')
        self.temp_file.close()

    def emit(self, data: dict[str, Any]):
        data = orjson.loads(orjson.dumps(
            data, option=orjson.OPT_SERIALIZE_NUMPY,
            default=self.fallback_serializer))
        # Config will always be first emit
        if data['table'] == 'configuration':
            metadata = data['data'].pop('metadata')
            data['data'] = {**metadata, **data['data']}
            data['experiment_id'] = data['data'].pop('experiment_id')
            data['agent_id'] = data['data'].pop('agent_id')
            data['seed'] = data['data'].pop('seed')
            data['generation'] = len(data['agent_id'])
            # TODO: These keys need to be added
            data['variant'] = 0
            data = flatten_dict(data)
            self.temp_file.write(orjson.dumps(
                data, option=orjson.OPT_SERIALIZE_NUMPY,
                default=self.fallback_serializer))
            encodings = {}
            for k, v in data.items():
                encoding = get_encoding(v)
                if encoding is not None:
                    encodings[k] = encoding
            self.executor.submit(write_parquet, self.temp_file,
                self.config_outdir / 'config.parquet', encodings)
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
                self.history_outdir / f'{self.batched_emits}.parquet',
                self.encodings)
            self.temp_file = tempfile.TemporaryFile()
