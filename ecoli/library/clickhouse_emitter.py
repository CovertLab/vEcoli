import atexit
import tempfile
import pathlib
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, BinaryIO

import clickhouse_connect
import orjson
from clickhouse_connect.driver.tools import insert_file
from vivarium.core.emitter import Emitter
from vivarium.core.serialize import make_fallback_serializer_function

from ecoli.library.parquet_emitter import flatten_dict


def get_ch_type(fieldname: str, py_val: Any) -> tuple[str, str]:
    """
    Get Clickhouse type and optimal encoding for value.

    Args:
        fieldname: Name of field for informative error
        py_val: Value to get type and codec for
    
    Returns:
        ``(type, codec)``
    """
    if isinstance(py_val, bool):
        return 'Bool', 'ZSTD'
    elif isinstance(py_val, int):
        return 'Int64', 'Delta,ZSTD'
    elif isinstance(py_val, float):
        return 'Float64', 'Gorilla,ZSTD'
    elif isinstance(py_val, str):
        return 'LowCardinality(String)', 'ZSTD'
    elif isinstance(py_val, Mapping):
        return 'JSON', 'ZSTD'
    elif isinstance(py_val, list):
        inner_fieldname = fieldname + '.0'
        inner_type, codec = get_ch_type(inner_fieldname, py_val[0])
        if inner_type == 'Int64' or inner_type == 'Bool':
            codec = 'T64,ZSTD'
        else:
            codec = 'ZSTD'
        return f'Array({inner_type})', codec
    raise TypeError(f'{fieldname} has unsupported type {type(py_val)}')


def add_new_fields(client: Any,
                   new_fields: dict[str, str],
                   table_id: str) -> list[str]:
    """
    Add new fields to a table (creates table if not exists).

    Args:
        client: Kwargs for :py:func:`clickhouse_connect.get_client`
        new_fields: Mapping of new field names to values
        table_id: Name of table to create new fields for

    Returns:
        Current fieldnames in table.
    """
    col_types = []
    for k, v in new_fields.items():
        ch_type, ch_codec = get_ch_type(k, v)
        col_types.append(f'`{k}` {ch_type} CODEC({ch_codec})')
    create_table_cmd = "CREATE TABLE IF NOT EXISTS {table_id:Identifier} ( " \
        f"{', '.join(col_types)} ) ENGINE = MergeTree " \
        "PRIMARY KEY (experiment_id,variant,seed,generation,agent_id,time)"
    add_cols_cmd = "ALTER TABLE {table_id:Identifier} ADD COLUMN IF NOT " \
        f"EXISTS {', ADD COLUMN IF NOT EXISTS '.join(col_types)}"
    curr_cols_cmd = "SELECT name from system.columns WHERE " \
        "table={table_id:String} ORDER BY position"
    client.command(create_table_cmd,  {'table_id': table_id})
    client.command(add_cols_cmd,  {'table_id': table_id})
    curr_cols = client.query_np(curr_cols_cmd, {'table_id': table_id})
    return curr_cols[:, 0].tolist()


def push_to_db(temp_file: BinaryIO, client: Any):
    """
    Compresses and sends newline-delimited JSONs to ClickHouse DB.
    """
    subprocess.run(['zstd', temp_file.name, '-o',
        f'{temp_file.name}.gz', '--rm', '-f'], check=True)
    insert_file(client, 'history', f'{temp_file.name}.gz',
        'JSONEachRow', compression='zstd')
    pathlib.Path(f'{temp_file.name}.gz').unlink()
    temp_file.close()


class ClickHouseEmitter(Emitter):
    """
    Emit data to a ClickHouse database.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Setup connection to database and configure emitter.

        Args:
            config: Must include ``experiment_id`` key. Can include keys for
                ``host``, ``port``, ``user``, ``database``, and ``password``
                to use as kwargs for :py:func:`clickhouse_connect.get_client`.
                Can also specify number of emits to send in each batch with
                ``emits_to_batch``.
        """
        self.outdir = pathlib.Path(config.get('outdir', 'out'))
        self.experiment_id = config.get('experiment_id')
        connection_args = {
            'host': config.get('host', 'localhost'),
            'port': config.get('port', 8123),
            'user': config.get('user', 'default'),
            'database': config.get('database', 'default'),
            'password': config.get('password', '')
        }
        self.client = clickhouse_connect.get_client(**connection_args,
            settings={'allow_experimental_object_type': 1})
        self.executor = ThreadPoolExecutor()
        self.curr_fields = []
        self.fallback_serializer = make_fallback_serializer_function()
        # Write emits to temp file and send to ClickHouse in batches
        self.temp_file = tempfile.NamedTemporaryFile(
            dir=self.outdir, prefix=self.experiment_id)
        self.batched_emits = 0
        self.emits_to_batch = config.get('emits_to_batch', 50)
        atexit.register(self._shutdown)
    
    def _shutdown(self):
        """
        Compresses and sends final batch of emits to ClickHouse DB at sim end.
        """
        subprocess.run(['zstd', self.temp_file.name, '-o',
        f'{self.temp_file.name}.gz', '--rm', '-f'], check=True)
        insert_file(self.client, 'history', f'{self.temp_file.name}.gz',
            'JSONEachRow', compression='zstd')
        pathlib.Path(f'{self.temp_file.name}.gz').unlink()

    def emit(self, data: dict[str, Any]):
        """
        Serializes emit data with ``orjson`` and writes newline-delimited
        JSONs in a temporary file. Users can specify ``emits_to_batch``
        in their emitter config to control how many such JSONs are written
        before being sent to ClickHouse DB.
        """
        data = orjson.loads(orjson.dumps(
            data, option=orjson.OPT_SERIALIZE_NUMPY,
            default=self.fallback_serializer))
        # Config will always be first emit
        if data['table'] == 'configuration':
            metadata = data['data'].pop('metadata')
            data['data'] = {**metadata, **data['data']}
            agent_id = data['data'].get('agent_id', '0')
            self.index_keys = {
                'experiment_id': data['data'].get('experiment_id', 'default'),
                'agent_id': agent_id,
                'seed': data['data'].get('seed', 0),
                'generation': len(agent_id),
                'variant': data['data'].get('variant', 'default'),
                'time': data['data'].get('initial_global_time', 0.0),
            }
            data.update(self.index_keys)
            curr_config_fields = add_new_fields(
                self.client, data, 'configuration')
            data = [data[k] for k in curr_config_fields]
            self.client.insert('configuration', [data])
            return
        assert len(data['data']['agents']) == 1
        for agent_data in data['data']['agents'].values():
            agent_data.update(self.index_keys)
            agent_data['time'] = data['data']['time']
            agent_data = flatten_dict(agent_data)
            new_cols = set(agent_data) - set(self.curr_fields)
            if len(new_cols) > 0:
                self.curr_fields = add_new_fields(self.client,
                    {k: agent_data[k] for k in new_cols}, 'history')
            json_str = orjson.dumps(agent_data)
            self.temp_file.write(json_str)
            self.temp_file.write('\n'.encode('utf-8'))
        self.batched_emits += 1
        if self.batched_emits % self.emits_to_batch == 0:
            self.executor.submit(push_to_db, self.temp_file, self.client)
            self.temp_file = tempfile.NamedTemporaryFile(
            dir=self.outdir, prefix=self.experiment_id)
