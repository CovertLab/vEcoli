import argparse
import atexit
import os
import pathlib
import sys
import tempfile
from typing import Any, BinaryIO, Mapping, Union

import orjson
import pyarrow as pa
from pyarrow import dataset as ds
from pyarrow import fs
from pyarrow import json as pj
from pyarrow import parquet as pq
from vivarium.core.emitter import Emitter
from vivarium.core.serialize import make_fallback_serializer_function

USE_UINT16 = {
    'listeners__rna_synth_prob__n_bound_TF_per_TU',
    'listeners__rna_synth_prob__n_bound_TF_per_cistron',
    'listeners__rnap_data__rna_init_event_per_cistron',
    'listeners__rna_synth_prob__gene_copy_number',
    'listeners__rna_synth_prob__expected_rna_init_per_cistron',
    'listeners__rna_degradation_listener__count_RNA_degraded_per_cistron',
    'listeners__rna_degradation_listener__count_rna_degraded',
    'listeners__transcript_elongation_listener__count_rna_synthesized',
    'listeners__rnap_data__rna_init_event',
    'listeners__rna_synth_prob__promoter_copy_number',
    'listeners__ribosome_data__n_ribosomes_on_each_mRNA',
    'listeners__ribosome_data__mRNA_TU_index',
    'listeners__complexation_listener__complexation_events',
    'listeners__rnap_data__active_rnap_n_bound_ribosomes',
    'listeners__rnap_data__active_rnap_domain_indexes',
    'listeners__rna_synth_prob__bound_TF_indexes',
    'listeners__rna_synth_prob__bound_TF_domains'
}
"""uint16 is 4x smaller than int64 for values between 0 - 65,535."""

USE_UINT32 = {
    'listeners__ribosome_data__ribosome_init_event_per_monomer',
    'listeners__ribosome_data__n_ribosomes_per_transcript',
    'listeners__rna_counts__partial_mRNA_cistron_counts',
    'listeners__rna_counts__mRNA_cistron_counts',
    'listeners__rna_counts__full_mRNA_cistron_counts',
    'listeners__ribosome_data__n_ribosomes_on_partial_mRNA_per_transcript',
    'listeners__monomer_counts',
    'listeners__rna_counts__partial_mRNA_counts',
    'listeners__rna_counts__mRNA_counts',
    'listeners__rna_counts__full_mRNA_counts',
    'listeners__fba_results__catalyst_counts'
}
"""uint32 is 2x smaller than int64 for values between 0 - 4,294,967,295."""


def flush_data(files_to_flush: list[BinaryIO]):
    """
    Ensure that all buffered data gets written to files.
    """
    for f in files_to_flush:
        f.flush()
        os.fsync(f.fileno())


def cleanup_files(ndjson: str, schema_file: str, options: str):
    """
    Registered to delete temporary sim output files when sim finishes,
    regardless of whether conversion to Parquet was successful. Same
    arguments as :py:func:`~.json_to_parquet`.
    """
    pathlib.Path(ndjson).unlink()
    pathlib.Path(schema_file).unlink()
    pathlib.Path(options).unlink()


def json_to_parquet(ndjson: str, schema_file: str, options: str):
    """
    Registered to convert temporary sim output files into single Parquet
    file at end of simulation.

    Args:
        ndjson: Path to temporary file with newline-delimited JSON emits
        schema_file: Path to temporary file with PyArrow schema for emits
        options: Path to temporary file with three lines - output URI in first
            line, JSON of custom Parquet column encodings in second line,
            and integer batch size of JSONs to write per Parquet row group
    """
    with open(options, 'rb') as f:
        out_uri = f.readline().split(b'\n')[0].decode('utf-8')
        encodings = orjson.loads(f.readline())
        batch_size = int(f.readline().split(b'\n')[0].decode('utf-8'))
    schema = pq.read_schema(schema_file)
    filesystem, outdir = fs.FileSystem.from_uri(out_uri)
    parse_options = pj.ParseOptions(explicit_schema=schema)
    # Keep memory usage down with minimal performance impact by disabling
    # multithreaded JSON reading
    read_options = pj.ReadOptions(use_threads=False, block_size=int(1e7))
    filesystem.create_dir(outdir)
    writer = pq.ParquetWriter(os.path.join(outdir, 'data.parquet'), 
        schema, use_dictionary=False, compression='zstd',
        column_encoding=encodings, filesystem=filesystem)
    # PyArrow JSON reader does not natively support streaming so we write
    # ``batch_size`` rows from main JSON into temp file for PyArrow to read
    with open(ndjson, 'rb') as f:
        temp_file = tempfile.NamedTemporaryFile()
        for i, line in enumerate(f):
            temp_file.write(line)
            temp_file.write('\n'.encode('utf-8'))
            if i % batch_size == 0 and i != 0:
                t = pj.read_json(temp_file.name, read_options=read_options,
                                 parse_options=parse_options)
                writer.write_table(t)
                temp_file.close()
                # Keep memory usage down by allowing t to be garbage collected
                del t
                temp_file = tempfile.NamedTemporaryFile()
        t = pj.read_json(temp_file.name, read_options=read_options,
                         parse_options=parse_options)
        writer.write_table(t)
        temp_file.close()


def get_datasets(out_dir: Union[str, pathlib.Path]=None,
                 out_uri: str=None, unify_schemas: bool=False
                 ) -> tuple[ds.Dataset, ds.Dataset]:
    """
    PyArrow currently reads and uses the schema of one Parquet file at
    random for the entire dataset. If fields are present in other files that
    do not appear in the chosen file, they will not be found. This function
    offers a flag to scan all Parquet files and unify their schemas (can be
    slow for massive datasets). For better performance, if know exactly what
    fields and types you would like to access, you can manually ensure those
    they are included as follows::

        from ecoli.library.parquet_emitter import get_datasets
        # Do not set unify_schemas flag to True
        history_ds, config_ds = get_datasets(out_dir)
        # Create schema for new field(s) of interest
        new_fields = pa.schema([
            # new_field_1 contains variable-length lists of integers
            ('new_field_1', pa.list_(pa.int64())),
            # new_field_2 contains 64-bit floats
            ('new_field_2', pa.float64())
        ])
        # Load dataset with new unified schema
        history_ds = history_ds.replace_schema(pa.unify_schemas([
            history_ds.schema, new_fields
        ]))

    Args:
        out_dir: Relative or absolute path to local directory containing
            ``history`` and ``configuration`` datasets
        out_uri: URI of directory with datasets (supersedes ``out_dir``)
        unify_schemas: Whether to scan all Parquet files and unify schemas
    
    Returns:
        Tuple ``(configuration dataset, history dataset)``.
    """
    if out_uri is None:
        out_uri = pathlib.Path(out_dir).resolve().as_uri()
    filesystem, outdir = fs.FileSystem.from_uri(out_uri)
    history = ds.dataset(os.path.join(outdir, 'history'),
                         partitioning='hive', filesystem=filesystem)
    config = ds.dataset(os.path.join(outdir, 'configuration'),
                        partitioning='hive', filesystem=filesystem)
    if unify_schemas:
        history_schema = pa.unify_schemas((
            pq.read_schema(f, filesystem=filesystem)
            for f in history.files), promote_options='permissive')
        history_schema = pa.unify_schemas([history.schema, history_schema],
                                          promote_options='permissive')
        history = history.replace_schema(history_schema)
        config_schema = pa.unify_schemas((
            pq.read_schema(f, filesystem=filesystem)
            for f in config.files), promote_options='permissive')
        config_schema = pa.unify_schemas([config.schema, config_schema],
                                          promote_options='permissive')
        config = config.replace_schema(config_schema)
    return config, history


def get_encoding(val: Any, use_uint16: bool=False, use_uint32: bool=False
                 ) -> tuple[Any, str]:
    """
    Get optimal PyArrow type and Parquet encoding for input value. Returns
    no encoding for the default dictionary encoding.
    """
    if isinstance(val, float):
        return pa.float64(), 'BYTE_STREAM_SPLIT'
    elif isinstance(val, bool):
        return pa.bool_(), None
    elif isinstance(val, int):
        # Optimize memory usage for select integer fields
        if use_uint16:
            pa_type = pa.uint16()
        elif use_uint32:
            pa_type = pa.uint32()
        else:
            pa_type = pa.int64()
        return pa_type, 'DELTA_BINARY_PACKED'
    elif isinstance(val, str):
        return pa.string(), 'DELTA_BYTE_ARRAY'
    elif isinstance(val, list):
        inner_type, encoding = get_encoding(val[0], use_uint16, use_uint32)
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
        Configure emitter.

        Args:
            config: Should be a dictionary as follows::

                {
                    'type': 'parquet',
                    'config': {
                        'emits_to_batch': Number of emits per Parquet row
                            group (default: 400),
                        # Only one of the following is required
                        'out_dir': absolute or relative output directory,
                        'out_uri': URI of output directory (e.g. s3://...,
                            gs://..., etc), supersedes out_dir
                    }
                }

        """
        out_dir = config['config'].get('out_dir', None)
        self.out_uri = config['config'].get('out_uri', None)
        if self.out_uri is None:
            self.out_uri = pathlib.Path(out_dir).resolve().as_uri()
        self.batch_size = config['config'].get('batch_size', 400)
        self.fallback_serializer = make_fallback_serializer_function()
        # Write emits as newline-delimited JSON into temporary file
        # then read/write them to Parquet at the end with unified schema
        self.temp_data = tempfile.NamedTemporaryFile(delete=False)
        self.temp_schema = tempfile.NamedTemporaryFile(delete=False)
        self.temp_options = tempfile.NamedTemporaryFile(delete=False)
        # Keep a cache of field encodings and fields encountered
        self.encodings = {}
        self.schema = pa.schema([])
        # Convert emits to Parquet on shutdown
        atexit.register(self._start_conversion)
    
    def _start_conversion(self):
        """
        Replaces current Python process with fresh process to convert
        newline-delimited JSON to Parquet (new process frees RAM).
        """
        flush_data([self.temp_data, self.temp_schema, self.temp_options])
        os.execlp('python', 'python', __file__, '-d', self.temp_data.name,
                  '-s', self.temp_schema.name, '-o', self.temp_options.name)

    def emit(self, data: dict[str, Any]):
        """
        Serializes emit data with ``orjson`` and writes newline-delimited
        JSONs in a temporary file to be batched before conversion to Parquet.
        
        The output directory will have the following hive-partitioned structure::

            - configuration
                - experiment_id=...
                    - variant=...
                        - seed=...
                            - generation=...
                                - agent_id=...
                                    - config.parquet
            - history
                - experiment_id=...
                    - variant=...
                        - seed=...
                            - generation=...
                                - agent_id=...
                                    - data.parquet

        By using a single output directory for many runs of a model, advanced
        filtering and computation can be performed on data from all those
        runs using PyArrow datasets (see :py:func:`~.get_datasets`).
        """
        data = orjson.loads(orjson.dumps(
            data, option=orjson.OPT_SERIALIZE_NUMPY,
            default=self.fallback_serializer))
        # Config will always be first emit
        if data['table'] == 'configuration':
            metadata = data['data'].pop('metadata')
            data['data'] = {**metadata, **data['data']}
            data['time'] = data['data'].get('initial_global_time', 0.0)
            # Manually create filepaths with hive partitioning
            agent_id = data['data'].get('agent_id', '0')
            partitioning_keys = {
                'experiment_id': data['data'].get('experiment_id', 'default'),
                'variant': data['data'].get('variant', 'default'),
                'seed': data['data'].get('seed', 0),
                'generation': len(agent_id),
                'agent_id': agent_id
            }
            for k, v in partitioning_keys.items():
                self.history_outdir = self.history_outdir / f'{k}={v}'
                self.config_outdir = self.config_outdir / f'{k}={v}'
            self.filesystem.create_dir(str(self.history_outdir))
            self.filesystem.create_dir(str(self.config_outdir))
            data = flatten_dict(data)
            self.temp_data.write(orjson.dumps(
                data, option=orjson.OPT_SERIALIZE_NUMPY,
                default=self.fallback_serializer))
            encodings = {}
            schema = []
            schema = []
            for k, v in data.items():
                pa_type, encoding = get_encoding(v)
                pa_type, encoding = get_encoding(v)
                if encoding is not None:
                    encodings[k] = encoding
                schema.append((k, pa_type))
            out_uri = os.path.join(self.out_uri, data['table'],
                                   self.partitioning_path)
            self.temp_options.write(out_uri.encode('utf-8'))
            self.temp_options.write('\n'.encode('utf-8'))
            self.temp_options.write(orjson.dumps(encodings))
            self.temp_options.write('\n'.encode('utf-8'))
            self.temp_options.write(str(self.batch_size).encode('utf-8'))
            self.temp_options.write('\n'.encode('utf-8'))
            flush_data([self.temp_data, self.temp_schema, self.temp_options])
            pq.write_metadata(pa.schema(schema), self.temp_schema.name)
            json_to_parquet(self.temp_data.name, self.temp_schema.name,
                            self.temp_options.name)
            self.temp_data = open(self.temp_data.name, 'w+b')
            self.temp_schema = open(self.temp_schema.name, 'w+b')
            self.temp_options = open(self.temp_options.name, 'w+b')
            return
        # Currently we only support running a single cell per Engine
        # Use EcoliEngineProcess for colony simulations (Engine per cell)
        assert len(data['data']['agents']) == 1
        for agent_data in data['data']['agents'].values():
            agent_data['time'] = float(data['data']['time'])
            agent_data = flatten_dict(agent_data)
            # If new fields are not frequently added to schema,
            # we do not have to update encodings often
            new_keys = set(agent_data) - set(self.schema.names)
            if len(new_keys) > 0:
                new_schema = []
                for k in new_keys:
                    pa_type, encoding = get_encoding(
                        agent_data[k], k in USE_UINT16, k in USE_UINT32)
                    if encoding is not None:
                        self.encodings[k] = encoding
                    self.schema = self.schema.append(pa.field(k, pa_type))
                out_uri = os.path.join(self.out_uri, data['table'],
                                       self.partitioning_path)
                # Replace previous schema and options to include new fields
                self.temp_schema.close()
                self.temp_options.close()
                self.temp_schema = open(self.temp_schema.name, 'w+b')
                self.temp_options = open(self.temp_options.name, 'w+b')
                self.temp_options.write(out_uri.encode('utf-8'))
                self.temp_options.write('\n'.encode('utf-8'))
                self.temp_options.write(orjson.dumps(self.encodings))
                self.temp_options.write('\n'.encode('utf-8'))
                self.temp_options.write(str(self.batch_size).encode('utf-8'))
                pq.write_metadata(self.schema, self.temp_schema.name)


def main():
    """
    DO NOT run this file directly. Should be called automatically at end of
    simulation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', '-d', action='store',
        help='Path to newline-delimited JSON from simulation.')
    parser.add_argument(
        '--schema', '-s', action='store',
        help='Path to Parquet file containing output schema.')
    parser.add_argument(
        '--options', '-o', action='store',
        help='Path to file containing output URI on first line, '
        'JSON column encodings on second, and batch size on third.')
    args = parser.parse_args()
    atexit.register(cleanup_files, args.data, args.schema, args.options)
    json_to_parquet(args.data, args.schema, args.options)


if __name__ == '__main__':
    main()
