import argparse
import atexit
import os
import pathlib
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
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


def cleanup_files(ndjson_dir: str, schema_file: str, encodings_file: str):
    """
    Registered to delete temporary sim output files when sim finishes,
    regardless of whether conversion to Parquet was successful.
    """
    shutil.rmtree(ndjson_dir)
    pathlib.Path(schema_file).unlink()
    pathlib.Path(encodings_file).unlink()


def json_to_parquet(ndjson: str, encodings: dict[str, str],
    parse_options: pj.ParseOptions, read_options: pj.ReadOptions,
    filesystem: fs.FileSystem, outdir: str):
    """
    Reads newline-delimited JSON file and converts to Parquet file.

    Args:
        ndjson: Path to newline-delimited JSON file.
        encodings: Mapping of column names to Parquet encodings
        parse_options: PyArrow JSON parse options
        read_options: PyArrow JSON read options
        filesystem: PyArrow filesystem for Parquet output
        outdir: Path to output directory for Parquet files
    """
    t = pj.read_json(ndjson, read_options=read_options,
                     parse_options=parse_options)
    outfile = os.path.join(outdir, os.path.basename(ndjson))
    pq.write_table(t, outfile, use_dictionary=False, compression='zstd',
        column_encoding=encodings, filesystem=filesystem)


def convert_multithreaded(ndjson_dir: str, schema_file: str,
                          encoding_file: str, out_uri: str):
    """
    Converts temporary sim output files into Parquet files at end of sim.

    Args:
        ndjson_dir: Path to temporary directory with newline-delimited JSONs
            batched into separate files
        schema_file: Path to temporary file with PyArrow schema for emits
        encoding_file: Path to temporary file with map from column names
            to Parquet encodings
        out_uri: URI of directory to write final Parquet files
    """
    with open(encoding_file, 'rb') as f:
        encodings = orjson.loads(f.readline())
    schema = pq.read_schema(schema_file)
    filesystem, outdir = fs.FileSystem.from_uri(out_uri)
    parse_opt = pj.ParseOptions(explicit_schema=schema)
    read_opt = pj.ReadOptions(use_threads=False, block_size=int(1e7))
    filesystem.create_dir(outdir)
    # Leverage any single core SMT without increasing memory usage too much
    executor = ThreadPoolExecutor(2)
    for temp_file in os.listdir(ndjson_dir):
        executor.submit(json_to_parquet, os.path.join(ndjson_dir, temp_file),
                        encodings, parse_opt, read_opt, filesystem, outdir)


def get_datasets(out_dir: Union[str, pathlib.Path]=None,
                 out_uri: str=None, unify_schemas: bool=False
                 ) -> tuple[ds.Dataset, ds.Dataset]:
    """
    PyArrow currently reads and uses the schema of one Parquet file at
    random for the entire dataset. If fields are present in other files that
    do not appear in the chosen file, they will not be found. This function
    offers a flag to scan all Parquet files and unify their schemas (can be
    slow for massive datasets). For better performance, if you know exactly
    what fields and types you would like to access, you can manually ensure
    they are included as follows::

        from ecoli.library.parquet_emitter import get_datasets
        # Do not set unify_schemas flag to True
        history_ds, config_ds = get_datasets(out_dir)
        # Create schema for new field(s) of interest
        new_fields = pa.schema([
            # new_field_1 contains variable-length lists of integers
            ('new_field_1', pa.list_(pa.int64())),
            # new_field_2 contains 64-bit floats
            ('new_field_2', pa.float64()),
            # Almost always want to include the following for filtering
            ('experiment_id', pa.string()),
            ('variant', pa.string()),
            ('seed', pa.int64()),
            ('generation', pa.int64()),
            ('agent_id', pa.string())
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
    partitioning_schema = pa.schema([
        ('experiment_id', pa.string()),
        ('variant', pa.string()),
        ('seed', pa.int64()),
        ('generation', pa.int64()),
        ('agent_id', pa.string())
    ])
    partitioning = ds.partitioning(partitioning_schema, flavor='hive')
    history = ds.dataset(os.path.join(outdir, 'history'),
                         partitioning=partitioning, filesystem=filesystem)
    config = ds.dataset(os.path.join(outdir, 'configuration'),
                        partitioning=partitioning, filesystem=filesystem)
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


def get_encoding(val: Any, field_name: str, use_uint16: bool=False,
                 use_uint32: bool=False) -> tuple[Any, str]:
    """
    Get optimal PyArrow type and Parquet encoding for input value.
    """
    if isinstance(val, float):
        return pa.float64(), 'BYTE_STREAM_SPLIT', field_name
    elif isinstance(val, bool):
        return pa.bool_(), 'RLE', field_name
    elif isinstance(val, int):
        # Optimize memory usage for select integer fields
        if use_uint16:
            pa_type = pa.uint16()
        elif use_uint32:
            pa_type = pa.uint32()
        else:
            pa_type = pa.int64()
        return pa_type, 'DELTA_BINARY_PACKED', field_name
    elif isinstance(val, str):
        return pa.string(), 'DELTA_BYTE_ARRAY', field_name
    elif isinstance(val, list):
        inner_type, _, field_name = get_encoding(
            val[0], field_name, use_uint16, use_uint32)
        # PLAIN encoding yields overall better compressed size for lists
        return pa.list_(inner_type), 'PLAIN', field_name + '.list.element'
    raise TypeError(f'{field_name} has unsupported type {type(val)}.')


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
        self.temp_dir = tempfile.NamedTemporaryFile().name
        os.makedirs(self.temp_dir)
        self.temp_data = open(os.path.join(self.temp_dir, '0.pq'), 'w+b')
        self.temp_schema = tempfile.NamedTemporaryFile(delete=False)
        self.temp_encodings = tempfile.NamedTemporaryFile(delete=False)
        # Keep a cache of field encodings and fields encountered
        self.encodings = {}
        self.schema = pa.schema([])
        self.num_emits = 0
        # Convert emits to Parquet on shutdown
        atexit.register(self._start_conversion)
    
    def _start_conversion(self):
        """
        Replaces current Python process with fresh process to convert
        newline-delimited JSON to Parquet (new process frees RAM).
        """
        flush_data([self.temp_data, self.temp_schema, self.temp_encodings])
        os.execlp('python', 'python', __file__, '-d', self.temp_dir,
                  '-s', self.temp_schema.name, '-e', self.temp_encodings.name,
                  '-o', os.path.join(self.out_uri, 'history',
                                     self.partitioning_path))

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
            data_str = orjson.dumps(
                data, option=orjson.OPT_SERIALIZE_NUMPY,
                default=self.fallback_serializer)
            self.temp_data.write(data_str)
            data = orjson.loads(data_str)
            encodings = {}
            schema = []
            schema = []
            for k, v in data.items():
                pa_type, encoding, field_name = get_encoding(v, k)
                if encoding is not None:
                    encodings[field_name] = encoding
                schema.append((k, pa_type))
            out_uri = os.path.join(self.out_uri, data['table'],
                                   self.partitioning_path)
            filesystem, outdir = fs.FileSystem.from_uri(out_uri)
            parse_options = pj.ParseOptions(explicit_schema=pa.schema(schema))
            read_options = pj.ReadOptions(use_threads=False, block_size=int(1e7))
            filesystem.create_dir(outdir)
            json_to_parquet(self.temp_data.name, encodings, parse_options,
                            read_options, filesystem, outdir)
            self.temp_data = open(self.temp_data.name, 'w+b')
            return
        # Currently we only support running a single cell per Engine
        # Use EcoliEngineProcess for colony simulations (Engine per cell)
        assert len(data['data']['agents']) == 1
        for agent_data in data['data']['agents'].values():
            agent_data['time'] = float(data['data']['time'])
            agent_data = flatten_dict(agent_data)
            agent_data_str = orjson.dumps(
                agent_data, option=orjson.OPT_SERIALIZE_NUMPY,
                default=self.fallback_serializer)
            self.temp_data.write(agent_data_str)
            agent_data = orjson.loads(agent_data_str)
            self.temp_data.write('\n'.encode('utf-8'))
            new_keys = set(agent_data) - set(self.schema.names)
            if len(new_keys) > 0:
                new_schema = []
                for k in new_keys:
                    pa_type, encoding, field_name = get_encoding(
                        agent_data[k], k, k in USE_UINT16, k in USE_UINT32)
                    if encoding is not None:
                        self.encodings[field_name] = encoding
                    self.schema = self.schema.append(pa.field(k, pa_type))
                out_uri = os.path.join(self.out_uri, data['table'],
                                       self.partitioning_path)
                # Replace previous schema and options to include new fields
                self.temp_schema.close()
                self.temp_encodings.close()
                self.temp_schema = open(self.temp_schema.name, 'w+b')
                self.temp_encodings = open(self.temp_encodings.name, 'w+b')
                self.temp_encodings.write(orjson.dumps(self.encodings))
                pq.write_metadata(self.schema, self.temp_schema.name)
        self.num_emits += 1
        if self.num_emits % self.batch_size == 0:
            self.temp_data.close()
            self.temp_data = open(os.path.join(
                self.temp_dir, f'{self.num_emits}.pq'), 'w+b')


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
        '--encodings', '-e', action='store',
        help='Path to file containing map of column names to encodings')
    parser.add_argument(
        '--outuri', '-o', action='store',
        help='URI of directory to write final Parquet files')
    args = parser.parse_args()
    atexit.register(cleanup_files, args.data, args.schema, args.encodings)
    convert_multithreaded(args.data, args.schema, args.encodings, args.outuri)


if __name__ == '__main__':
    main()
