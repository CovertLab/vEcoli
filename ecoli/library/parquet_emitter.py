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


def get_datasets(out_dir: Union[str, pathlib.Path]=None,
                 out_uri: str=None) -> tuple[ds.Dataset, ds.Dataset]:
    """
    PyArrow does not currently support schema evolution: variation in the
    fields/types of Parquet files that are part of the same dataset.
    Since this is a common occurence in Vivarium-based models (e.g.
    creating/deleting stores), we provide this convenience function
    to automatically unify all schemas return PyArrow Datasets that
    can see and operate on all possible fields.

    Args:
        out_dir: Relative or absolute path to local directory containing
            ``history`` and ``configuration`` datasets
        out_uri: URI of directory containing datasets (supersedes ``out_dir``)
    
    Returns:
        Tuple ``(configuration dataset, history dataset)``.
    """
    if out_uri is None:
        out_uri = pathlib.Path(out_dir).resolve().as_uri()
    filesystem, outdir = fs.FileSystem.from_uri(out_uri)
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


def get_encoding(val: Any, use_uint16: bool=False, use_uint32: bool=False
                 ) -> tuple[Any, str]:
    """
    Get optimal Parquet encoding for input value. Returns None if the default
    dictionary encoding is the best option.
    """
    if isinstance(val, float):
        return pyarrow.float64(), 'BYTE_STREAM_SPLIT'
        return pyarrow.float64(), 'BYTE_STREAM_SPLIT'
    elif isinstance(val, bool):
        return pyarrow.bool_(), None
        return pyarrow.bool_(), None
    elif isinstance(val, int):
        # Optimize memory usage for select integer fields
        if use_uint16:
            pa_type = pyarrow.uint16()
        elif use_uint32:
            pa_type = pyarrow.uint32()
        else:
            pa_type = pyarrow.int64()
        return pa_type, 'DELTA_BINARY_PACKED'
    elif isinstance(val, str):
        return pyarrow.string(), 'DELTA_BYTE_ARRAY'
        return pyarrow.string(), 'DELTA_BYTE_ARRAY'
    elif isinstance(val, list):
        inner_type, encoding = get_encoding(val[0], use_uint16, use_uint32)
        return pyarrow.list_(inner_type), encoding


def write_parquet(tempfile: BinaryIO, outfile: str, filesystem: fs.FileSystem,
                  encodings: dict[str, str]=None, schema: pyarrow.Schema=None):
    """
    Read newline-delimited JSON of simulation output and write Parquet file.

    Args:
        tempfile: Newline-delimited JSON file object (each emit on new line)
        outfile: Path and name of output Parquet file
        filesystem: FileSystem object inferred from ``config['out_dir']`` or
            for local output or ``config['out_uri']`` for S3, GCS, etc.
        encodings: Mapping of field names to non-default encodings (see
            :py:func:`~.get_encoding`)
    """
    tempfile.seek(0)
    table = pj.read_json(tempfile,
        read_options=pj.ReadOptions(block_size=int(1e7), use_threads=False),
        parse_options=pj.ParseOptions(explicit_schema=schema))
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
        Configure emitter.

        Args:
            config: Should be a dictionary as follows::

                {
                    'type': 'parquet',
                    'config': {
                        'emits_to_batch': Number of emits to batch before
                            converting to Parquet file (default: 50),
                        # Only one of the following is required
                        'out_dir': absolute or relative output directory,
                        'out_uri': URI of output directory (e.g. s3://...,
                            gs://..., etc), supersedes out_dir
                    }
                }

        """
        out_dir = config['config'].get('out_dir', None)
        out_uri = config['config'].get('out_uri', None)
        if out_uri is None:
            out_uri = pathlib.Path(out_dir).resolve().as_uri()
        self.filesystem, outdir = fs.FileSystem.from_uri(out_uri)
        self.history_outdir = pathlib.Path(outdir) / 'history'
        self.config_outdir = pathlib.Path(outdir) / 'configuration'
        self.fallback_serializer = make_fallback_serializer_function()
        # Write emits to temp file and convert to Parquet in batches
        self.temp_file = tempfile.TemporaryFile()
        self.batched_emits = 0
        self.emits_to_batch = config['config'].get('emits_to_batch', 100)
        # PyArrow can release the GIL and is often I/O bound.
        # Call in separate thread to minimize blocking.
        self.executor = ThreadPoolExecutor()
        # Keep a cache of field encodings and fields encountered
        self.encodings = {}
        self.schema = pyarrow.schema([])
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
            f'{self.batched_emits}.parquet'), self.filesystem, self.encodings, self.schema)
        pq.write_metadata(self.schema, str(self.history_outdir /
            '_common_metadata'), filesystem=self.filesystem)
        self.temp_file.close()

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
                                    - config.parquet: Simulation config
            - history
                - experiment_id=...
                    - variant=...
                        - seed=...
                            - generation=...
                                - agent_id=...
                                    - {num}.parquet: Batched emits
                                    - _common_metadata: Unified schema

        By using a single output directory for many runs of a model, advanced
        filtering and computation can be performed on data from all those
        runs using the datasets returned by :py:func:`~.get_datasets`.
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
            write_parquet(self.temp_file, str(self.config_outdir /
                'config.parquet'), self.filesystem, encodings, pyarrow.schema(schema))
            self.temp_file = tempfile.TemporaryFile()
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
                    new_schema.append((k, pa_type))
                self.schema = pyarrow.unify_schemas([
                    self.schema, pyarrow.schema(new_schema)])
            self.temp_file.write(orjson.dumps(agent_data))
            self.temp_file.write('\n'.encode('utf-8'))
        self.batched_emits += 1
        if self.batched_emits % self.emits_to_batch == 0:
            self.executor.submit(write_parquet, self.temp_file,
                str(self.history_outdir / f'{self.batched_emits}.parquet'),
                self.filesystem, self.encodings, self.schema)
            self.temp_file = tempfile.TemporaryFile()
