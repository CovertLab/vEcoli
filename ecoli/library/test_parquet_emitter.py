import os
import re
import tempfile
import shutil
import duckdb
import numpy as np
import polars as pl
import pytest
import time
import math
import datetime
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue

from ecoli.library.parquet_emitter import (
    json_to_parquet,
    np_dtype,
    named_idx,
    ndidx_to_duckdb_expr,
    flatten_dict,
    union_pl_dtypes,
    ParquetEmitter,
)


class TestHelperFunctions:
    @pytest.fixture
    def query_conn(self):
        """Create a Polars DataFrame for DuckDB to query."""
        conn = duckdb.connect(":memory:")
        # Create Polars DataFrame for DuckDB to query
        df = pl.DataFrame(  # noqa: F841
            {
                "a": [[0.1, 0.0, 0.3], [0.4, 0.5, 0.0], [None, 0.8, 0.9]],
                "b": [
                    [[0.1, 0.2], [0.3, None]],
                    [[0.5, 0.6], [0.0, 0.8]],
                    [[0.9, 0.0], [1.1, 1.2]],
                ],
                "c": [[[0.1, 0.2], [0.3]], [[0.5], [0.0, 0.8]], [[0.9], [1.1]]],
            }
        )
        conn.sql("CREATE OR REPLACE TABLE test_table AS SELECT * FROM df")
        yield conn

    def test_named_idx(self, query_conn):
        col_expr = named_idx("a", ["col1", "col2", "col3"], [[0, 1, 2]])
        result = query_conn.sql(f"SELECT {col_expr} FROM test_table").pl()
        expected = pl.DataFrame(
            {"col1": [0.1, 0.4, None], "col2": [0.0, 0.5, 0.8], "col3": [0.3, 0.0, 0.9]}
        )
        assert result.equals(expected)

        col_expr = named_idx(
            "a", ["col1", "col2", "col3"], [[0, 1, 2]], zero_to_null=True
        )
        result = query_conn.sql(f"SELECT {col_expr} FROM test_table").pl()
        expected = pl.DataFrame(
            {
                "col1": [0.1, 0.4, None],
                "col2": [None, 0.5, 0.8],
                "col3": [0.3, None, 0.9],
            }
        )
        assert result.equals(expected)

        col_expr = named_idx(
            "b", ["col1", "col2", "col3", "col4"], [[0, 1], [0, 1]], zero_to_null=True
        )
        result = query_conn.sql(f"SELECT {col_expr} FROM test_table").pl()
        expected = pl.DataFrame(
            {
                "col1": [0.1, 0.5, 0.9],
                "col2": [0.2, 0.6, None],
                "col3": [0.3, None, 1.1],
                "col4": [None, 0.8, 1.2],
            }
        )
        assert result.equals(expected)

    def test_ndidx_to_duckdb_expr(self, query_conn):
        expr = ndidx_to_duckdb_expr("b", [0, 1])
        result = query_conn.sql(f"SELECT {expr} FROM test_table").pl()
        expected = pl.DataFrame({"b": [[[0.2]], [[0.6]], [[0.0]]]})
        assert result.equals(expected)

        expr = ndidx_to_duckdb_expr("b", [":", [True, False]])
        print(expr)
        result = query_conn.sql(f"SELECT {expr} FROM test_table").pl()
        expected = pl.DataFrame({"b": [[[0.1], [0.3]], [[0.5], [0.0]], [[0.9], [1.1]]]})
        print(result)
        assert result.equals(expected)

        expr = ndidx_to_duckdb_expr("c", [[0], ":"])
        result = query_conn.sql(f"SELECT {expr} FROM test_table").pl()
        expected = pl.DataFrame({"c": [[[0.1, 0.2]], [[0.5]], [[0.9]]]})
        assert result.equals(expected)

    def test_flatten_dict(self):
        # Simple dictionary
        assert flatten_dict({"a": 1, "b": 2}) == {"a": 1, "b": 2}

        # Nested dictionary
        assert flatten_dict({"a": {"b": 1, "c": 2}, "d": 3}) == {
            "a__b": 1,
            "a__c": 2,
            "d": 3,
        }

        # Deeply nested dictionary
        assert flatten_dict({"a": {"b": {"c": {"d": 1}}}, "e": 2}) == {
            "a__b__c__d": 1,
            "e": 2,
        }

        # Empty dictionary
        assert flatten_dict({}) == {}

        # Dictionary with mixed value types
        nested = flatten_dict({"a": [1, 2, 3], "b": {"c": np.array([4, 5, 6])}})
        assert nested["a"] == [1, 2, 3]
        np.testing.assert_array_equal(nested["b__c"], np.array([4, 5, 6]))

    def test_np_dtype(self):
        # Basic types
        np_type = np_dtype(1.0, "float_field")
        assert np_type == np.float64

        np_type = np_dtype(True, "bool_field")
        assert np_type == np.bool_

        np_type = np_dtype("text", "string_field")
        assert np_type == np.dtypes.StringDType

        # Integer with different encodings
        np_type = np_dtype(42, "int_field")
        assert np_type == np.int64

        np_type = np_dtype(42, "listeners__ribosome_data__mRNA_TU_index")
        assert np_type == np.uint16

        np_type = np_dtype(42, "listeners__monomer_counts")
        assert np_type == np.uint32

        # Arrays with various dimensions
        np_type = np_dtype(np.array([1, 2, 3]), "array1d_field")
        assert np_type == np.int64

        np_type = np_dtype(np.array([[1, 2], [3, 4]]), "array2d_field")
        assert np_type == np.int64

        # Empty arrays still have a dtype
        np_type = np_dtype(np.array([]), "empty_array_field")
        assert np_type == np.float64

        # Raise error for empty lists to fall back to Polars serialization
        with pytest.raises(ValueError, match="empty_list_field has unsupported"):
            np_type = np_dtype([[], [], None], "empty_list_field")

        # Raise error for none to fall back to Polars serialization
        with pytest.raises(ValueError, match="none_field has unsupported"):
            np_type = np_dtype(None, "none_field")

        # Invalid types
        with pytest.raises(ValueError, match="complex_field has unsupported type"):
            np_dtype(complex(1, 2), "complex_field")

    def test_union_pl_dtypes(self):
        # Basic types
        with pytest.raises(
            TypeError, match=re.escape("Incompatible inner types for field")
        ):
            union_pl_dtypes(pl.Int32, pl.Int64, "fail")
        with pytest.raises(
            TypeError, match=re.escape("Incompatible inner types for field")
        ):
            union_pl_dtypes(pl.Float32, pl.String, "fail")

        # Nested types
        with pytest.raises(
            TypeError, match=re.escape("Incompatible inner types for field")
        ):
            union_pl_dtypes(pl.List(pl.Int16), pl.List(pl.Int64), "nest")
        with pytest.raises(
            TypeError, match=re.escape("Incompatible inner types for field")
        ):
            union_pl_dtypes(pl.List(pl.UInt16), pl.List(pl.String), "nest_fail")
        with pytest.raises(
            TypeError, match=re.escape("Incompatible inner types for field")
        ):
            union_pl_dtypes(
                pl.List(pl.List(pl.UInt16)), pl.List(pl.String), "nest_fail"
            )
        with pytest.raises(
            TypeError, match=re.escape("Incompatible inner types for field")
        ):
            union_pl_dtypes(
                pl.List(pl.UInt16), pl.List(pl.Array(pl.String, (1,))), "nest_fail"
            )
        assert union_pl_dtypes(
            pl.List(pl.UInt16), pl.List(pl.Int64), "force_u32", pl.UInt32
        ) == pl.List(pl.UInt32)

        # Forced types: a bit scary but we assume user knows what they are doing
        assert union_pl_dtypes(pl.Int16, pl.UInt8, "force_u16", pl.UInt16) == pl.UInt16
        assert union_pl_dtypes(pl.UInt16, pl.Int64, "force_u32", pl.UInt32) == pl.UInt32
        assert (
            union_pl_dtypes(pl.UInt16, pl.String, "force_u32", pl.UInt32) == pl.UInt32
        )
        assert union_pl_dtypes(
            pl.List(pl.UInt16), pl.List(pl.String), "force_u32", pl.UInt32
        ) == pl.List(pl.UInt32)
        assert union_pl_dtypes(
            pl.List(pl.UInt16), pl.List(pl.Int64), "force_u32", pl.UInt32
        ) == pl.List(pl.UInt32)
        assert union_pl_dtypes(
            pl.Array(pl.UInt16, (1, 1)),
            pl.List(pl.List(pl.Int64)),
            "force_u16",
            pl.UInt16,
        ) == pl.List(pl.List(pl.UInt16))

        # Null merge
        assert union_pl_dtypes(pl.Null, pl.Int64, "null_merge") == pl.Int64
        assert union_pl_dtypes(pl.Null, pl.Float64, "force_u16", pl.UInt16) == pl.UInt16
        assert union_pl_dtypes(
            pl.Null, pl.List(pl.Int64), "force_u16", pl.UInt16
        ) == pl.List(pl.UInt16)
        assert union_pl_dtypes(
            pl.List(pl.Null), pl.List(pl.List(pl.Float32)), "null_merge"
        ) == pl.List(pl.List(pl.Float32))
        assert union_pl_dtypes(
            pl.Array(pl.Null, (1, 1, 1)),
            pl.List(pl.Array(pl.Float32, (1, 1))),
            "null_merge",
        ) == pl.List(pl.List(pl.List(pl.Float32)))
        assert union_pl_dtypes(
            pl.List(pl.Null), pl.List(pl.String), "force_u16", pl.UInt16
        ) == pl.List(pl.UInt16)
        assert union_pl_dtypes(
            pl.List(pl.Null), pl.List(pl.List(pl.Int32)), "force_u32", pl.UInt32
        ) == pl.List(pl.List(pl.UInt32))
        assert union_pl_dtypes(
            pl.List(pl.Null), pl.List(pl.List(pl.List(pl.Int32))), "null_merge"
        ) == pl.List(pl.List(pl.List(pl.Int32)))
        assert union_pl_dtypes(
            pl.List(pl.Null),
            pl.List(pl.List(pl.List(pl.Int32))),
            "force_u32",
            pl.UInt32,
        ) == pl.List(pl.List(pl.List(pl.UInt32)))


def compare_nested(a: list, b: list) -> bool:
    """
    Compare two lists for equality, including special handling for NaN.
    """
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(compare_nested(a[i], b[i]) for i in range(len(a)))
    if a != b:
        try:
            return math.isnan(a) and math.isnan(b)
        except TypeError:
            return False
    return True


class TestParquetEmitter:
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        tmp = tempfile.mkdtemp()
        yield tmp
        shutil.rmtree(tmp)

    def test_initialization(self, temp_dir):
        """Test ParquetEmitter initialization with different configs."""
        # Test with out_dir
        emitter = ParquetEmitter({"out_dir": temp_dir})
        emitter.experiment_id = "test_exp"
        emitter.partitioning_path = "path/to/output"
        assert emitter.out_uri == os.path.abspath(temp_dir)
        assert emitter.batch_size == 400

        # Test with out_uri and custom batch size
        emitter = ParquetEmitter({"out_uri": "gs://bucket/path", "batch_size": 100})
        emitter.experiment_id = "test_exp"
        emitter.partitioning_path = "path/to/output"
        assert emitter.out_uri == "gs://bucket/path"
        assert emitter.batch_size == 100

    def test_emit_configuration(self, temp_dir):
        """Test emitting configuration data."""
        emitter = ParquetEmitter({"out_dir": temp_dir})

        # Setup ThreadPoolExecutor mock
        future = Future()
        future.set_result(None)
        emitter.executor.submit = Mock(return_value=future)

        # Test with basic config data
        config_data = {
            "table": "configuration",
            "data": {
                "experiment_id": "test_exp",
                "variant": 1,
                "lineage_seed": 100,
                "agent_id": "1",
                "nested": {"value": 42},
                "metadata": {"meta1": "value1", "meta2": "value2"},
            },
        }

        emitter.emit(config_data)

        # Verify partitioning path
        assert emitter.experiment_id == "test_exp"
        assert "experiment_id=test_exp" in emitter.partitioning_path
        assert "variant=1" in emitter.partitioning_path

        # Verify json_to_parquet was called
        assert emitter.executor.submit.called
        args, _ = emitter.executor.submit.call_args
        assert args[0] == json_to_parquet

    def test_emit_simulation_data(self, temp_dir):
        """Test emitting simulation data with various types."""
        emitter = ParquetEmitter({"out_dir": temp_dir, "batch_size": 2})

        # Configuration emit to initialize variables
        config_data = {
            "table": "configuration",
            "data": {
                "experiment_id": "test_exp",
                "variant": 1,
                "lineage_seed": 1,
                "agent_id": "1",
            },
        }
        emitter.emit(config_data)
        emitter.last_batch_future.result()

        # Create test data with various types
        sim_data1 = {
            "table": "simulation",
            "data": {
                "time": 1.0,
                "agents": {
                    "agent1": {
                        "int_field": 42,
                        "float_field": 3.14,
                        "bool_field": True,
                        "string_field": "hello",
                        "array_field": np.array([1, 2, 3]),
                        "nested": {"value": 100},
                    }
                },
            },
        }

        # First emit
        emitter.emit(sim_data1)
        assert emitter.num_emits == 1
        emitter.last_batch_future.result()

        # Check that fields were properly encoded
        assert "int_field" in emitter.buffered_emits
        assert emitter.buffered_emits["int_field"][0] == 42

        assert "float_field" in emitter.buffered_emits
        assert emitter.buffered_emits["float_field"][0] == 3.14

        assert "bool_field" in emitter.buffered_emits
        assert emitter.buffered_emits["bool_field"][0]

        assert "string_field" in emitter.buffered_emits
        assert emitter.buffered_emits["string_field"][0] == "hello"

        assert "array_field" in emitter.buffered_emits
        np.testing.assert_array_equal(
            emitter.buffered_emits["array_field"][0], np.array([1, 2, 3])
        )

        assert "nested__value" in emitter.buffered_emits
        assert emitter.buffered_emits["nested__value"][0] == 100

        # Second emit to trigger batch writing
        emitter.emit(sim_data1)
        assert emitter.num_emits == 2
        emitter.last_batch_future.result()

        # Check output
        t = pl.read_parquet(
            os.path.join(
                emitter.out_uri,
                emitter.experiment_id,
                "history",
                emitter.partitioning_path,
                "*.pq",
            )
        )
        assert t["int_field"].to_list() == [42] * 2
        assert t["float_field"].to_list() == [3.14] * 2
        assert t["bool_field"].to_list() == [True] * 2
        assert all(t["string_field"] == ["hello"] * 2)
        np.testing.assert_array_equal(t["array_field"].to_list(), [[1, 2, 3]] * 2)
        assert all(t["nested__value"] == [100] * 2)
        assert emitter.buffered_emits == {}

    def test_variable_length_arrays(self, temp_dir):
        """Test handling arrays with changing dimensions."""
        emitter = ParquetEmitter({"out_dir": temp_dir, "batch_size": 3})
        # Configuration emit to initialize variables
        config_data = {
            "table": "configuration",
            "data": {
                "experiment_id": "test_exp",
                "variant": 1,
                "lineage_seed": 1,
                "agent_id": "1",
            },
        }
        emitter.emit(config_data)
        emitter.last_batch_future.result()

        sim_data1 = {
            "table": "simulation",
            "data": {
                "time": 1.0,
                "agents": {
                    "agent1": {
                        "dynamic_array": np.array([1, 2, 3]),
                        "ragged_nd": [[1, 2, 3], [1, 2], [1]],
                    }
                },
            },
        }
        emitter.emit(sim_data1)
        emitter.last_batch_future.result()

        # Verify arrays were stored correctly
        assert "dynamic_array" in emitter.buffered_emits
        assert emitter.buffered_emits["dynamic_array"].shape[1:] == (3,)
        assert "ragged_nd" in emitter.buffered_emits
        assert all(
            emitter.buffered_emits["ragged_nd"][0]
            == pl.Series([[1, 2, 3], [1, 2], [1]])
        )

        # Second emit with different shapes
        sim_data2 = {
            "table": "simulation",
            "data": {
                "time": 2.0,
                "agents": {
                    "agent1": {
                        "dynamic_array": np.array([4, 5, 6, 7]),
                        "ragged_nd": [[1], [1, 2], [1, 2, 3]],
                    }
                },
            },
        }
        emitter.emit(sim_data2)
        emitter.last_batch_future.result()

        # Verify conversion to variable length type
        assert isinstance(emitter.buffered_emits["dynamic_array"], list)
        assert emitter.buffered_emits["dynamic_array"][0] == [1, 2, 3]
        assert emitter.buffered_emits["dynamic_array"][1].to_list() == [4, 5, 6, 7]
        assert all(
            emitter.buffered_emits["ragged_nd"][1]
            == pl.Series([[1], [1, 2], [1, 2, 3]])
        )

        # Write to Parquet and check output
        emitter.emit(sim_data2)
        emitter.last_batch_future.result()

        t = pl.read_parquet(
            os.path.join(
                emitter.out_uri,
                emitter.experiment_id,
                "history",
                emitter.partitioning_path,
                "*.pq",
            )
        )
        assert t["dynamic_array"].to_list() == [
            [1, 2, 3],
            [4, 5, 6, 7],
            [4, 5, 6, 7],
        ]
        assert t["ragged_nd"].to_list() == [
            [[1, 2, 3], [1, 2], [1]],
            [[1], [1, 2], [1, 2, 3]],
            [[1], [1, 2], [1, 2, 3]],
        ]

    def test_extreme_data_types(self, temp_dir):
        """Test with extreme data types and edge cases."""
        emitter = ParquetEmitter({"out_dir": temp_dir, "batch_size": 2})
        # Create test data with extreme values and special cases
        sim_data = {
            "table": "configuration",
            "data": {
                "time": 1.0,
                "experiment_id": "test_exp",
                "variant": 1,
                "lineage_seed": 100,
                "agent_id": "1",
                "agents": {
                    "agent1": {
                        # Extreme values
                        "max_int": np.iinfo(np.int64).max,
                        "min_int": np.iinfo(np.int64).min,
                        "max_float": np.finfo(np.float64).max,
                        "tiny_float": 1e-100,
                        "nan_value": np.nan,
                        "inf_value": np.inf,
                        # Special cases
                        "empty_array": np.array([]),
                        "zero_dim_array": np.array(42),
                        "unicode_string": "Unicode: 日本語",
                        "very_long_string": "x" * 10000,
                        # Nested structures
                        "deep_nesting": {"level1": {"level2": {"level3": [1, 2, 3]}}},
                        "ragged_nullable": [None, [1, 2], [None, 1, 2, 3]],
                        # Python datetime objects
                        "datetime_list": [
                            datetime.datetime(2000, 12, 25),
                            datetime.datetime(2001, 4, 1, 12),
                            datetime.datetime(2002, 1, 1, 0, 1),
                            datetime.datetime(2003, 2, 14, 5, 5, 5),
                            datetime.datetime(2003, 7, 4, 7, 8, 9, 10),
                        ],
                        "time_list": [
                            datetime.time(1),
                            datetime.time(2, 3),
                            datetime.time(4, 5, 6),
                            datetime.time(7, 8, 9, 10),
                        ],
                        "datetime": datetime.datetime(1776, 7, 4),
                        # Test bytes
                        "npbytes": np.array([b"test bytes"])[0],
                        "pybytes": b"test bytes",
                        "npbytes_list": np.array([b"test1", b"test2"]),
                        "pybytes_list": [b"test1", b"test2"],
                    }
                },
            },
        }

        # Try to emit the extreme data
        # First configuration emit to set variables
        emitter.emit(sim_data)
        emitter.last_batch_future.result()
        # Then simulation emit
        sim_data["table"] = "simulation"
        emitter.emit(sim_data)
        emitter.last_batch_future.result()
        assert emitter.num_emits == 1

        # Verify fields were processed
        assert "max_int" in emitter.buffered_emits
        assert emitter.buffered_emits["max_int"][0] == np.iinfo(np.int64).max

        assert "min_int" in emitter.buffered_emits
        assert emitter.buffered_emits["min_int"][0] == np.iinfo(np.int64).min

        assert "max_float" in emitter.buffered_emits
        assert emitter.buffered_emits["max_float"][0] == np.finfo(np.float64).max

        assert "tiny_float" in emitter.buffered_emits
        assert emitter.buffered_emits["tiny_float"][0] == 1e-100

        assert "nan_value" in emitter.buffered_emits
        assert np.isnan(emitter.buffered_emits["nan_value"][0])

        assert "unicode_string" in emitter.buffered_emits
        assert emitter.buffered_emits["unicode_string"][0] == "Unicode: 日本語"

        assert "deep_nesting__level1__level2__level3" in emitter.buffered_emits
        assert np.array_equal(
            emitter.buffered_emits["deep_nesting__level1__level2__level3"][0],
            np.array([1, 2, 3], dtype=int),
        )

        sim_data_2 = {
            "table": "simulation",
            "data": {
                "time": 2.0,
                "experiment_id": "test_exp",
                "variant": 1,
                "lineage_seed": 100,
                "agent_id": "1",
                "agents": {
                    "agent1": {
                        # Shuffle extreme values
                        "max_int": np.iinfo(np.int64).min,
                        "min_int": np.iinfo(np.int64).max,
                        "max_float": np.finfo(np.float64).min,
                        "tiny_float": 1e100,
                        "nan_value": np.inf,
                        "inf_value": np.nan,
                        # More special cases
                        "empty_array": np.array([np.nan]),
                        "zero_dim_array": np.array(0),
                        "unicode_string": "Unicode: 日本語 再び",
                        "very_long_string": "x" * 100000,
                        # Nested structures
                        "deep_nesting": {
                            "level1": {
                                # Add a new field mid-batch
                                "level2": {"level3": [1, 2, 3, 4], "level4": [5, 6, 7]}
                            }
                        },
                        "ragged_nullable": [
                            [1, 3, 4],
                            [None, None, 1],
                            None,
                            [1, 2, None],
                        ],
                        # Python datetime objects
                        "datetime_list": [datetime.datetime(2000, 12, 25)],
                        "time_list": [datetime.time(1)],
                        "datetime": datetime.datetime(2000, 12, 25),
                        # Test bytes
                        "npbytes": np.array([b"short"])[0],
                        "pybytes": b"short",
                        "npbytes_list": np.array([b"much longer bytestring", b"1"]),
                        "pybytes_list": [b"much longer bytestring", b"1"],
                    }
                },
            },
        }
        emitter.emit(sim_data_2)
        emitter.last_batch_future.result()
        assert emitter.buffered_emits == {}

        out_path = os.path.join(
            emitter.out_uri,
            emitter.experiment_id,
            "history",
            emitter.partitioning_path,
            f"{emitter.num_emits}.pq",
        )
        output_pl = pl.read_parquet(out_path)

        output_data = {
            "max_int": [np.iinfo(np.int64).max, np.iinfo(np.int64).min],
            "min_int": [np.iinfo(np.int64).min, np.iinfo(np.int64).max],
            "max_float": [np.finfo(np.float64).max, np.finfo(np.float64).min],
            "tiny_float": [1e-100, 1e100],
            "nan_value": [np.nan, np.inf],
            "inf_value": [np.inf, np.nan],
            "empty_array": [[], [np.nan]],
            "zero_dim_array": [42, 0],
            "unicode_string": ["Unicode: 日本語", "Unicode: 日本語 再び"],
            "very_long_string": ["x" * 10000, "x" * 100000],
            "deep_nesting__level1__level2__level3": [[1, 2, 3], [1, 2, 3, 4]],
            "deep_nesting__level1__level2__level4": [None, [5, 6, 7]],
            "ragged_nullable": [
                [None, [1, 2], [None, 1, 2, 3]],
                [[1, 3, 4], [None, None, 1], None, [1, 2, None]],
            ],
            "datetime_list": [
                [
                    datetime.datetime(2000, 12, 25),
                    datetime.datetime(2001, 4, 1, 12),
                    datetime.datetime(2002, 1, 1, 0, 1),
                    datetime.datetime(2003, 2, 14, 5, 5, 5),
                    datetime.datetime(2003, 7, 4, 7, 8, 9, 10),
                ],
                [datetime.datetime(2000, 12, 25)],
            ],
            "time_list": [
                [
                    datetime.time(1),
                    datetime.time(2, 3),
                    datetime.time(4, 5, 6),
                    datetime.time(7, 8, 9, 10),
                ],
                [datetime.time(1)],
            ],
            "datetime": [
                datetime.datetime(1776, 7, 4),
                datetime.datetime(2000, 12, 25),
            ],
            # Test bytes
            "npbytes": [b"test bytes", b"short"],
            "pybytes": [b"test bytes", b"short"],
            # Note the truncation for the NumPy bytes array
            "npbytes_list": [[b"test1", b"test2"], [b"much\x20", b"1"]],
            "pybytes_list": [[b"test1", b"test2"], [b"much longer bytestring", b"1"]],
        }
        for key, value in output_data.items():
            assert compare_nested(output_pl[key].to_list(), value), (
                f"Mismatch in field {key}"
            )

    def test_finalize(self, temp_dir):
        """Test finalize method that handles remaining data."""
        emitter = ParquetEmitter({"out_dir": temp_dir})
        emitter.experiment_id = "test_exp"
        emitter.partitioning_path = "path/to/output"

        # Add data to buffered_emits
        emitter.buffered_emits = {
            "field1": np.zeros((emitter.batch_size,), dtype=np.int64),
            "field2": np.zeros((emitter.batch_size,), dtype=np.float64),
        }
        emitter.pl_types = {
            "field1": pl.Int64,
            "field2": pl.Float64,
        }
        emitter.buffered_emits["field1"][0] = 10
        emitter.buffered_emits["field2"][0] = 20.5
        emitter.num_emits = 1  # Only one emit happened

        # Mock json_to_parquet
        with patch(
            "ecoli.library.parquet_emitter.json_to_parquet"
        ) as mock_json_to_parquet:
            # Test finalize
            emitter.finalize()

            # Verify json_to_parquet was called with truncated data
            mock_json_to_parquet.assert_called_once()
            args, _ = mock_json_to_parquet.call_args
            assert args[0]["field1"].shape[0] == 1  # Only 1 item should remain
            assert args[0]["field1"][0] == 10
            assert args[0]["field2"][0] == 20.5

        # Test success flag
        emitter.success = True
        emitter.finalize()
        assert os.path.exists(
            os.path.join(
                emitter.out_uri,
                emitter.experiment_id,
                "success",
                emitter.partitioning_path,
                "s.pq",
            )
        )

    def test_multiple_agents(self, temp_dir):
        emitter = ParquetEmitter({"out_dir": temp_dir})
        emitter.experiment_id = "test_exp"
        emitter.partitioning_path = "path/to/output"

        # Create data with multiple agents
        sim_data = {
            "table": "simulation",
            "data": {
                "time": 1.0,
                "agents": {"agent1": {"field1": 10}, "agent2": {"field1": 20}},
            },
        }

        # Should return early without processing
        emitter.emit(sim_data)
        assert emitter.num_emits == 0
        assert emitter.buffered_emits == {}

    def test_batch_processing(self, temp_dir):
        """Test multiple emits and batch processing."""
        # Small batch size for testing
        emitter = ParquetEmitter({"out_dir": temp_dir, "batch_size": 3})

        # Configuration emit to initialize variables
        config_data = {
            "table": "configuration",
            "data": {
                "experiment_id": "test_exp",
                "variant": 1,
                "lineage_seed": 1,
                "agent_id": "1",
            },
        }
        emitter.emit(config_data)
        emitter.last_batch_future.result()

        # Create simulation data
        sim_data = {
            "table": "simulation",
            "data": {"time": 1.0, "agents": {"agent1": {"value": 10}}},
        }

        # Emit 4 times (should trigger batch processing after 3)
        for i in range(4):
            sim_data["data"]["time"] = float(i)
            sim_data["data"]["agents"]["agent1"]["value"] = i * 10
            emitter.emit(sim_data)
            emitter.last_batch_future.result()

        # Verify batch was processed
        assert emitter.num_emits == 4

        # One value should remain in buffer
        assert len(emitter.buffered_emits["value"]) == emitter.batch_size
        assert emitter.buffered_emits["value"][0] == 30  # Last value (3*10)


class TestParquetEmitterEdgeCases:
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        tmp = tempfile.mkdtemp()
        yield tmp
        shutil.rmtree(tmp)

    @patch("ecoli.library.parquet_emitter.ThreadPoolExecutor")
    def test_multithreaded_buffer_clearing(self, mock_executor_class, temp_dir):
        """
        Test to verify that clearing buffers after submitting to ThreadPoolExecutor
        doesn't cause race conditions with the worker thread.
        """
        # Create a real executor and a queue to track what's passed to json_to_parquet
        real_executor = ThreadPoolExecutor(max_workers=1)
        data_capture_queue = Queue()

        # Setup a custom submit function that will capture the dictionaries
        # being passed to json_to_parquet before they can be cleared
        def capture_submit(func, *args, **kwargs):
            # Make a deep copy of the dictionaries to capture their state
            emit_dict_copy = {
                k: v.copy() if hasattr(v, "copy") else v for k, v in args[0].items()
            }
            pl_types_copy = args[2].copy()

            # Put the copies in our queue for later inspection
            data_capture_queue.put((emit_dict_copy, pl_types_copy))

            # Create a future that will complete after a delay to simulate
            # the worker thread taking time to process
            future = Future()

            # Submit the real work to a real executor
            def delayed_execution():
                time.sleep(0.1)  # Delay to ensure main thread moves on
                result = func(*args, **kwargs)
                future.set_result(result)
                return result

            real_executor.submit(delayed_execution)
            return future

        # Configure our mock executor to use the capture_submit function
        mock_executor = Mock()
        mock_executor.submit.side_effect = capture_submit
        mock_executor_class.return_value = mock_executor

        # Initialize the emitter with a small batch size
        emitter = ParquetEmitter({"out_dir": temp_dir, "batch_size": 2})
        # Configuration emit to initialize variables
        config_data = {
            "table": "configuration",
            "data": {
                "experiment_id": "test_exp",
                "variant": 1,
                "lineage_seed": 1,
                "agent_id": "1",
            },
        }
        emitter.emit(config_data)
        emitter.last_batch_future.result()

        # First emit with simple data
        sim_data1 = {
            "table": "simulation",
            "data": {
                "time": 1.0,
                "agents": {"agent1": {"field1": np.array([1, 2, 3]), "field2": 42}},
            },
        }
        emitter.emit(sim_data1)

        # Second emit that will trigger batch writing
        # This data has the same structure
        sim_data2 = {
            "table": "simulation",
            "data": {
                "time": 2.0,
                "agents": {"agent1": {"field1": np.array([4, 5, 6]), "field2": 43}},
            },
        }
        emitter.emit(sim_data2)

        # At this point, the batch should have been submitted to the executor
        # and the buffers cleared in the main thread

        # Now immediately emit data with a bunch of additional fields
        # This should not cause any issues even though the previous data is still
        # being processed by the worker thread
        sim_data3 = {
            "table": "simulation",
            "data": {
                "time": 3.0,
                "agents": {
                    "agent1": {
                        f"field{i}": np.array([7, 8, 9, 10]) for i in range(1, 10)
                    }
                },
            },
        }
        emitter.emit(sim_data3)

        # Verify the captured data matches what was in buffers before clearing
        captured_data, captured_types = data_capture_queue.get(timeout=1)
        assert captured_data["experiment_id"] == "test_exp"
        assert captured_data["variant"] == 1
        assert captured_data["lineage_seed"] == 1
        assert captured_types == {
            "experiment_id": pl.String,
            "variant": pl.Int64,
            "lineage_seed": pl.Int64,
            "agent_id": pl.String,
            "time": pl.Float64,
        }

        captured_data, captured_types = data_capture_queue.get(timeout=1)
        assert len(captured_data["field1"]) == emitter.batch_size
        assert captured_data["field1"][0].tolist() == [1, 2, 3]
        assert captured_data["field1"][1].tolist() == [4, 5, 6]
        assert captured_types == {
            "time": pl.Float64,
            "field1": pl.List(pl.Int64),
            "field2": pl.Int64,
        }

        # Changed type for field2 to list so should fail
        with pytest.raises(pl.exceptions.InvalidOperationError):
            emitter.finalize()
        # Cleanup the real executor
        real_executor.shutdown()

    def test_variable_shape_detection_at_boundaries(self, temp_dir):
        """
        Test the fixed vs variable shape field detection logic specifically at
        the boundary points (start of sim, after disk write).
        """
        # Use a small batch size to quickly hit the boundary
        emitter = ParquetEmitter({"out_dir": temp_dir, "batch_size": 3})

        # Setup: Emit configuration data to intitialize variables
        config_data = {
            "table": "configuration",
            "data": {
                "experiment_id": "test_exp",
                "variant": 1,
                "lineage_seed": 1,
                "agent_id": "1",
            },
        }
        emitter.emit(config_data)

        # PHASE 1: Test at start of sim (first emit)
        # Start with a variable-shape field, the code should assume it's fixed-shape
        sim_data1 = {
            "table": "simulation",
            "data": {
                "time": 1.0,
                "agents": {
                    "agent1": {
                        "dynamic_array": np.array([1, 2, 3]),
                        "subtle_array": np.array([[1], [2], [3]]),
                    }
                },
            },
        }
        emitter.emit(sim_data1)

        # Verify it was stored as a fixed-shape numpy array
        assert isinstance(emitter.buffered_emits["dynamic_array"], np.ndarray)
        assert emitter.buffered_emits["dynamic_array"].shape[1:] == (3,)

        # PHASE 2: Immediately send different shape data
        # This should trigger conversion to variable-shape
        sim_data2 = {
            "table": "simulation",
            "data": {
                "time": 2.0,
                "agents": {
                    "agent1": {
                        "dynamic_array": np.array([1, 2, 3, 4, 5]),
                        "subtle_array": np.array([[1], [2], [3]]),
                    }
                },
            },
        }
        emitter.emit(sim_data2)

        # Verify it was converted to a list
        assert isinstance(emitter.buffered_emits["dynamic_array"], list)

        # PHASE 3: Send one more emit to trigger batch writing
        sim_data3 = {
            "table": "simulation",
            "data": {
                "time": 3.0,
                "agents": {
                    "agent1": {
                        "dynamic_array": np.array([1, 2, 3, 4, 5, 6, 7]),
                        "subtle_array": np.array([[1], [2], [3]]),
                    }
                },
            },
        }
        emitter.emit(sim_data3)

        # PHASE 4: subtle_array changed shape but we are at the start of a new batch.
        # ParquetEmitter is designed to assume all arrays are fixed-shape until
        # proven otherwise (better performance and memory usage). The distinction
        # between a fixed-shape and variable-shape array is purely an implmentation
        # detail of the buffering logic. In the end, both are written to Parquet
        # as variable-shape arrays. As such, even though subtle_array as a whole
        # is variable in shape, as long as it takes on a consistent shape within
        # a batch of emits, ParquetEmitter is more than happy to treat it as
        # fixed-shape and reap the performance benefits.
        sim_data4 = {
            "table": "simulation",
            "data": {
                "time": 4.0,
                "agents": {
                    "agent1": {
                        "dynamic_array": np.array([1]),
                        "subtle_array": np.array([[1], [2], [3], [4], [5]]),
                    }
                },
            },
        }
        emitter.emit(sim_data4)

        # Should be instantiated as variable length from last batch
        assert isinstance(emitter.buffered_emits["dynamic_array"], list)
        assert emitter.buffered_emits["dynamic_array"][0].to_list() == [1]
        # Should be treated as fixed-shape still
        assert isinstance(emitter.buffered_emits["subtle_array"], np.ndarray)
        assert emitter.buffered_emits["subtle_array"].shape[1:] == (5, 1)

        # Trigger another Parquet write and read them both back to confirm
        emitter.emit(sim_data4)
        emitter.emit(sim_data4)

        emitter.last_batch_future.result()
        t = pl.read_parquet(
            os.path.join(
                emitter.out_uri,
                emitter.experiment_id,
                "history",
                emitter.partitioning_path,
                "*.pq",
            )
        )
        assert t["dynamic_array"].to_list() == [
            [1, 2, 3],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6, 7],
            [1],
            [1],
            [1],
        ]
        assert t["subtle_array"].to_list() == [
            [[1], [2], [3]],
            [[1], [2], [3]],
            [[1], [2], [3]],
            [[1], [2], [3], [4], [5]],
            [[1], [2], [3], [4], [5]],
            [[1], [2], [3], [4], [5]],
        ]

    def test_expected_failures(self, temp_dir):
        """
        Test a few cases that are expected to fail.
        """
        # Use a small batch size to quickly hit the boundary
        emitter = ParquetEmitter({"out_dir": temp_dir, "batch_size": 3})

        # Setup: Emit configuration data to intitialize variables
        config_data = {
            "table": "configuration",
            "data": {
                "experiment_id": "test_exp",
                "variant": 1,
                "lineage_seed": 1,
                "agent_id": "1",
            },
        }
        emitter.emit(config_data)

        # Test initial null/empty list (null types), empty NumPy array (typed)
        sim_data1 = {
            "table": "simulation",
            "data": {
                "time": 1.0,
                "agents": {
                    "agent1": {
                        "init_null": None,
                        "init_empty_list": [],
                        "init_empty_array": np.array([]),
                        # Setup initial value for later test
                        "3d_array": np.random.rand(2, 3, 4),
                        "another_3d_array": np.random.rand(2, 3, 4),
                    }
                },
            },
        }
        emitter.emit(sim_data1)

        assert isinstance(emitter.buffered_emits["init_null"], list)
        assert emitter.buffered_emits["init_null"][0] is None
        assert emitter.pl_types["init_null"] == pl.Null
        assert isinstance(emitter.buffered_emits["init_empty_list"], list)
        assert emitter.buffered_emits["init_empty_list"][0].dtype == pl.Null
        assert emitter.pl_types["init_empty_list"] == pl.List(pl.Null)
        assert isinstance(emitter.buffered_emits["init_empty_array"], np.ndarray)
        assert emitter.buffered_emits["init_empty_array"].dtype == np.float64
        assert emitter.pl_types["init_empty_array"] == pl.List(pl.Float64)

        # Try adding another dimension to empty array
        sim_data2 = {
            "table": "simulation",
            "data": {
                "time": 2.0,
                "agents": {
                    "agent1": {
                        "init_empty_array": [[]],
                    }
                },
            },
        }
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Incompatible inner types for field init_empty_array: Float64 and List(Null)."
            ),
        ):
            emitter.emit(sim_data2)

        # Try adding 2D array with non-nulls to 3D array field
        sim_data3 = {
            "table": "simulation",
            "data": {
                "time": 2.0,
                "agents": {
                    "agent1": {
                        "3d_array": [[1.0, 2.0]],
                    }
                },
            },
        }
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Incompatible inner types for field 3d_array: List(Float64) and Float64."
            ),
        ):
            emitter.emit(sim_data3)

        # Try NumPy unsupported datetime64 resolution
        sim_data4 = {
            "table": "simulation",
            "data": {
                "time": 3.0,
                "agents": {
                    "agent1": {
                        "datetime_arr": np.array(
                            [
                                np.datetime64("2023-01-01T01"),
                                np.datetime64("2023-01-02T02:02"),
                            ]
                        ),
                    }
                },
            },
        }
        with pytest.raises(ValueError, match="incorrect NumPy datetime resolution"):
            emitter.emit(sim_data4)

        # Try NumPy void
        sim_data5 = {
            "table": "simulation",
            "data": {
                "time": 4.0,
                "agents": {
                    "agent1": {
                        "npvoid": np.array([b"test bytes"], dtype=np.void)[0],
                    }
                },
            },
        }
        with pytest.raises(ValueError):
            emitter.emit(sim_data5)

        # Try NumPy datetime64 in Python list
        sim_data6 = {
            "table": "simulation",
            "data": {
                "time": 3.0,
                "agents": {
                    "agent1": {
                        "mixed_datetime": [
                            np.datetime64("2023-01-01"),
                            np.datetime64("2023-01-02"),
                        ],
                    }
                },
            },
        }
        with pytest.raises(
            TypeError, match=re.escape("not yet implemented: Nested object types")
        ):
            emitter.emit(sim_data6)

        # Try list of NumPy arrays
        sim_data7 = {
            "table": "simulation",
            "data": {
                "time": 3.0,
                "agents": {
                    "agent1": {
                        "mixed_nested": [
                            np.array([1, 2, 3]),
                            np.array([4, 5]),
                            [6],
                            None,
                        ],
                    }
                },
            },
        }
        with pytest.raises(
            TypeError,
            match=re.escape("failed to determine supertype of object and list[i64]"),
        ):
            emitter.emit(sim_data7)

        # Try shape-varying 3D NumPy array
        # Polars can gracefully handle nested Python lists wihout explicit type
        # information, but not NumPy arrays. For example:
        # WORKS: pl.Series([[[1, 2], [3, 4]]])
        # FAILS: pl.Series([np.array([[1, 2], [3, 4]])])
        # This is thankfully not an issue for 1D NumPy arrays, which are the
        # only type of ragged NumPy arrays in vEcoli. I do not think it
        # makes much sense to have a ND NumPy array field with variable
        # shape anyways as it would still be constrained to a data cube.
        # Nested Python lists would let you deviate from a strict data cube
        # and even have null values, if desired.
        sim_data8 = {
            "table": "simulation",
            "data": {
                "time": 3.0,
                "agents": {
                    "agent1": {
                        "another_3d_array": np.zeros((10, 10, 10)),
                    }
                },
            },
        }
        with pytest.raises(
            ValueError,
            match=re.escape("cannot parse numpy data type dtype('O')"),
        ):
            emitter.emit(sim_data8)

    def test_nested_nullable(self, temp_dir):
        """Test handling nullable nested types that increase in depth."""
        emitter = ParquetEmitter({"out_dir": temp_dir, "batch_size": 4})
        # Configuration emit to initialize variables
        config_data = {
            "table": "configuration",
            "data": {
                "experiment_id": "test_exp",
                "variant": 1,
                "lineage_seed": 1,
                "agent_id": "1",
            },
        }
        emitter.emit(config_data)
        emitter.last_batch_future.result()

        sim_data1 = {
            "table": "simulation",
            "data": {
                "time": 0.0,
                "agents": {
                    "agent1": {
                        "nullable_nested": None,
                    }
                },
            },
        }
        emitter.emit(sim_data1)
        emitter.last_batch_future.result()

        # Verify arrays were stored correctly
        assert isinstance(emitter.buffered_emits["nullable_nested"], list)
        assert emitter.buffered_emits["nullable_nested"][0] is None
        assert emitter.pl_types["nullable_nested"] == pl.Null

        sim_data2 = {
            "table": "simulation",
            "data": {
                "time": 1.0,
                "agents": {
                    "agent1": {
                        "nullable_nested": [None, None],
                    }
                },
            },
        }
        emitter.emit(sim_data2)
        emitter.last_batch_future.result()

        # Verify arrays were stored correctly
        assert isinstance(emitter.buffered_emits["nullable_nested"], list)
        assert emitter.buffered_emits["nullable_nested"][1].dtype == pl.Null
        assert emitter.pl_types["nullable_nested"] == pl.List(pl.Null)

        # One level deeper
        sim_data3 = {
            "table": "simulation",
            "data": {
                "time": 2.0,
                "agents": {
                    "agent1": {
                        "nullable_nested": [None, [None], [], [None, None], []],
                    }
                },
            },
        }
        emitter.emit(sim_data3)
        emitter.last_batch_future.result()

        # Verify arrays were stored correctly
        assert emitter.buffered_emits["nullable_nested"][2].dtype == pl.List(pl.Null)
        assert emitter.pl_types["nullable_nested"] == pl.List(pl.List(pl.Null))

        # One level deeper and defines non-null values
        sim_data4 = {
            "table": "simulation",
            "data": {
                "time": 2.0,
                "agents": {
                    "agent1": {
                        "nullable_nested": [
                            [],
                            [["wow", "this", "is"], [], ["deep"], None],
                            None,
                            [[], None],
                        ],
                    }
                },
            },
        }
        emitter.emit(sim_data4)
        emitter.last_batch_future.result()

        # Check output
        t = pl.read_parquet(
            os.path.join(
                emitter.out_uri,
                emitter.experiment_id,
                "history",
                emitter.partitioning_path,
                "4.pq",
            )
        )
        assert t["nullable_nested"].to_list() == [
            None,
            [None, None],
            [None, [None], [], [None, None], []],
            [[], [["wow", "this", "is"], [], ["deep"], None], None, [[], None]],
        ]

        emitter.emit(sim_data1)
        emitter.last_batch_future.result()
        # Should remember that we fully defined the type before
        assert isinstance(emitter.buffered_emits["nullable_nested"], list)
        assert emitter.buffered_emits["nullable_nested"][0] is None
        assert emitter.pl_types["nullable_nested"] == pl.List(
            pl.List(pl.List(pl.String))
        )

        # Emit until batch size and check output
        for _ in range(3):
            emitter.emit(sim_data1)
            emitter.last_batch_future.result()

        # Check output
        t = pl.read_parquet(
            os.path.join(
                emitter.out_uri,
                emitter.experiment_id,
                "history",
                emitter.partitioning_path,
                "8.pq",
            )
        )
        assert t["nullable_nested"].to_list() == [None] * 4
        assert t["nullable_nested"].dtype == pl.List(pl.List(pl.List(pl.String)))
