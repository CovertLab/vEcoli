import os
import tempfile
import shutil
import numpy as np
import polars as pl
import pytest
import time
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue

from ecoli.library.parquet_emitter import (
    json_to_parquet,
    get_encoding,
    flatten_dict,
    ParquetEmitter,
)


class TestHelperFunctions:
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

    def test_get_encoding(self):
        # Basic types
        np_type = get_encoding(1.0, "float_field")
        assert np_type == np.float64

        np_type = get_encoding(True, "bool_field")
        assert np_type == np.bool_

        np_type = get_encoding("text", "string_field")
        assert np_type == np.dtypes.StringDType

        # Integer with different encodings
        np_type = get_encoding(42, "int_field")
        assert np_type == np.int64

        np_type = get_encoding(42, "uint16_field", use_uint16=True)
        assert np_type == np.uint16

        np_type = get_encoding(42, "uint32_field", use_uint32=True)
        assert np_type == np.uint32

        # Arrays with various dimensions
        np_type = get_encoding(np.array([1, 2, 3]), "array1d_field")
        assert np_type == np.int64

        np_type = get_encoding(np.array([[1, 2], [3, 4]]), "array2d_field")
        assert np_type == np.int64

        # Empty arrays still have a dtype
        np_type = get_encoding(np.array([]), "empty_array_field")
        assert np_type == np.float64

        # Empty lists do not have a dtype and are skipped
        np_type = get_encoding([], "empty_list_field")
        assert np_type is None

        # None values
        np_type = get_encoding(None, "none_field")
        assert np_type is None

        # Invalid types
        with pytest.raises(TypeError):
            get_encoding(complex(1, 2), "complex_field")


class TestParquetEmitter:
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        tmp = tempfile.mkdtemp()
        yield tmp
        shutil.rmtree(tmp)

    @patch("ecoli.library.parquet_emitter.url_to_fs")
    def test_initialization(self, mock_url_to_fs, temp_dir):
        """Test ParquetEmitter initialization with different configs."""

        mock_fs = Mock()
        mock_url_to_fs.return_value = (mock_fs, temp_dir)

        # Test with out_dir
        emitter = ParquetEmitter({"out_dir": temp_dir})
        assert emitter.out_uri == os.path.abspath(temp_dir)
        assert emitter.filesystem == mock_fs
        assert emitter.batch_size == 400

        # Test with out_uri and custom batch size
        emitter = ParquetEmitter({"out_uri": "gs://bucket/path", "batch_size": 100})
        assert emitter.out_uri == "gs://bucket/path"
        assert emitter.batch_size == 100

    @patch("ecoli.library.parquet_emitter.url_to_fs")
    def test_emit_configuration(self, mock_url_to_fs, temp_dir):
        """Test emitting configuration data."""
        mock_fs = Mock()
        mock_url_to_fs.return_value = (mock_fs, temp_dir)

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

        # Verify filesystem operations
        mock_fs.delete.assert_called()
        mock_fs.makedirs.assert_called()

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

    @patch("ecoli.library.parquet_emitter.url_to_fs")
    def test_variable_length_arrays(self, mock_url_to_fs, temp_dir):
        """Test handling arrays with changing dimensions."""
        mock_fs = Mock()
        mock_url_to_fs.return_value = (mock_fs, temp_dir)

        emitter = ParquetEmitter({"out_dir": temp_dir})
        emitter.experiment_id = "test_exp"
        emitter.partitioning_path = "path/to/output"

        # First emit with 3-element array
        sim_data1 = {
            "table": "simulation",
            "data": {
                "time": 1.0,
                "agents": {"agent1": {"dynamic_array": np.array([1, 2, 3])}},
            },
        }
        emitter.emit(sim_data1)
        emitter.last_batch_future.result()

        # Verify array was stored correctly
        assert "dynamic_array" in emitter.buffered_emits
        assert emitter.buffered_emits["dynamic_array"].shape[1:] == (3,)

        # Second emit with 4-element array (different shape)
        sim_data2 = {
            "table": "simulation",
            "data": {
                "time": 2.0,
                "agents": {
                    "agent1": {
                        "dynamic_array": np.array([4, 5, 6, 7])  # Different length
                    }
                },
            },
        }
        emitter.emit(sim_data2)
        emitter.last_batch_future.result()

        # Verify conversion to variable length type
        assert "dynamic_array" in emitter.var_len_dims
        assert isinstance(emitter.buffered_emits["dynamic_array"], list)
        assert emitter.buffered_emits["dynamic_array"][0].tolist() == [1, 2, 3]
        assert emitter.buffered_emits["dynamic_array"][1].tolist() == [4, 5, 6, 7]

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
                        "zero_dim_array": np.array(42),  # 0-d array
                        "unicode_string": "Unicode: 日本語",
                        "very_long_string": "x" * 10000,
                        # Nested structures
                        "deep_nesting": {"level1": {"level2": {"level3": [1, 2, 3]}}},
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
                        "zero_dim_array": np.array(np.inf),  # 0-d array
                        "unicode_string": "Unicode: 日本語 再び",
                        "very_long_string": "x" * 100000,
                        # Nested structures
                        "deep_nesting": {
                            "level1": {
                                "level2": {"level3": [1, 2, 3, 4], "level4": [5, 6, 7]}
                            }
                        },
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
            "max_int": np.array(
                [np.iinfo(np.int64).max, np.iinfo(np.int64).min], dtype=np.int64
            ),
            "min_int": np.array(
                [np.iinfo(np.int64).min, np.iinfo(np.int64).max], dtype=np.int64
            ),
            "max_float": np.array(
                [np.finfo(np.float64).max, np.finfo(np.float64).min], dtype=np.float64
            ),
            "tiny_float": np.array([1e-100, 1e100], dtype=np.float64),
            "nan_value": np.array([np.nan, np.inf], dtype=np.float64),
            "inf_value": np.array([np.inf, np.nan], dtype=np.float64),
            "empty_array": np.array([np.array([]), np.array([np.nan])], dtype=object),
            # Silently incorrectly stores np.inf
            "zero_dim_array": np.array([42, -9223372036854775808], dtype=np.int64),
            "unicode_string": np.array(["Unicode: 日本語", "Unicode: 日本語 再び"]),
            "very_long_string": np.array(["x" * 10000, "x" * 100000]),
            "deep_nesting__level1__level2__level3": np.array(
                [np.array([1, 2, 3], dtype=int), np.array([1, 2, 3, 4], dtype=int)],
                dtype=object,
            ),
            "deep_nesting__level1__level2__level4": np.array(
                [np.array([], dtype=int), np.array([5, 6, 7], dtype=int)], dtype=object
            ),
        }
        for key, value in output_data.items():
            try:
                assert np.array_equal(
                    output_pl[key].to_numpy(), value, equal_nan=True
                ), f"Mismatch for key: {key}"
                print("basic", key, output_pl[key].to_numpy(), value)
            except TypeError:
                for pq, vl in zip(output_pl[key].to_numpy(), value):
                    try:
                        assert np.array_equal(pq, vl, equal_nan=True), (
                            f"Mismatch for key: {key} with value: {vl}"
                        )
                        print("zip", key, output_pl[key].to_numpy(), value)
                    except TypeError:
                        assert pq == vl, f"Mismatch for key: {key} with value: {vl}"
                        print("direct", key, output_pl[key].to_numpy(), value)

    @patch("ecoli.library.parquet_emitter.url_to_fs")
    def test_finalize(self, mock_url_to_fs, temp_dir):
        """Test _finalize method that handles remaining data."""
        mock_fs = Mock()
        mock_fs.exists.return_value = False
        mock_url_to_fs.return_value = (mock_fs, temp_dir)

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
            # Test _finalize
            emitter._finalize()

            # Verify json_to_parquet was called with truncated data
            mock_json_to_parquet.assert_called_once()
            args, _ = mock_json_to_parquet.call_args
            assert args[0]["field1"].shape[0] == 1  # Only 1 item should remain
            assert args[0]["field1"][0] == 10
            assert args[0]["field2"][0] == 20.5

        # Test success flag
        emitter.success = True
        with patch("polars.DataFrame.write_parquet") as mock_write:
            emitter._finalize()
            # Verify success file was written
            mock_write.assert_called()

    @patch("ecoli.library.parquet_emitter.url_to_fs")
    def test_multiple_agents(self, mock_url_to_fs, temp_dir):
        """Test that multi-agent data is ignored."""
        mock_fs = Mock()
        mock_url_to_fs.return_value = (mock_fs, temp_dir)

        emitter = ParquetEmitter({"out_dir": temp_dir})

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
                        f"field{i}": np.array([7, 8, 9, 10]) for i in range(3, 30)
                    }
                },
            },
        }
        emitter.emit(sim_data3)

        # # Verify the captured data matches what was in buffers before clearing
        captured_data, captured_types = data_capture_queue.get(timeout=1)
        assert captured_data["experiment_id"] == "test_exp"
        assert captured_data["variant"] == 1
        assert captured_data["lineage_seed"] == 1

        captured_data, captured_types = data_capture_queue.get(timeout=1)
        assert len(captured_data["field1"]) == emitter.batch_size
        assert captured_data["field1"][0].tolist() == [1, 2, 3]
        assert captured_data["field1"][1].tolist() == [4, 5, 6]

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

        # PHASE 4: Test after disk write with 2D array instead of expected 1D
        sim_data4 = {
            "table": "simulation",
            "data": {
                "time": 4.0,
                "agents": {
                    "agent1": {
                        "dynamic_array": np.array([[1, 2], [3, 4]]),
                        "subtle_array": np.array([[1], [2], [3]]),
                    }
                },
            },
        }
        with pytest.raises(ValueError):
            emitter.emit(sim_data4)

        # PHASE 5: subtle_array changed shape but we are at the start of a new batch.
        # ParquetEmitter is designed to assume all arrays are fixed-shape until
        # proven otherwise (better performance and memory usage). The distinction
        # between a fixed-shape and variable-shape array is purely an implmentation
        # detail of the buffering logic. In the end, both are written to Parquet
        # as variable-shape arrays. As such, even though subtle_array as a whole
        # is variable in shape, as long as it takes on a consistent shape within
        # a batch of emits, ParquetEmitter is more than happy to treat it as
        # fixed-shape and reap the performance benefits.

        sim_data5 = {
            "table": "simulation",
            "data": {
                "time": 5.0,
                "agents": {
                    "agent1": {
                        "dynamic_array": np.array([1]),
                        "subtle_array": np.array([[1], [2], [3], [4], [5]]),
                    }
                },
            },
        }
        emitter.emit(sim_data5)

        # Should be instantiated as variable length from last batch
        assert isinstance(emitter.buffered_emits["dynamic_array"], list)
        for dyn_arr in emitter.buffered_emits["dynamic_array"]:
            assert isinstance(dyn_arr, np.ndarray) and dyn_arr.ndim == 1
        assert emitter.buffered_emits["dynamic_array"][0].tolist() == [1]
        # Should be treated as fixed-shape still
        assert isinstance(emitter.buffered_emits["subtle_array"], np.ndarray)
        assert emitter.buffered_emits["subtle_array"].shape[1:] == (5, 1)

        # Trigger another Parquet write and read them both back to confirm
        emitter.emit(sim_data5)
        emitter.emit(sim_data5)

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
