"""
Comprehensive tests for analysis.py refactored functions
"""

import pytest
import json
import warnings
import importlib
from types import ModuleType
from runscripts.analysis import (
    parse_variant_data_dir,
    make_sim_data_dict,
    build_duckdb_filter,
    load_variant_metadata,
    filter_variant_dicts,
    build_query_strings,
    run_analysis_loop,
)


class MockConnection:
    def __init__(self, return_data: list[tuple[str, int]]):
        self.return_data = return_data

    def sql(conn, query):
        class MockResult:
            def fetchall(self):
                # Return test data: (exp_id, variant)
                return conn.return_data

        return MockResult()


class TestParseVariantDataDir:
    """Tests for parse_variant_data_dir function"""

    def test_parse_single_experiment(self, tmp_path):
        """Test parsing variant data directory for a single experiment"""
        exp_id = "test_exp"
        variant_dir = tmp_path / "variants"
        variant_dir.mkdir()

        metadata = {
            "test_variant": {
                "0": {"param1": "value1"},
                "1": {"param1": "value2"},
            }
        }
        metadata_file = variant_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

        # Create variant sim_data files
        (variant_dir / "0.cPickle").touch()
        (variant_dir / "1.cPickle").touch()

        v_metadata, sim_data_dict, v_names = parse_variant_data_dir(
            [exp_id], [str(variant_dir)]
        )

        assert exp_id in v_metadata
        assert 0 in v_metadata[exp_id]
        assert 1 in v_metadata[exp_id]
        assert v_metadata[exp_id][0]["param1"] == "value1"
        assert v_metadata[exp_id][1]["param1"] == "value2"
        assert exp_id in sim_data_dict
        assert 0 in sim_data_dict[exp_id]
        assert 1 in sim_data_dict[exp_id]
        assert v_names[exp_id] == "test_variant"

    def test_parse_multiple_experiments(self, tmp_path):
        """Test parsing variant data directories for multiple experiments"""
        exp_ids = ["exp1", "exp2"]
        variant_dirs = []

        for i, exp_id in enumerate(exp_ids):
            variant_dir = tmp_path / f"variants_{i}"
            variant_dir.mkdir()
            variant_dirs.append(str(variant_dir))

            metadata = {
                f"variant_{exp_id}": {
                    "0": {"param1": f"value_{exp_id}_0"},
                    "1": {"param1": f"value_{exp_id}_1"},
                }
            }
            metadata_file = variant_dir / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f)

            (variant_dir / "0.cPickle").touch()
            (variant_dir / "1.cPickle").touch()

        v_metadata, sim_data_dict, v_names = parse_variant_data_dir(
            exp_ids, variant_dirs
        )

        # Check both experiments are present
        assert "exp1" in v_metadata
        assert "exp2" in v_metadata
        assert "exp1" in sim_data_dict
        assert "exp2" in sim_data_dict

        # Check variant data for exp1
        assert v_metadata["exp1"][0]["param1"] == "value_exp1_0"
        assert v_metadata["exp1"][1]["param1"] == "value_exp1_1"
        assert v_names["exp1"] == "variant_exp1"

        # Check variant data for exp2
        assert v_metadata["exp2"][0]["param1"] == "value_exp2_0"
        assert v_metadata["exp2"][1]["param1"] == "value_exp2_1"
        assert v_names["exp2"] == "variant_exp2"

    def test_parse_no_variant_files(self, tmp_path):
        """Test parsing when no variant files are present"""
        exp_id = "test_exp"
        variant_dir = tmp_path / "variants"
        variant_dir.mkdir()

        metadata = {"test_variant": {}}
        metadata_file = variant_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

        v_metadata, sim_data_dict, v_names = parse_variant_data_dir(
            [exp_id], [str(variant_dir)]
        )

        assert exp_id in v_metadata
        assert len(v_metadata[exp_id]) == 0
        assert exp_id in sim_data_dict
        assert len(sim_data_dict[exp_id]) == 0
        assert v_names[exp_id] == "test_variant"


class TestMakeSimDataDict:
    """Tests for make_sim_data_dict function"""

    def test_make_sim_data_dict_basic(self):
        """Test creating sim_data_dict with basic inputs"""
        exp_id = "test_exp"
        variants = [0, 1, 2]
        sim_data_paths = ["/path/0.cPickle", "/path/1.cPickle", "/path/2.cPickle"]

        result = make_sim_data_dict(exp_id, variants, sim_data_paths)

        assert exp_id in result
        assert len(result[exp_id]) == 3
        assert result[exp_id][0] == "/path/0.cPickle"
        assert result[exp_id][1] == "/path/1.cPickle"
        assert result[exp_id][2] == "/path/2.cPickle"

    def test_make_sim_data_dict_empty_variants(self):
        """Test that empty variants list raises ValueError"""
        exp_id = "test_exp"
        variants = []
        sim_data_paths = []

        with pytest.raises(ValueError, match="Must specify variant"):
            make_sim_data_dict(exp_id, variants, sim_data_paths)

    def test_make_sim_data_dict_mismatched_lengths(self):
        """Test that mismatched variant and path lengths raises ValueError"""
        exp_id = "test_exp"
        variants = [0, 1, 2]
        sim_data_paths = ["/path/0.cPickle", "/path/1.cPickle"]

        with pytest.raises(
            ValueError, match="Must specify sim_data_path for each variant"
        ):
            make_sim_data_dict(exp_id, variants, sim_data_paths)

    def test_make_sim_data_dict_single_variant(self):
        """Test creating sim_data_dict with a single variant"""
        exp_id = "test_exp"
        variants = [5]
        sim_data_paths = ["/path/5.cPickle"]

        result = make_sim_data_dict(exp_id, variants, sim_data_paths)

        assert exp_id in result
        assert len(result[exp_id]) == 1
        assert result[exp_id][5] == "/path/5.cPickle"


class TestBuildDuckdbFilter:
    """Tests for build_duckdb_filter function"""

    def test_single_string_filter(self):
        """Test building filter with single string value"""
        config = {"experiment_id": ["test_exp"]}
        result = build_duckdb_filter(config)
        expected = "experiment_id = 'test_exp'"
        assert result == expected

    def test_multiple_string_filters(self):
        """Test building filter with multiple string values"""
        config = {"experiment_id": ["exp1", "exp2", "exp3"]}
        result = build_duckdb_filter(config)
        assert "experiment_id IN ('exp1', 'exp2', 'exp3')" == result

    def test_single_int_filter(self):
        """Test building filter with single int value"""
        config = {"variant": [5]}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = build_duckdb_filter(config)
            assert result == "variant = 5"
            assert len(w) == 1
            assert "applicable data for the skipped" in str(w[0].message).lower()

    def test_multiple_int_filters(self):
        """Test building filter with multiple int values"""
        config = {"variant": [0, 1, 2]}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = build_duckdb_filter(config)
            assert "variant IN (0, 1, 2)" == result
            assert len(w) == 1
            assert "applicable data for the skipped" in str(w[0].message).lower()

    def test_combined_filters(self):
        """Test building filter with multiple filter types"""
        config = {
            "experiment_id": ["test_exp"],
            "variant": [0, 1],
            "lineage_seed": [42],
        }
        result = build_duckdb_filter(config)
        assert "experiment_id = 'test_exp'" in result
        assert "variant IN (0, 1)" in result
        assert "lineage_seed = 42" in result
        assert " AND " in result

    def test_range_filter(self):
        """Test that range filters are converted to lists"""
        config = {"variant_range": [0, 5]}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = build_duckdb_filter(config)
            assert "variant IN (0, 1, 2, 3, 4)" == result
            # Check that variant list was created in config
            assert config["variant"] == [0, 1, 2, 3, 4]
            assert len(w) == 1
            assert "applicable data for the skipped" in str(w[0].message).lower()

    def test_range_precedence_over_value(self):
        """Test that range takes precedence over explicit values"""
        config = {
            "variant": [10, 20],
            "variant_range": [0, 3],
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = build_duckdb_filter(config)
            # There may be multiple warnings (range precedence + skipped filters)
            assert any(
                "Range takes precedence" in str(warning.message) for warning in w
            )

        assert "variant IN (0, 1, 2)" == result
        assert config["variant"] == [0, 1, 2]

    def test_skipped_filter_warning(self):
        """Test warning when filters are skipped"""
        config = {
            "experiment_id": ["test_exp"],
            "generation": [5],  # Skips variant and lineage_seed
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            build_duckdb_filter(config)
            assert len(w) == 1
            assert "skipped" in str(w[0].message).lower()

    def test_empty_config(self):
        """Test with no filters specified"""
        config = {}
        result = build_duckdb_filter(config)
        assert result == ""

    def test_url_encoding_in_string_filter(self):
        """Test that special characters in strings are URL-encoded"""
        config = {"experiment_id": ["test exp"]}  # Space should be encoded
        result = build_duckdb_filter(config)
        # URL encoding happens with parse.quote_plus
        assert result == "experiment_id = 'test+exp'"

    def test_agent_id_string_filter(self):
        """Test agent_id string filtering"""
        config = {"agent_id": ["agent_001", "agent_002"]}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = build_duckdb_filter(config)
            assert len(w) == 1
            assert "applicable data for the skipped" in str(w[0].message).lower()
            assert "agent_id IN ('agent_001', 'agent_002')" == result


class TestLoadVariantMetadata:
    """Tests for load_variant_metadata function"""

    def test_load_from_variant_data_dir(self, tmp_path):
        """Test loading metadata from variant_data_dir"""

        variant_dir = tmp_path / "variants"
        variant_dir.mkdir()

        metadata = {
            "test_variant": {
                "0": {"param": "value0"},
                "1": {"param": "value1"},
            }
        }
        with open(variant_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        (variant_dir / "0.cPickle").touch()
        (variant_dir / "1.cPickle").touch()

        config = {
            "experiment_id": ["test_exp"],
            "variant_data_dir": [str(variant_dir)],
        }

        v_metadata, sim_data_dict, v_names = load_variant_metadata(config)

        assert "test_exp" in v_metadata
        assert 0 in v_metadata["test_exp"]
        assert 1 in v_metadata["test_exp"]
        assert v_names["test_exp"] == "test_variant"

    def test_load_from_variant_metadata_path(self, tmp_path):
        """Test loading metadata from variant_metadata_path"""

        metadata_file = tmp_path / "metadata.json"
        metadata = {
            "test_variant": {
                "0": {"param": "value0"},
                "1": {"param": "value1"},
            }
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

        config = {
            "experiment_id": ["test_exp"],
            "variant_metadata_path": str(metadata_file),
            "variant": [0, 1],
            "sim_data_path": ["/path/0.cPickle", "/path/1.cPickle"],
        }

        v_metadata, sim_data_dict, v_names = load_variant_metadata(config)

        assert "test_exp" in v_metadata
        assert 0 in v_metadata["test_exp"]
        assert 1 in v_metadata["test_exp"]
        assert v_names["test_exp"] == "test_variant"

    def test_no_variant_metadata(self):
        """Test with no variant metadata provided"""
        config = {
            "experiment_id": ["test_exp"],
            "variant": [0],
            "sim_data_path": ["/path/0.cPickle"],
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            v_metadata, sim_data_dict, v_names = load_variant_metadata(config)
            assert len(w) == 1
            assert "No variant metadata" in str(w[0].message)

        assert "test_exp" in v_metadata
        assert len(v_metadata["test_exp"]) == 0

    def test_missing_experiment_id(self):
        """Test that missing experiment_id raises KeyError"""
        config = {}
        with pytest.raises(KeyError, match="experiment ID"):
            load_variant_metadata(config)

    def test_multiple_experiments_without_variant_data_dir(self):
        """Test that multiple experiments require variant_data_dir"""
        config = {"experiment_id": ["exp1", "exp2"]}
        with pytest.raises(AssertionError, match="variant_data_dir"):
            load_variant_metadata(config)

    def test_variant_data_dir_precedence(self, tmp_path):
        """Test that variant_data_dir takes precedence"""

        variant_dir = tmp_path / "variants"
        variant_dir.mkdir()

        metadata = {"test_variant": {"0": {"param": "value0"}}}
        with open(variant_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        (variant_dir / "0.cPickle").touch()

        metadata_file = tmp_path / "other_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump({"other": {}}, f)

        config = {
            "experiment_id": ["test_exp"],
            "variant_data_dir": [str(variant_dir)],
            "variant_metadata_path": str(metadata_file),
            "sim_data_path": ["/other/path.cPickle"],
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            v_metadata, sim_data_dict, v_names = load_variant_metadata(config)
            # Should warn about ignoring the other parameters
            assert len(w) == 2
            assert any("variant_metadata_path" in str(warning.message) for warning in w)
            assert any("sim_data_path" in str(warning.message) for warning in w)

        assert v_names["test_exp"] == "test_variant"


class TestFilterVariantDicts:
    """Tests for filter_variant_dicts function"""

    def test_filter_basic(self):
        """Test basic filtering"""
        variant_set = {("exp1", 0), ("exp1", 1)}
        variant_metadata = {
            "exp1": {
                0: {"param": "value0"},
                1: {"param": "value1"},
                2: {"param": "value2"},  # Should be filtered out
            }
        }
        sim_data_dict = {
            "exp1": {
                0: "/path/0.cPickle",
                1: "/path/1.cPickle",
                2: "/path/2.cPickle",
            }
        }
        variant_names = {"exp1": "test_variant"}

        (
            filtered_metadata,
            filtered_sim_data,
            filtered_names,
        ) = filter_variant_dicts(
            variant_set, variant_metadata, sim_data_dict, variant_names
        )

        assert len(filtered_metadata["exp1"]) == 2
        assert 0 in filtered_metadata["exp1"]
        assert 1 in filtered_metadata["exp1"]
        assert 2 not in filtered_metadata["exp1"]
        assert len(filtered_sim_data["exp1"]) == 2
        assert 0 in filtered_sim_data["exp1"]
        assert 1 in filtered_sim_data["exp1"]
        assert 2 not in filtered_sim_data["exp1"]
        assert filtered_names["exp1"] == "test_variant"

    def test_filter_multiple_experiments(self):
        """Test filtering across multiple experiments"""
        variant_set = {("exp1", 0), ("exp2", 1)}
        variant_metadata = {
            "exp1": {0: {"param": "val0"}, 1: {"param": "val1"}},
            "exp2": {0: {"param": "val0"}, 1: {"param": "val1"}},
        }
        sim_data_dict = {
            "exp1": {0: "/path/exp1_0.cPickle", 1: "/path/exp1_1.cPickle"},
            "exp2": {0: "/path/exp2_0.cPickle", 1: "/path/exp2_1.cPickle"},
        }
        variant_names = {"exp1": "variant1", "exp2": "variant2"}

        (
            filtered_metadata,
            filtered_sim_data,
            filtered_names,
        ) = filter_variant_dicts(
            variant_set, variant_metadata, sim_data_dict, variant_names
        )

        assert "exp1" in filtered_metadata
        assert "exp2" in filtered_metadata
        assert len(filtered_metadata["exp1"]) == 1
        assert len(filtered_metadata["exp2"]) == 1
        assert 0 in filtered_metadata["exp1"]
        assert 1 in filtered_metadata["exp2"]
        assert "exp1" in filtered_sim_data
        assert "exp2" in filtered_sim_data
        assert len(filtered_sim_data["exp1"]) == 1
        assert len(filtered_sim_data["exp2"]) == 1
        assert 0 in filtered_sim_data["exp1"]
        assert 1 in filtered_sim_data["exp2"]
        assert filtered_names == variant_names

    def test_filter_with_missing_metadata(self):
        """Test filtering when some variants lack metadata"""
        variant_set = {("exp1", 0), ("exp1", 1)}
        variant_metadata = {"exp1": {0: {"param": "value0"}}}  # Missing variant 1
        sim_data_dict = {"exp1": {0: "/path/0.cPickle", 1: "/path/1.cPickle"}}
        variant_names = {"exp1": "test_variant"}

        (
            filtered_metadata,
            filtered_sim_data,
            filtered_names,
        ) = filter_variant_dicts(
            variant_set, variant_metadata, sim_data_dict, variant_names
        )

        # Metadata should only have variant 0
        assert len(filtered_metadata["exp1"]) == 1
        assert 0 in filtered_metadata["exp1"]
        # But sim_data should have both
        assert len(filtered_sim_data["exp1"]) == 2
        assert variant_names == filtered_names

    def test_filter_empty_set(self):
        """Test with empty variant set"""
        variant_set = set()
        variant_metadata = {"exp1": {0: {"param": "value0"}}}
        sim_data_dict = {"exp1": {0: "/path/0.cPickle"}}
        variant_names = {"exp1": "test_variant"}

        (
            filtered_metadata,
            filtered_sim_data,
            filtered_names,
        ) = filter_variant_dicts(
            variant_set, variant_metadata, sim_data_dict, variant_names
        )

        assert len(filtered_metadata) == 0
        assert len(filtered_sim_data) == 0
        assert len(filtered_names) == 0


class TestBuildQueryStrings:
    """Tests for build_query_strings function"""

    def test_build_with_id_cols(self):
        """Test building query strings for analysis type with id_cols"""
        # Create a mock connection that returns test data
        conn = MockConnection([("exp1", 0), ("exp1", 1)])
        analysis_type = "multiseed"  # Has id_cols = ["experiment_id", "variant"]
        duckdb_filter = "experiment_id = 'exp1'"
        config_sql = "SELECT * FROM config"
        history_sql = "SELECT * FROM history"
        success_sql = "SELECT * FROM success"
        outdir = "/tmp/test_output"

        query_strings = build_query_strings(
            analysis_type,
            duckdb_filter,
            config_sql,
            history_sql,
            success_sql,
            outdir,
            conn,
        )

        # Should have one entry for (exp1, 0) and one for (exp1, 1)
        assert len(query_strings) == 2

    def test_build_without_id_cols(self):
        """Test building query strings for analysis type without id_cols"""

        conn = MockConnection([("exp1", 0), ("exp1", 1)])
        analysis_type = "multiexperiment"  # Has id_cols = []
        duckdb_filter = "experiment_id = 'exp1'"
        config_sql = "SELECT * FROM config"
        history_sql = "SELECT * FROM history"
        success_sql = "SELECT * FROM success"
        outdir = "/tmp/test_output"

        query_strings = build_query_strings(
            analysis_type,
            duckdb_filter,
            config_sql,
            history_sql,
            success_sql,
            outdir,
            conn,
        )

        # Should have single entry with the base filter
        assert len(query_strings) == 1
        assert duckdb_filter in query_strings

    def test_query_string_structure(self):
        """Test that query strings have correct structure"""

        conn = MockConnection([("exp1", 0)])
        analysis_type = "multiseed"
        duckdb_filter = "experiment_id = 'exp1'"
        config_sql = "SELECT * FROM config"
        history_sql = "SELECT * FROM history"
        success_sql = "SELECT * FROM success"
        outdir = "/tmp/test_output"

        query_strings = build_query_strings(
            analysis_type,
            duckdb_filter,
            config_sql,
            history_sql,
            success_sql,
            outdir,
            conn,
        )

        # Get first (and only) query string tuple
        key = list(query_strings.keys())[0]
        assert key == "experiment_id='exp1' AND variant=0"
        history_q, config_q, success_q, curr_outdir, variant_set = query_strings[key]

        # Check structure
        filter = "WHERE experiment_id='exp1' AND variant=0"
        assert filter in history_q
        assert filter in config_q
        assert filter in success_q
        assert curr_outdir == f"{outdir}/experiment_id=exp1/variant=0"
        assert isinstance(variant_set, set)
        assert ("exp1", 0) in variant_set


class TestIntegration:
    """Integration tests that test multiple functions together"""

    def test_full_workflow_simple(self, tmp_path):
        """Test a simple end-to-end workflow"""

        # Set up test data
        variant_dir = tmp_path / "variants"
        variant_dir.mkdir()

        metadata = {
            "test_variant": {
                "0": {"param": "value0"},
                "1": {"param": "value1"},
            }
        }
        with open(variant_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        (variant_dir / "0.cPickle").touch()
        (variant_dir / "1.cPickle").touch()

        # Build config
        config = {
            "experiment_id": ["test_exp"],
            "variant": [0, 1],
            "variant_data_dir": [str(variant_dir)],
        }

        # Build filter
        duckdb_filter = build_duckdb_filter(config)
        assert "experiment_id = 'test_exp' AND variant IN (0, 1)" == duckdb_filter

        # Load metadata
        v_metadata, sim_data_dict, v_names = load_variant_metadata(config)
        assert len(v_metadata["test_exp"]) == 2

        # Filter for variant 0 only
        variant_set = {("test_exp", 0)}
        filtered_metadata, filtered_sim_data, filtered_names = filter_variant_dicts(
            variant_set, v_metadata, sim_data_dict, v_names
        )

        assert len(filtered_metadata["test_exp"]) == 1
        assert 0 in filtered_metadata["test_exp"]
        assert 1 not in filtered_metadata["test_exp"]


class TestRunAnalysisLoop:
    """Tests for run_analysis_loop function"""

    def test_run_with_mock_analysis(self, tmp_path, monkeypatch):
        """Test running analysis loop with mocked analysis module"""
        # Create mock analysis module
        mock_analysis = ModuleType("mock_analysis")
        mock_analysis.plot_calls = []

        def mock_plot(*args, **kwargs):
            mock_analysis.plot_calls.append({"args": args, "kwargs": kwargs})

        mock_analysis.plot = mock_plot

        # Mock importlib.import_module
        def mock_import(module_name):
            if "test_analysis" in module_name:
                return mock_analysis
            raise ImportError(f"No module named {module_name}")

        monkeypatch.setattr(importlib, "import_module", mock_import)

        # Create mock connection
        conn = MockConnection([("test_exp", 0)])

        # Set up config
        outdir = tmp_path / "output"
        outdir.mkdir()

        config = {
            "outdir": str(outdir),
            "analysis_types": ["multiseed"],
            "multiseed": {"test_analysis": {}},
            "validation_data_path": [],
        }

        variant_metadata = {"test_exp": {0: {"param": "value"}}}
        sim_data_dict = {"test_exp": {0: "/path/0.cPickle"}}
        variant_names = {"test_exp": "test_variant"}

        # Run analysis loop
        stats = run_analysis_loop(
            config,
            conn,
            "SELECT * FROM history",
            "SELECT * FROM config",
            "SELECT * FROM success",
            "experiment_id = 'test_exp'",
            variant_metadata,
            sim_data_dict,
            variant_names,
        )

        # Check stats
        assert stats["total_runs"] == 1
        assert stats["skipped"] == 0
        assert stats["errors"] == 0

        # Check that plot was called
        assert len(mock_analysis.plot_calls) == 1

    def test_run_with_empty_analysis_type(self, tmp_path):
        """Test running with empty analysis type"""
        conn = MockConnection([])
        outdir = tmp_path / "output"
        outdir.mkdir()

        config = {
            "outdir": str(outdir),
            "analysis_types": ["multiseed"],
            "multiseed": {},  # Empty
        }

        stats = run_analysis_loop(
            config,
            conn,
            "SELECT * FROM history",
            "SELECT * FROM config",
            "SELECT * FROM success",
            "",
            {},
            {},
            {},
        )

        assert stats["skipped"] == 1
        assert stats["total_runs"] == 0

    def test_run_with_missing_analysis_type(self, tmp_path):
        """Test that missing analysis type raises KeyError"""
        conn = MockConnection([])
        outdir = tmp_path / "output"
        outdir.mkdir()

        config = {
            "outdir": str(outdir),
            "analysis_types": ["multiseed"],
            # multiseed not in config
        }

        with pytest.raises(KeyError, match="multiseed"):
            run_analysis_loop(
                config,
                conn,
                "SELECT * FROM history",
                "SELECT * FROM config",
                "SELECT * FROM success",
                "",
                {},
                {},
                {},
            )

    def test_run_with_error_handling(self, tmp_path, monkeypatch):
        """Test that errors are caught and counted"""

        def mock_import(module_name):
            raise ImportError(f"Mock error for {module_name}")

        monkeypatch.setattr(importlib, "import_module", mock_import)

        conn = MockConnection([("test_exp", 0)])
        outdir = tmp_path / "output"
        outdir.mkdir()

        config = {
            "outdir": str(outdir),
            "analysis_types": ["multiseed"],
            "multiseed": {"test_analysis": {}},
        }

        stats = run_analysis_loop(
            config,
            conn,
            "SELECT * FROM history",
            "SELECT * FROM config",
            "SELECT * FROM success",
            "",
            {},
            {},
            {},
        )

        assert stats["errors"] == 1
        assert stats["total_runs"] == 0

    def test_run_auto_detects_analysis_types(self, tmp_path, monkeypatch):
        """Test that analysis types are auto-detected when not specified"""
        mock_analysis = ModuleType("mock_analysis")
        mock_analysis.plot_calls = []

        def mock_plot(*args, **kwargs):
            mock_analysis.plot_calls.append({"args": args, "kwargs": kwargs})

        mock_analysis.plot = mock_plot

        def mock_import(module_name):
            return mock_analysis

        monkeypatch.setattr(importlib, "import_module", mock_import)

        conn = MockConnection([("test_exp", 0)])
        outdir = tmp_path / "output"
        outdir.mkdir()

        # Don't specify analysis_types, it should detect from config
        config = {
            "outdir": str(outdir),
            "multiseed": {"test_analysis": {}},
            "multivariant": {"another_analysis": {}},
        }

        stats = run_analysis_loop(
            config,
            conn,
            "SELECT * FROM history",
            "SELECT * FROM config",
            "SELECT * FROM success",
            "",
            {},
            {},
            {},
        )

        # Should have run both analysis types
        assert stats["total_runs"] == 2
        assert "analysis_types" in config
        assert "multiseed" in config["analysis_types"]
        assert "multivariant" in config["analysis_types"]

    def test_run_filters_variants_correctly(self, tmp_path, monkeypatch):
        """Test that variants are filtered correctly for each analysis"""
        mock_analysis = ModuleType("mock_analysis")
        mock_analysis.plot_calls = []

        def mock_plot(
            plot_config,
            conn,
            history_q,
            config_q,
            success_q,
            sim_data_dict,
            validation_data_path,
            outdir,
            variant_metadata,
            variant_names,
        ):
            # Record what was passed
            mock_analysis.plot_calls.append(
                {
                    "sim_data_dict": sim_data_dict,
                    "variant_metadata": variant_metadata,
                }
            )

        mock_analysis.plot = mock_plot

        def mock_import(module_name):
            return mock_analysis

        monkeypatch.setattr(importlib, "import_module", mock_import)

        conn = MockConnection([("test_exp", 0), ("test_exp", 1)])
        outdir = tmp_path / "output"
        outdir.mkdir()

        config = {
            "outdir": str(outdir),
            "analysis_types": ["multiseed"],
            "multiseed": {"test_analysis": {}},
        }

        # Full metadata has 3 variants, but query will only return 2
        variant_metadata = {
            "test_exp": {
                0: {"param": "value0"},
                1: {"param": "value1"},
                2: {"param": "value2"},  # Should be filtered out
            }
        }
        sim_data_dict = {
            "test_exp": {
                0: "/path/0.cPickle",
                1: "/path/1.cPickle",
                2: "/path/2.cPickle",
            }
        }
        variant_names = {"test_exp": "test_variant"}

        stats = run_analysis_loop(
            config,
            conn,
            "SELECT * FROM history",
            "SELECT * FROM config",
            "SELECT * FROM success",
            "",
            variant_metadata,
            sim_data_dict,
            variant_names,
        )

        assert stats["total_runs"] == 2  # One for variant 0, one for variant 1

        # Check that filtered data was passed to plot
        for call in mock_analysis.plot_calls:
            # Should only have variants 0 or 1, not 2
            assert 2 not in call["variant_metadata"].get("test_exp", {})
            assert 2 not in call["sim_data_dict"].get("test_exp", {})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
