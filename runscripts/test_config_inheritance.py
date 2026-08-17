"""
Unit tests for configuration inheritance helper functions in workflow.py
"""

import json
import tempfile
import pytest
from pathlib import Path

from runscripts.workflow import (
    load_config_with_inheritance,
    _merge_configs,
)


class TestMergeConfigs:
    """Test cases for the _merge_configs helper function"""

    def test_simple_key_override(self):
        """Test that overlay values override base values for simple keys"""
        base = {"key1": "value1", "key2": "value2"}
        overlay = {"key2": "new_value2", "key3": "value3"}

        _merge_configs(base, overlay)

        assert base == {"key1": "value1", "key2": "new_value2", "key3": "value3"}

    def test_nested_dict_merge(self):
        """Test recursive merging of nested dictionaries"""
        base = {"nested": {"level1": {"key1": "value1", "key2": "value2"}}}
        overlay = {"nested": {"level1": {"key2": "new_value2", "key3": "value3"}}}

        _merge_configs(base, overlay)

        assert base == {
            "nested": {
                "level1": {"key1": "value1", "key2": "new_value2", "key3": "value3"}
            }
        }

    def test_list_keys_merge(self):
        """Test that LIST_KEYS_TO_MERGE are concatenated and deduplicated"""
        base = {"save_times": [1, 2, 3], "other_key": "value"}
        overlay = {"save_times": [3, 4, 5]}

        _merge_configs(base, overlay)

        # Should concatenate, deduplicate, and sort
        assert base["save_times"] == [1, 2, 3, 4, 5]
        assert base["other_key"] == "value"

    def test_engine_process_reports_tuple_conversion(self):
        """Test that engine_process_reports items are converted to tuples"""
        base = {"engine_process_reports": [["path", "to", "process1"]]}
        overlay = {"engine_process_reports": [["path", "to", "process2"]]}

        _merge_configs(base, overlay)

        # Should convert lists to tuples
        assert all(isinstance(item, tuple) for item in base["engine_process_reports"])
        assert len(base["engine_process_reports"]) == 2

    def test_list_key_deduplication(self):
        """Test that duplicate items in list keys are removed"""
        base = {"processes": ["proc1", "proc2"]}
        overlay = {"processes": ["proc2", "proc3", "proc1"]}

        _merge_configs(base, overlay)

        # Should deduplicate and sort
        assert base["processes"] == ["proc1", "proc2", "proc3"]

    def test_dict_overwrite_non_dict(self):
        """Test that dict values overwrite non-dict values"""
        base = {"key": "string_value"}
        overlay = {"key": {"nested": "dict_value"}}

        _merge_configs(base, overlay)

        assert base["key"] == {"nested": "dict_value"}

    def test_non_dict_overwrite_dict(self):
        """Test that non-dict values overwrite dict values"""
        base = {"key": {"nested": "dict_value"}}
        overlay = {"key": "string_value"}

        _merge_configs(base, overlay)

        assert base["key"] == "string_value"

    def test_empty_base_config(self):
        """Test merging into empty base config"""
        base = {}
        overlay = {"key1": "value1", "key2": [1, 2, 3]}

        _merge_configs(base, overlay)

        assert base == {"key1": "value1", "key2": [1, 2, 3]}

    def test_empty_overlay_config(self):
        """Test merging empty overlay config"""
        base = {"key1": "value1"}
        overlay = {}

        _merge_configs(base, overlay)

        assert base == {"key1": "value1"}

    def test_multiple_list_keys(self):
        """Test merging multiple LIST_KEYS_TO_MERGE simultaneously"""
        base = {"save_times": [1, 2], "processes": ["proc1"]}
        overlay = {
            "save_times": [2, 3],
            "processes": ["proc2"],
            "add_processes": ["new_proc"],
        }

        _merge_configs(base, overlay)

        assert base["save_times"] == [1, 2, 3]
        assert base["processes"] == ["proc1", "proc2"]
        assert base["add_processes"] == ["new_proc"]


class TestLoadConfigWithInheritance:
    """Test cases for load_config_with_inheritance function"""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for config files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def write_config(self, path: Path, config: dict):
        """Helper to write a config file"""
        with open(path, "w") as f:
            json.dump(config, f)

    def test_no_inheritance(self, temp_config_dir):
        """Test loading a config with no inheritance"""
        config_path = temp_config_dir / "config.json"
        config = {"key1": "value1", "key2": "value2"}
        self.write_config(config_path, config)

        result = load_config_with_inheritance(str(config_path))

        assert result == config

    def test_single_level_inheritance(self, temp_config_dir, monkeypatch):
        """Test loading a config that inherits from one other config"""
        # Patch CONFIG_DIR_PATH to use our temp directory
        monkeypatch.setattr("runscripts.workflow.CONFIG_DIR_PATH", str(temp_config_dir))

        # Create base config
        base_config = {"key1": "base_value", "key2": "base_value2"}
        self.write_config(temp_config_dir / "base.json", base_config)

        # Create config that inherits from base
        child_config = {
            "inherit_from": ["base.json"],
            "key2": "child_value2",
            "key3": "child_value3",
        }
        self.write_config(temp_config_dir / "child.json", child_config)

        result = load_config_with_inheritance(str(temp_config_dir / "child.json"))

        # Child should override key2, inherit key1, and add key3
        assert result == {
            "inherit_from": ["base.json"],
            "key1": "base_value",
            "key2": "child_value2",
            "key3": "child_value3",
        }
        # inherit_from should still appear in result (for record-keeping)
        assert "inherit_from" in result

    def test_multi_level_inheritance(self, temp_config_dir, monkeypatch):
        """Test A inherits from B which inherits from C"""
        monkeypatch.setattr("runscripts.workflow.CONFIG_DIR_PATH", str(temp_config_dir))

        # C (base)
        c_config = {"key1": "c_value", "key2": "c_value"}
        self.write_config(temp_config_dir / "c.json", c_config)

        # B inherits from C
        b_config = {"inherit_from": ["c.json"], "key2": "b_value"}
        self.write_config(temp_config_dir / "b.json", b_config)

        # A inherits from B
        a_config = {"inherit_from": ["b.json"], "key1": "a_value"}
        self.write_config(temp_config_dir / "a.json", a_config)

        result = load_config_with_inheritance(str(temp_config_dir / "a.json"))

        # Priority: A > B > C
        assert result == {
            "inherit_from": ["b.json"],
            "key1": "a_value",  # From A (highest priority)
            "key2": "b_value",  # From B (overrode C)
        }

    def test_multiple_inheritance_priority(self, temp_config_dir, monkeypatch):
        """Test A inherits from [B, D] with correct priority: A > B > D"""
        monkeypatch.setattr("runscripts.workflow.CONFIG_DIR_PATH", str(temp_config_dir))

        # D
        d_config = {"key1": "d_value", "key2": "d_value", "key3": "d_value"}
        self.write_config(temp_config_dir / "d.json", d_config)

        # B
        b_config = {"key1": "b_value", "key2": "b_value"}
        self.write_config(temp_config_dir / "b.json", b_config)

        # A inherits from [B, D] - B should have higher priority than D
        a_config = {"inherit_from": ["b.json", "d.json"], "key1": "a_value"}
        self.write_config(temp_config_dir / "a.json", a_config)

        result = load_config_with_inheritance(str(temp_config_dir / "a.json"))

        # Priority: A > B > D
        assert result == {
            "inherit_from": ["b.json", "d.json"],
            "key1": "a_value",  # From A (highest)
            "key2": "b_value",  # From B
            "key3": "d_value",  # From D (lowest)
        }

    def test_complex_inheritance_tree(self, temp_config_dir, monkeypatch):
        """Test A inherits from [B, D] where B inherits from C"""
        monkeypatch.setattr("runscripts.workflow.CONFIG_DIR_PATH", str(temp_config_dir))

        # C
        c_config = {"key1": "c", "key2": "c", "key3": "c"}
        self.write_config(temp_config_dir / "c.json", c_config)

        # B inherits from C
        b_config = {"inherit_from": ["c.json"], "key2": "b"}
        self.write_config(temp_config_dir / "b.json", b_config)

        # D
        d_config = {"key1": "d", "key2": "d", "key3": "d", "key4": "d"}
        self.write_config(temp_config_dir / "d.json", d_config)

        # A inherits from [B, D]
        a_config = {"inherit_from": ["b.json", "d.json"], "key1": "a"}
        self.write_config(temp_config_dir / "a.json", a_config)

        result = load_config_with_inheritance(str(temp_config_dir / "a.json"))

        # Priority: A > B > C > D
        assert result == {
            "inherit_from": ["b.json", "d.json"],
            "key1": "a",  # From A (highest)
            "key2": "b",  # From B
            "key3": "c",  # From C
            "key4": "d",  # From D (lowest)
        }

    def test_list_merge_through_inheritance(self, temp_config_dir, monkeypatch):
        """Test that LIST_KEYS_TO_MERGE accumulate through inheritance chain"""
        monkeypatch.setattr("runscripts.workflow.CONFIG_DIR_PATH", str(temp_config_dir))

        # Base config
        base_config = {"save_times": [1, 2, 3], "processes": ["proc1"]}
        self.write_config(temp_config_dir / "base.json", base_config)

        # Child config
        child_config = {
            "inherit_from": ["base.json"],
            "save_times": [3, 4, 5],
            "processes": ["proc2"],
        }
        self.write_config(temp_config_dir / "child.json", child_config)

        result = load_config_with_inheritance(str(temp_config_dir / "child.json"))

        # Lists should be merged, deduplicated, and sorted
        assert result["save_times"] == [1, 2, 3, 4, 5]
        assert result["processes"] == ["proc1", "proc2"]

    def test_nested_dict_merge_through_inheritance(self, temp_config_dir, monkeypatch):
        """Test nested dictionary merging through inheritance"""
        monkeypatch.setattr("runscripts.workflow.CONFIG_DIR_PATH", str(temp_config_dir))

        # Base config with nested dict
        base_config = {
            "emitter_arg": {"out_dir": "/base/path", "setting1": "base_value"}
        }
        self.write_config(temp_config_dir / "base.json", base_config)

        # Child overrides part of nested dict
        child_config = {
            "inherit_from": ["base.json"],
            "emitter_arg": {"out_dir": "/child/path", "setting2": "child_value"},
        }
        self.write_config(temp_config_dir / "child.json", child_config)

        result = load_config_with_inheritance(str(temp_config_dir / "child.json"))

        # Nested dicts should be merged
        assert result["emitter_arg"] == {
            "out_dir": "/child/path",  # Child overrides
            "setting1": "base_value",  # Inherited from base
            "setting2": "child_value",  # Added by child
        }

    def test_diamond_inheritance_pattern(self, temp_config_dir, monkeypatch):
        """Test diamond pattern: A inherits from [B, C], both B and C inherit from D"""
        monkeypatch.setattr("runscripts.workflow.CONFIG_DIR_PATH", str(temp_config_dir))

        # D (base)
        d_config = {"key1": "d", "key_d": "d"}
        self.write_config(temp_config_dir / "d.json", d_config)

        # B inherits from D
        b_config = {"inherit_from": ["d.json"], "key1": "b", "key_b": "b"}
        self.write_config(temp_config_dir / "b.json", b_config)

        # C inherits from D
        c_config = {"inherit_from": ["d.json"], "key1": "c", "key_c": "c"}
        self.write_config(temp_config_dir / "c.json", c_config)

        # A inherits from [B, C]
        a_config = {"inherit_from": ["b.json", "c.json"], "key1": "a"}
        self.write_config(temp_config_dir / "a.json", a_config)

        result = load_config_with_inheritance(str(temp_config_dir / "a.json"))

        # Priority: A > B > ... > C > D
        assert result["key1"] == "a"  # A overrides all
        assert result["key_b"] == "b"  # From B
        assert result["key_c"] == "c"  # From C
        assert result["key_d"] == "d"  # From D

    def test_empty_inherit_from_list(self, temp_config_dir, monkeypatch):
        """Test config with empty inherit_from list"""
        monkeypatch.setattr("runscripts.workflow.CONFIG_DIR_PATH", str(temp_config_dir))

        config = {"inherit_from": [], "key1": "value1"}
        self.write_config(temp_config_dir / "config.json", config)

        result = load_config_with_inheritance(str(temp_config_dir / "config.json"))

        assert result == {"key1": "value1", "inherit_from": []}
        assert "inherit_from" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
