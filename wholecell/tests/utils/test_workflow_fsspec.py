import os
import sys
import tempfile
import pytest
from unittest.mock import MagicMock, patch

# Import the functions to test
from runscripts.workflow import parse_uri, copy_to_filesystem


# Helper to simulate fsspec not being available
def simulate_no_fsspec(monkeypatch):
    """Patch to simulate fsspec not being installed"""
    monkeypatch.setattr("runscripts.workflow.FSSPEC_AVAILABLE", False)
    # Also hide any actual fsspec import that might be available
    if "fsspec" in sys.modules:
        monkeypatch.setitem(sys.modules, "fsspec", None)


class TestParseUri:
    def test_local_path_with_fsspec(self):
        """Test parsing a local path when fsspec is available"""
        local_path = "/tmp/test_dir"

        with patch("runscripts.workflow.FSSPEC_AVAILABLE", True):
            with patch("runscripts.workflow.url_to_fs") as mock_url_to_fs:
                # Mock return value for the fsspec filesystem and path
                mock_fs = MagicMock()
                mock_url_to_fs.return_value = (mock_fs, local_path)

                # Call the function
                fs, path = parse_uri(local_path)

                # Verify the function called url_to_fs
                mock_url_to_fs.assert_called_once()
                assert fs is mock_fs
                assert path == local_path

    def test_local_path_without_fsspec(self, monkeypatch):
        """Test parsing a local path when fsspec is not available"""
        local_path = "/tmp/test_dir"
        simulate_no_fsspec(monkeypatch)

        # Call the function
        fs, path = parse_uri(local_path)

        # Check results
        assert fs is None
        assert path == os.path.abspath(local_path)

    def test_cloud_uri_with_fsspec(self):
        """Test parsing a cloud URI when fsspec is available"""
        cloud_uri = "gs://my-bucket/my-folder"

        with patch("runscripts.workflow.FSSPEC_AVAILABLE", True):
            with patch("runscripts.workflow.url_to_fs") as mock_url_to_fs:
                # Mock return value for the fsspec filesystem and path
                mock_fs = MagicMock()
                mock_url_to_fs.return_value = (mock_fs, "my-bucket/my-folder")

                # Call the function
                fs, path = parse_uri(cloud_uri)

                # Verify the function called url_to_fs
                mock_url_to_fs.assert_called_once_with(cloud_uri)
                assert fs is mock_fs
                assert path == "my-bucket/my-folder"

    def test_cloud_uri_without_fsspec(self, monkeypatch):
        """Test parsing a cloud URI when fsspec is not available"""
        cloud_uri = "gs://my-bucket/my-folder"
        simulate_no_fsspec(monkeypatch)

        # This should just return None and the absolute path
        with pytest.raises(RuntimeError, match="fsspec is not available"):
            fs, path = parse_uri(cloud_uri)


class TestCopyToFilesystem:
    def test_copy_local_filesystem_none(self):
        """Test copying with filesystem=None (local copy)"""
        # Create temporary source and destination directories
        with tempfile.TemporaryDirectory() as source_dir:
            with tempfile.TemporaryDirectory() as dest_dir:
                # Create a test file
                source_file = os.path.join(source_dir, "test_file.txt")
                with open(source_file, "w") as f:
                    f.write("Test content")

                # Define destination path
                dest_file = os.path.join(dest_dir, "subfolder", "test_file.txt")

                # Copy the file
                copy_to_filesystem(source_file, dest_file, filesystem=None)

                # Verify the file was copied correctly
                assert os.path.exists(dest_file)
                with open(dest_file, "r") as f:
                    assert f.read() == "Test content"

                # Verify the subfolder was created
                assert os.path.exists(os.path.join(dest_dir, "subfolder"))

    def test_copy_with_fsspec_filesystem(self):
        """Test copying with a mock fsspec filesystem"""
        # Create a temporary source file
        with tempfile.NamedTemporaryFile(delete=False) as source_file:
            source_file.write(b"Test content")
            source_path = source_file.name

        try:
            # Create a mock filesystem
            mock_fs = MagicMock()
            mock_file = MagicMock()
            mock_fs.open.return_value.__enter__.return_value = mock_file

            # Define destination path
            dest_path = "gs://bucket/path/to/file.txt"

            # Copy the file
            copy_to_filesystem(source_path, dest_path, filesystem=mock_fs)

            # Verify the filesystem's open method was called correctly
            mock_fs.open.assert_called_once_with(dest_path, mode="wb")

            # Verify the write method was called (with any content)
            mock_file.write.assert_called_once()
        finally:
            # Clean up the temporary file
            os.unlink(source_path)

    @pytest.mark.parametrize("fsspec_available", [True, False])
    def test_copy_integration(self, monkeypatch, fsspec_available):
        """Integration test for both parse_uri and copy_to_filesystem"""
        # Set up fsspec availability based on parameter
        if not fsspec_available:
            simulate_no_fsspec(monkeypatch)

        # Create temporary directories
        with tempfile.TemporaryDirectory() as source_dir:
            with tempfile.TemporaryDirectory() as dest_dir:
                # Create a test file
                source_file = os.path.join(source_dir, "test_file.txt")
                with open(source_file, "w") as f:
                    f.write("Test content")

                # Get filesystem and path using parse_uri
                if fsspec_available:
                    # Mock url_to_fs for the local filesystem case
                    with patch("runscripts.workflow.url_to_fs") as mock_url_to_fs:
                        local_fs = MagicMock()
                        mock_url_to_fs.return_value = (local_fs, dest_dir)

                        fs, path = parse_uri(dest_dir)

                        # Mock filesystem.open to actually write to the local file
                        def mock_open(dest, mode="rb"):
                            os.makedirs(os.path.dirname(dest), exist_ok=True)
                            return open(dest, mode)

                        local_fs.open.side_effect = mock_open

                        # Copy the file
                        dest_file = os.path.join(path, "test_copy.txt")
                        copy_to_filesystem(source_file, dest_file, fs)
                else:
                    # Without fsspec, it should use the local filesystem
                    fs, path = parse_uri(dest_dir)
                    dest_file = os.path.join(path, "test_copy.txt")
                    copy_to_filesystem(source_file, dest_file, fs)

                # Verify the file was copied
                assert os.path.exists(dest_file)
                with open(dest_file, "r") as f:
                    assert f.read() == "Test content"
