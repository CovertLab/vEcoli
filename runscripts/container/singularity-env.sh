#!/bin/bash
# Shared environment variables for Singularity build and runtime

# Silence warning about cache and sync targets being on different filesystems
export UV_LINK_MODE="copy"

# Prevent uv dirs from being created in any locations that may overlap with
# host filesystems that Apptainer may automount
export UV_TOOL_DIR="/vEcoli/.uv_tools"
export UV_TOOL_BIN_DIR="/vEcoli/.uv_tools/bin"
export UV_PYTHON_INSTALL_DIR="/vEcoli/.uv_python"
export UV_PYTHON_BIN_DIR="/vEcoli/.uv_python/bin"

# Cache directory
export UV_CACHE_DIR="/vEcoli/.uv_cache"

# Limit concurrency to try to fix sporadic build issues on HPC clusters
export UV_CONCURRENT_BUILDS=1
export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_INSTALLS=1
# Try to disable cache to fix sporadic build issues on HPC clusters
export UV_NO_CACHE=1
