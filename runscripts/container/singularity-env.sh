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
export UV_CACHE_DIR="/vEcoli/.uv_cache"

# Pin BLAS/NumPy CPU code paths for cross-node reproducibility. Otherwise
# OpenBLAS (bundled in the numpy/scipy wheels) runtime-dispatches a different
# kernel per CPU model and NumPy's own SIMD reductions vary with AVX-512
# availability, both of which change floating-point rounding across nodes.
# x86-64 only: these names are invalid on aarch64 and would error, so there we
# let OpenBLAS/NumPy autodetect. Respects any value set by the caller.
if [ "$(uname -m)" = "x86_64" ]; then
    export OPENBLAS_CORETYPE="${OPENBLAS_CORETYPE:-Haswell}"
    export NPY_DISABLE_CPU_FEATURES="${NPY_DISABLE_CPU_FEATURES:-AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR}"
fi
