import os

# Set BLAS threading environment variables before any imports
# This ensures single-threaded BLAS for better performance and reproducibility
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
