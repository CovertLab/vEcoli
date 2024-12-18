import os

# Improve performance and reproducibility
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

from ecoli.library.sim_data import LoadSimData, SIM_DATA_PATH, SIM_DATA_PATH_NO_OPERONS

LOAD_SIM_DATA = LoadSimData(sim_data_path=SIM_DATA_PATH, seed=0)
LOAD_SIM_DATA_NO_OPERONS = LoadSimData(sim_data_path=SIM_DATA_PATH_NO_OPERONS, seed=0)
