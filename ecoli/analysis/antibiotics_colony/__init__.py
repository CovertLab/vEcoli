import pandas as pd
from scipy.constants import N_A

DE_GENES = pd.read_csv('data/marA_binding/model_degenes.csv')
SPLIT_TIME = 11550
MAX_TIME = 26000
COUNTS_PER_FL_TO_NANOMOLAR = 1 / (1e-15) / N_A * (1e9)
