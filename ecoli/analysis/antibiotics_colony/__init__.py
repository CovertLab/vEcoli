import pandas as pd
from scipy.constants import N_A

DE_GENES = pd.read_csv('data/marA_binding/model_degenes.csv')
SPLIT_TIME = 11550
MAX_TIME = 26000
COUNTS_PER_FL_TO_NANOMOLAR = 1 / (1e-15) / N_A * (1e9)

def restrict_data(data: pd.DataFrame):
    """If there is more than one condition in data, keep up
    to SPLIT_TIME from the first condition and between SPLIT_TIME
    and MAX_TIME from the the rest."""
    conditions = data.loc[:, 'Condition'].unique()
    if len(conditions) > 1:
        data = data.set_index(['Condition'])
        condition_1_mask = ((data.loc[conditions[0]]['Time'] 
            <= SPLIT_TIME))
        filtered_data = [data.loc[conditions[0]].loc[
            condition_1_mask, :]]
        for exp_condition in conditions[1:]:
            condition_mask = ((data.loc[exp_condition]['Time'] 
                >= SPLIT_TIME) & (data.loc[exp_condition]['Time'] 
                <= MAX_TIME))
            filtered_data.append(data.loc[exp_condition].loc[
                condition_mask, :])
        data = pd.concat(filtered_data)
        data = data.reset_index()
    else:
        data = data.loc[data.loc[:, 'Time'] <= MAX_TIME, :]
    return data
