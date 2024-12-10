import numpy as np
import pandas as pd
import pickle
import os
from typing import Any, cast
from duckdb import DuckDBPyConnection

from ecoli.library.parquet_emitter import open_arbitrary_sim_data

## temp imports

import json
##


def plot(
    params: dict[str, Any],
    conn: DuckDBPyConnection,
    history_sql: str,
    config_sql: str,
    sim_data_dict: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_names: dict[str, str],
):


    exp_id = list(sim_data_dict.keys())[0]

    simData_all = list(sim_data_dict[exp_id].values())

    kcat_output = pd.DataFrame()


    for sim_data_path in simData_all:

        var_id = sim_data_path.split('.')[0].split('simData')[1]


        sim_data_dict_new = {}

        sim_data_var_new = {}

        sim_data_var_new['1'] = sim_data_path
        sim_data_dict_new[exp_id] = sim_data_var_new

        with open_arbitrary_sim_data(sim_data_dict_new) as f:
            sim_data: "SimulationDataEcoli" = pickle.load(f)


        kcats_var = sim_data.process.metabolism._kcats

        kcats_var = kcats_var[:,0]

        kcat_dict = {}

        for kcat_idx in range(len(kcats_var)):
            kcat_dict[f"kcat_{kcat_idx}"] = [kcats_var[kcat_idx]]


        query = f'''
        SELECT listeners__fba_results__external_exchange_fluxes,listeners__mass__instantaneous_growth_rate,time FROM ({history_sql}) WHERE lineage_seed=0 AND variant={int(var_id)}
        ORDER BY time
        '''

        output_var = conn.sql(query).df()
        # output_var.to_csv(os.path.join(outdir, f"results_var_{var_id}.tsv"),sep="\t")
        output_fluxes = output_var.iloc[-1,:]
        output_fluxes = output_fluxes['listeners__fba_results__external_exchange_fluxes']
        output_fluxes_dict = {}
        for rxn_idx in range(len(output_fluxes)):
            output_fluxes_dict[f"exc_rxn_{str(rxn_idx)}"] = [output_fluxes[rxn_idx]]
        kcat_dict = {**kcat_dict,**output_fluxes_dict}

        output_growth_rate = output_var.iloc[-1,:]
        output_growth_rate = output_growth_rate['listeners__mass__instantaneous_growth_rate']
        kcat_dict['growth_rate'] = [output_growth_rate]

        kcat_df = pd.DataFrame.from_dict(kcat_dict)



        kcat_output = pd.concat([kcat_output,kcat_df],axis=0)




    kcat_output.to_csv(os.path.join(outdir, f"kcat_samples.tsv"), sep='\t', index=False)





#%%

