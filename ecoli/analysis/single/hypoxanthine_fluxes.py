import os
from typing import Any, cast

from duckdb import DuckDBPyConnection
import polars as pl
import pickle
import matplotlib.pyplot as plt

from ecoli.library.parquet_emitter import (
    num_cells,
    read_stacked_columns,
    get_field_metadata,
    named_idx,
    open_arbitrary_sim_data,
    ndidx_to_duckdb_expr,
)
from wholecell.utils import units

def plot(
    params: dict[str, Any],
    conn: DuckDBPyConnection,
    history_sql: str,
    config_sql: str,
    sim_data_paths: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_name: str,
):
    assert (
        num_cells(conn, config_sql) == 1
    ), "Mass fraction summary plot requires single-cell data."

    with open_arbitrary_sim_data(sim_data_paths) as f:
        sim_data = pickle.load(f)


    import ipdb; ipdb.set_trace()
    hyp_rxn_ids = []
    for rxn in sim_data.process.metabolism.reaction_stoich:
        values = sim_data.process.metabolism.reaction_stoich[rxn].keys()
        if 'HYPOXANTHINE[c]' in values:
            hyp_rxn_ids.append(rxn)
        elif 'HYPOXANTHINE[p]' in values:
            hyp_rxn_ids.append(rxn)

    hyp_delta_metab_name = 'Hyp delta metab'
    hyp_ext_exc_name = 'Hyp external exchange'
    uracil_ext_exc_name = 'Uracil external exchange'
    cytosine_ext_exc_name = 'Cytosine external exchange'
    xanthine_ext_exc_name = 'Xanthine external exchange'
    adenine_ext_exc_name = 'Adenine external exchange'
    guanine_ext_exc_name = 'Guanine external exchange'
    ext_exc_names = [hyp_ext_exc_name, uracil_ext_exc_name, cytosine_ext_exc_name, xanthine_ext_exc_name, adenine_ext_exc_name, guanine_ext_exc_name]


    hypoxanthine_id = 'HYPOXANTHINE'
    uracil_id = 'URACIL'
    cytosine_id = 'CYTOSINE'
    xanthine_id = 'XANTHINE'
    adenine_id = 'ADENINE'
    guanine_id = 'GUANINE'
    nucl_ids = [hypoxanthine_id, uracil_id, cytosine_id, xanthine_id, adenine_id, guanine_id]

    # hyp_rxn_ids = ['ADENINE-DEAMINASE-RXN', 'DEOXYINOPHOSPHOR-RXN', 'HYPOXANPRIBOSYLTRAN-RXN (reverse)',
    #                'INOPHOSPHOR-RXN', 'INOSINATE-NUCLEOSIDASE-RXN', 'INOSINATE-NUCLEOSIDASE-RXN-IMP/WATER//HYPOXANTHINE/CPD-15318.34.',
    #                'INOSINATE-NUCLEOSIDASE-RXN-IMP/WATER//HYPOXANTHINE/CPD-16551.34.',
    #                'INOSINE-NUCLEOSIDASE-RXN', 'RXN-7682', 'RXN0-7206-HYPOXANTHINE//HYPOXANTHINE.27.', 'RXN0-7206-HYPOXANTHINE//HYPOXANTHINE.27. (reverse)',
    #                'TRANS-RXN0-562 (reverse)', 'TRANS-RXN0-579', 'TRANS-RXN0-611-HYPOXANTHINE//HYPOXANTHINE.27.',
    #                'TRANS-RXN0-611-HYPOXANTHINE//HYPOXANTHINE.27. (reverse)'
    #                ]

    hyp_base_rxn_ids = ['ADENINE-DEAMINASE-RXN', 'DEOXYINOPHOSPHOR-RXN', 'HYPOXANPRIBOSYLTRAN-RXN',
                   'INOPHOSPHOR-RXN', 'INOSINATE-NUCLEOSIDASE-RXN',
                   'INOSINE-NUCLEOSIDASE-RXN', 'RXN-7682', 'RXN0-7206',
                   'TRANS-RXN0-562', 'TRANS-RXN0-579', 'TRANS-RXN0-611',]
    # RXN-7682 (makes xanthine from hypoxanthine) and ADENINE-DEAMINASE-RXN (makes hypoxanthine from adenine) are the only non-zero ones.
    # Seems like RXN-7682 uses a lot of hypoxanthine, a lot more than adenine-deaminase makes. Where is all the hypoxanthine coming from then?
    # But delta metab is always around 0.
    # Cytosine is exchanged with outside

    metab_counts_dict = {
        mol[:-3]: i
        for i, mol in enumerate(
            get_field_metadata(conn, config_sql, "listeners__enzyme_kinetics__metabolite_counts_final")
        )
    }
    hypoxanthine_metab_idx = cast(int, metab_counts_dict[hypoxanthine_id])
    adenine_metab_idx = cast(int, metab_counts_dict['ADENINE'])

    external_exchange_dict = {
        mol[:-3]: i
        for i, mol in enumerate(
            get_field_metadata(conn, config_sql, "listeners__fba_results__external_exchange_fluxes")
        )
    }
    nucl_ext_exc_idxs = [cast(int, external_exchange_dict[mol]) for mol in nucl_ids]

    reaction_fluxes_dict = {
        rxn: i
        for i, rxn in enumerate(
            get_field_metadata(conn, config_sql, "listeners__fba_results__reaction_fluxes")
        )
    }
    hyp_rxn_idxs = [cast(int, reaction_fluxes_dict[rxn]) for rxn in hyp_rxn_ids]

    # conc_update_dict = {
    #     mol: i
    #     for i, mol in enumerate(
    #         get_field_metadata(conn, config_sql, "listeners__fba_results__conc_updates")
    #     )
    # }
    hom_obj_dict = {
        mol: i
        for i, mol in enumerate(
            get_field_metadata(conn, config_sql, "listeners__fba_results__homeostatic_objective_values")
        )
    }
    hyp_hom_idx = cast(int, hom_obj_dict[hypoxanthine_id+'[c]'])
    adenine_hom_idx = cast(int, hom_obj_dict['ADENINE[c]'])

    actual_flux_dict = {
        rxn: i
        for i, rxn in enumerate(
            get_field_metadata(conn, config_sql, "listeners__enzyme_kinetics__actual_fluxes")
        )
    }
    ade_rxn_idx = cast(int, actual_flux_dict[hyp_rxn_ids[0]])
    #in_kinetic = [x in actual_flux_dict for x in hyp_rxn_ids]

    base_rxn_dict = {
        rxn: i
        for i, rxn in enumerate(
            get_field_metadata(conn, config_sql, "listeners__fba_results__base_reaction_fluxes")
        )
    }
    hyp_base_rxn_idxs = [cast(int, base_rxn_dict[x]) for x in hyp_base_rxn_ids]

    hypoxanthine_delta_metab = named_idx(
        "listeners__fba_results__delta_metabolites", [hyp_delta_metab_name, 'adenine'], [hypoxanthine_metab_idx,
                                                                                         adenine_metab_idx]
    )

    kinetic_obj = named_idx(
        "listeners__fba_results__kinetic_objective_values", []
    )

    hypoxanthine_metab_final = named_idx(
        "listeners__enzyme_kinetics__metabolite_counts_final", ['hypoxanthine final', 'adenine final'], [hypoxanthine_metab_idx,
                                                                                                         adenine_metab_idx]
    )
    hypoxanthine_metab_init = named_idx(
        "listeners__enzyme_kinetics__metabolite_counts_init", ['hypoxanthine init', 'adenine init'], [hypoxanthine_metab_idx,
                                                                                      adenine_metab_idx]
    )
    nucls_ext_exc = named_idx(
        "listeners__fba_results__external_exchange_fluxes", ext_exc_names,
        nucl_ext_exc_idxs
    )
    hyp_rxn_fluxes = named_idx(
        "listeners__fba_results__reaction_fluxes", hyp_rxn_ids,
        hyp_rxn_idxs
    )
    hyp_hom_obj = named_idx(
        "listeners__fba_results__homeostatic_objective_values", ['hyp hom obj', 'adenine hom obj'],
        [hyp_hom_idx, adenine_hom_idx]
    )
    actual_flux_ade = named_idx(
        "listeners__enzyme_kinetics__actual_fluxes", ['ade_flux'],
        [ade_rxn_idx]
    )
    hyp_base_rxns = named_idx(
        "listeners__fba_results__base_reaction_fluxes", hyp_base_rxn_ids,
        hyp_base_rxn_idxs
    )

    # Extract data
    hyp_data = read_stacked_columns(
        history_sql,
        ['listeners__fba_results__delta_metabolites', 'listeners__fba_results__nucls_ext_exc',
         'listeners__fba_results__reaction_fluxes', 'listeners__fba_results__conc_updates',
         'listeners__enzyme_kinetics__metabolite_counts_init', 'listeners__enzyme_kinetics__metabolite_counts_final',
         'listeners__fba_results__homeostatic_objective_values', 'listeners__enzyme_kinetics__actual_fluxes',
         'listeners__fba_results__base_reaction_fluxes'],
        [hypoxanthine_delta_metab, nucls_ext_exc, hyp_rxn_fluxes, hypoxanthine_metab_init, hypoxanthine_metab_final,
         hyp_hom_obj, actual_flux_ade, hyp_base_rxns],
        conn=conn,
    )
    others = read_stacked_columns(
        history_sql,
        ['listeners__fba_results__coefficient', 'listeners__replication_data__fork_coordinates'],
        conn=conn
    )

    # TODO: look at the reaction fluxes. Then look at enzyme fluxes. See if could do smth w/ hypoxanthine concentration? :)
    # TODO: so rn, flux in and out is around 0.0058 in the beginning, and it rises exponentially (?) as cell doubles.
    #
    hyp_data = pl.DataFrame(hyp_data)
    other_data = pl.DataFrame(others)
    num_repl_forks = other_data['listeners__replication_data__fork_coordinates'].to_list()
    num_repl_forks = [len(y) for y in num_repl_forks]


    num_plots = len(hyp_rxn_ids) + len(nucl_ids) + 1
    fig, axs = plt.subplots(num_plots, figsize=(100, 5*num_plots))
    for i, rxn in enumerate(hyp_rxn_ids):
        axs[i].plot(hyp_data["time"], hyp_data[rxn])
        axs[i].set_title(rxn)

    for i, rxn in enumerate(ext_exc_names):
        axs[len(hyp_rxn_ids)+i].plot(hyp_data["time"], hyp_data[rxn])
        axs[len(hyp_rxn_ids)+i].set_title(rxn)

    axs[len(hyp_rxn_ids) + len(nucl_ids)].plot(hyp_data["time"], num_repl_forks)
    axs[len(hyp_rxn_ids) + len(nucl_ids)].set_title("Num replication forks")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "hypoxanthine_fluxes.pdf"))
