import os
import xmltodict
import pandas as pd
from notebooks.marimo.utils.biocyc_webservice import biocyc_credentials

wd_root = os.getcwd().split("/notebooks")[0]


dir_credentials = os.path.join(wd_root, "notebooks", "marimo", "credentials")

wd_out = os.path.join(wd_root, "notebooks", "marimo", "pathways")


biocyc_session = biocyc_credentials(dir_credentials)
url_pathways = "https://websvc.biocyc.org/apixml?fn=get-class-all-instances&id=ECOLI:Pathways&detail=none"
r_pathways = biocyc_session.get(url_pathways)

pathways_dict = xmltodict.parse(r_pathways.text)["ptools-xml"]["Pathway"]

pathways_frameids = [pwy["@frameid"] for pwy in pathways_dict]

# %%
# pwy_id = ""
pwy_rxns = {}

for pwy_id in pathways_frameids:
    rxns_url = f"https://websvc.biocyc.org/getxml?ECOLI:{pwy_id}"
    pwy_dict = xmltodict.parse(biocyc_session.get(rxns_url).text)["ptools-xml"][
        "Pathway"
    ]
    pwy_rxn = {}
    pwy_rxn["id"] = pwy_id
    pwy_rxn["name"] = pwy_dict["common-name"]["#text"]
    if "Reaction" in list(pwy_dict["reaction-list"].keys()):
        rxns_list = pwy_dict["reaction-list"]["Reaction"]
        if isinstance(rxns_list, list):
            pwy_rxn["rxns"] = [
                rxn["@frameid"] for rxn in pwy_dict["reaction-list"]["Reaction"]
            ]
        elif isinstance(rxns_list, dict):
            pwy_rxn["rxns"] = [rxns_list["@frameid"]]

        pwy_rxns[pwy_id] = pwy_rxn

# %% genes of rxn  ['EG10181 // EG10178 // EG10179 // EG10180', 'EG10801']}}

for pwy_id in list(pwy_rxns.keys()):
    pwy_entry = pwy_rxns[pwy_id]
    pwy_entry_rxns = pwy_entry["rxns"]
    pwy_entry_genes = []
    pwy_entry_compounds = []
    for rxn_id in pwy_entry_rxns:
        compounds_rxn = []
        genes_url = (
            f"https://websvc.biocyc.org/apixml?fn=genes-of-reaction&id=ECOLI:{rxn_id}"
        )

        genes_r = xmltodict.parse(biocyc_session.get(genes_url).text)["ptools-xml"]

        if "Gene" in list(genes_r.keys()):
            genes_list = xmltodict.parse(biocyc_session.get(genes_url).text)[
                "ptools-xml"
            ]["Gene"]

            if isinstance(genes_list, list):
                rxn_genes = [gene["@frameid"] for gene in genes_list]
                pwy_rxn_genes = " // ".join(rxn_genes)

                for gene in rxn_genes:
                    compound_url = f"https://websvc.biocyc.org/apixml?fn=all-products-of-gene&id=ECOLI:{gene}"
                    compound_list = xmltodict.parse(
                        biocyc_session.get(compound_url).text
                    )["ptools-xml"]["Protein"]
                    if isinstance(compound_list, list):
                        compounds_gene = " // ".join(
                            [compound["@frameid"] for compound in compound_list]
                        )
                    elif isinstance(compound_list, dict):
                        compounds_gene = compound_list["@frameid"]
                    compounds_rxn.append(compounds_gene)

            elif isinstance(genes_list, dict):
                pwy_rxn_genes = genes_list["@frameid"]
                compound_url = f"https://websvc.biocyc.org/apixml?fn=all-products-of-gene&id=ECOLI:{genes_list['@frameid']}"
                compound_list = xmltodict.parse(biocyc_session.get(compound_url).text)[
                    "ptools-xml"
                ]["Protein"]
                if isinstance(compound_list, list):
                    compounds_gene = " // ".join(
                        [compound["@frameid"] for compound in compound_list]
                    )
                elif isinstance(compound_list, dict):
                    compounds_gene = compound_list["@frameid"]
                compounds_rxn.append(compounds_gene)
        else:
            pwy_rxn_genes = ""
            compounds_rxn = [""]

        pwy_entry_genes.append(pwy_rxn_genes)
        pwy_entry_compounds.append(compounds_rxn)
        pwy_entry["genes"] = pwy_entry_genes
        pwy_entry["compounds"] = pwy_entry_compounds
    pwy_rxns[pwy_id] = pwy_entry

# %%
pwy_db_df = pd.DataFrame()

for pwy_id in pwy_rxns.keys():
    pwy_df_dict = {}
    pwy_entry = pwy_rxns[pwy_id]
    pwy_name = [pwy_rxns[pwy_id]["name"]] * len(pwy_entry["rxns"])
    pwy_row_rxns = pwy_entry["rxns"]
    pwy_genes = pwy_entry["genes"]
    pwy_compounds = [
        " // ".join(gene_compounds) for gene_compounds in pwy_entry["compounds"]
    ]

    pwy_df_dict["name"] = pwy_name
    pwy_df_dict["reactions"] = pwy_row_rxns
    pwy_df_dict["genes"] = pwy_genes
    pwy_df_dict["compounds"] = pwy_compounds

    pwy_df = pd.DataFrame(columns=list(pwy_df_dict.keys()))
    for col in pwy_df_dict.keys():
        pwy_df[col] = pwy_df_dict[col]

    pwy_db_df = pd.concat([pwy_db_df, pwy_df], ignore_index=True)


# %%
pwy_db_df.to_csv(os.path.join(wd_out, "pathways.txt"), sep="\t", index=False)
