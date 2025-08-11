import os
import re
import multiprocessing

from cobra import Metabolite, Model, Reaction
from cobra.core.gene import Gene
from cobra.io import write_sbml_model
from parse import parse
from scipy.io import loadmat
from tqdm import tqdm

from validate_migration import validate_model_migration


MAT_MODELS = ["base_model/Rpom_0.mat", "base_model/Rpom_02.mat", "base_model/Rpom_03.mat", "base_model/Rpom_04.mat", "base_model/Rpom_05.mat",
              "base_model/Rpom_06.mat", "base_model/Rpom_025.mat", "base_model/Rpom_035.mat", "base_model/Rpom_045.mat", "base_model/Rpom_055.mat"]


def convert_model(model):
    # Load model
    mat_model = loadmat(model)

    # Extract the model name, discard unecessary .mat file metadata
    model_name = [k for k in mat_model.keys() if not k.startswith("__")][0]
    mat_model = mat_model[model_name]

    # Unpack .mat model
    rxn_ids = [str(x[0]) for x in mat_model["rxns"][0, 0].flatten()]
    rxn_names = [str(x[0]) for x in mat_model["rxnNames"][0, 0].flatten()]
    lbs = [float(x) for x in mat_model["lb"][0, 0].flatten()]
    ubs = [float(x) for x in mat_model["ub"][0, 0].flatten()]
    rxn_ec = [str(x[0]) if x.size ==
                1 else "" for x in mat_model["rxnECNumbers"][0, 0].flatten()]
    rxn_kegg = [
        str(x[0]) if x.size else "" for x in mat_model["rxnKeggIDs"][0, 0].flatten()]
    met_ids = [str(x[0]) for x in mat_model["mets"][0, 0].flatten()]
    met_formulas = [str(x[0])
                    for x in mat_model["metFormulas"][0, 0].flatten()]
    met_names = [str(x[0]) for x in mat_model["metNames"][0, 0].flatten()]
    met_charges = [int(x) for x in mat_model["metCharges"][0, 0].flatten()]
    met_kegg = [str(x[0]) if (
        x.size == 1) else None for x in mat_model["metKeggIDs"][0, 0].flatten()]
    S = mat_model["S"][0, 0]
    gene_names = [str(g[0])
                    for g in mat_model["geneNames"][0, 0].flatten()]
    rules = [str(x[0]) if x.size == 1 else ""
                for x in mat_model["rules"][0, 0].flatten()]
    rev = [int(x) for x in mat_model["rev"][0, 0].flatten()]

    # Create empty SBML model
    sbml_model = Model(model_name)

    # populate with reactions
    # (data from mat_model is deeply nested in useless dimensions...)
    reactions = [
        Reaction(rxn_id,
                    rxn_name,
                    lower_bound=lb,
                    upper_bound=ub)
        for rxn_id, rxn_name, lb, ub in zip(
            rxn_ids, rxn_names, lbs, ubs
        )]

    # Load metabolites
    metabolites = [
        Metabolite(
            met_id,
            formula=formula,
            name=name,
            compartment=parse(
                "{name}[{compartment}]",
                met_id)["compartment"]
        )
        for met_id, formula, name in
        zip(met_ids, met_formulas, met_names)
    ]

    # Add metabolites to reactions using stoichiometry
    for c in range(S.shape[1]):
        rxn_stoich = S[:, c]
        rxn_mets, _ = rxn_stoich.nonzero()

        reactions[c].add_metabolites(
            {
                metabolites[met]: rxn_stoich[met, 0]
                for met in rxn_mets
            }
        )

    # Add genes to model, associate gene rules to reactions
    for gene in gene_names:
        sbml_model.genes.append(Gene(gene, gene))

    for rule, reaction in zip(rules, reactions):
        if rule == "":
            continue

        # replace x(gene_index) with gene_names[gene_index],
        # boolean '|' operator with 'or'
        #
        # Not sure if there are other operators here (&? !?),
        # may need to build in more substitutions
        try:
            # gene index - 1 since matlab does 1-based indexing
            clean_rule = re.sub(
                "x\((?P<gene>\d+)\)", lambda g: gene_names[int(g["gene"]) - 1], rule)
            clean_rule = re.sub("\|", " or ", clean_rule)
        except IndexError:
            raise IndexError(rule)

        # Put rule into reaction
        reaction.gene_reaction_rule = clean_rule

    # Store charge data, Kegg IDs in metabolites
    for metabolite, charge, kegg in zip(metabolites, met_charges, met_kegg):
        metabolite.charge = charge
        if kegg is not None:
            metabolite.annotation["Kegg ID"] = kegg

    # Store EC numbers, Kegg IDs in reactions
    for rxn, ec, kegg in zip(reactions, rxn_ec, rxn_kegg):
        rxn.annotation["EC Number"] = ec
        if kegg is not None:
            rxn.annotation["Kegg ID"] = kegg

    # Load reactions into model
    sbml_model.add_reactions(reactions)

    # Store some metadata
    sbml_model.compartments["c"] = "cytosol"
    sbml_model.compartments["p"] = "periplasm"
    sbml_model.compartments["e"] = "external"

    # Set objective
    sbml_model.objective = sbml_model.reactions.get_by_id("BiomassRxn")

    return sbml_model, model_name


def main():

    with multiprocessing.Pool() as pool:
        results = pool.map(convert_model, MAT_MODELS)
    
    for sbml_model, model_name in results:
        # Save off model
        outfile = os.path.join("base_model/", f"{model_name}.xml")
        write_sbml_model(sbml_model, outfile)

        # Validate the migration, checking reversibility...
        print(f"\nValidating model {model_name} ==================================")
        validate_model_migration(outfile)


if __name__ == "__main__":
    main()
