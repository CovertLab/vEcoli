"""
ValidationData for Ecoli

Raw data processed into forms convienent for validation and analysis
"""

import numpy as np
from unum import Unum

# Raw data class
from reconstruction.ecoli.knowledge_base_raw import KnowledgeBaseEcoli

# Data classes
from reconstruction.ecoli.dataclasses.getter_functions import (
    GetterFunctions,
    UNDEFINED_COMPARTMENT_IDS_TO_ABBREVS,
)
from reconstruction.ecoli.dataclasses.molecule_groups import MoleculeGroups
from reconstruction.ecoli.dataclasses.molecule_ids import MoleculeIds
from reconstruction.ecoli.dataclasses.constants import Constants
from reconstruction.ecoli.dataclasses.state.internal_state import InternalState
from reconstruction.ecoli.dataclasses.process.process import Process
from reconstruction.ecoli.dataclasses.growth_rate_dependent_parameters import (
    Mass,
    GrowthRateParameters,
)
from reconstruction.ecoli.dataclasses.relation import Relation

__all__ = [
    "KnowledgeBaseEcoli",
    "GetterFunctions",
    "MoleculeGroups",
    "MoleculeIds",
    "Constants",
    "InternalState",
    "Process",
    "Mass",
    "GrowthRateParameters",
    "Relation",
    "ValidationDataEcoli",
    "Protein",
    "ReactionFlux",
    "EssentialGenes",
    "GeneFunctions",
]


class ValidationDataEcoli(object):
    """ValidationDataEcoli"""

    def __init__(self):
        pass

    def initialize(self, validation_data_raw, knowledge_base_raw):
        self.protein = Protein(validation_data_raw, knowledge_base_raw)
        self.reactionFlux = ReactionFlux(validation_data_raw, knowledge_base_raw)
        self.essential_genes = EssentialGenes(validation_data_raw)
        self.geneFunctions = GeneFunctions(validation_data_raw)
        self.macromolecular_growth_rate_modulation = MacromolecularGrowthRateModulation(
            validation_data_raw
        )

        self._add_amino_acid_growth_rates(validation_data_raw)

    def _add_amino_acid_growth_rates(self, validation_data_raw):
        """
        Loads growth rates with single amino acids supplemented in media.

        amino_acid_media_growth_rates: dict with data from 4 replicates
                {
                        media ID (str):
                        {
                                'mean': mean max growth rate from 4 replicates (float with units per time)
                                'std': standard deviation for max growth rate from 4 replicates (float with units per time)
                        }
                }
        amino_acid_media_dose_dependent_growth_rates: dict with data from single measurements from 4 concentrations
                {
                        media ID (str):
                        {
                                'conc': concentration of the amino acid in media (np.ndarray[float] with units mol/volume)
                                'growth': max growth rate corresponding to each media concentration (float with units per time)
                        }
                }
        """

        rates = {}
        for row in validation_data_raw.amino_acid_growth_rates:
            rates[row["Media"]] = {
                "mean": row["Average max growth rate"],
                "std": row["Max growth rate standard deviation"],
            }
        self.amino_acid_growth_rates = rates

        rates = {}
        for row in validation_data_raw.amino_acid_growth_rates_dose_response:
            rates[row["Media"]] = {
                "conc": row["Media concentrations"],
                "growth": row["Max growth rates"],
            }
        self.amino_acid_dose_dependent_growth_rates = rates


class Protein(object):
    """Protein"""

    def __init__(self, validation_data_raw, knowledge_base_raw):
        compartment_ids_to_abbreviations = {
            comp["id"]: comp["abbrev"] for comp in knowledge_base_raw.compartments
        }

        # Compartments that don't exist in compartments.tsv
        compartment_ids_to_abbreviations.update(UNDEFINED_COMPARTMENT_IDS_TO_ABBREVS)

        protein_id_to_compartment_tag = {}

        for protein in knowledge_base_raw.proteins:
            exp_compartment = protein["experimental_compartment"]
            comp_compartment = protein["computational_compartment"]

            if len(exp_compartment) + len(comp_compartment) == 0:
                compartment = "CCO-CYTOSOL"
            elif len(exp_compartment) > 0:
                compartment = exp_compartment[0]
            else:
                compartment = comp_compartment[0]

            protein_id_to_compartment_tag.update(
                {protein["id"]: [compartment_ids_to_abbreviations[compartment]]}
            )

        # Build and save a dict from gene ID to monomerId
        rna_id_to_gene_id = {
            gene["rna_ids"][0]: gene["id"] for gene in knowledge_base_raw.genes
        }

        self.geneIdToMonomerId = {}

        for rna in knowledge_base_raw.rnas:
            if (
                len(rna["monomer_ids"]) > 0
                and rna["monomer_ids"][0] in protein_id_to_compartment_tag
            ):
                self.geneIdToMonomerId[rna_id_to_gene_id[rna["id"]]] = (
                    f"{rna['monomer_ids'][0]}[{protein_id_to_compartment_tag[rna['monomer_ids'][0]][0]}]"
                )

        # Build and save a dict from gene symbol to corresponding monomerId
        self.geneSymbolToMonomerId = {}
        gene_id_to_symbol = {
            gene["id"]: gene["symbol"] for gene in knowledge_base_raw.genes
        }

        for gene_id, monomer_id in self.geneIdToMonomerId.items():
            self.geneSymbolToMonomerId[gene_id_to_symbol[gene_id]] = monomer_id

        self._loadTaniguchi2010Counts(validation_data_raw)
        self._loadHouser2015Counts(validation_data_raw)
        self._loadWisniewski2014Counts(validation_data_raw, knowledge_base_raw)
        self._loadSchmidt2015Counts(validation_data_raw)
        self._load_li(validation_data_raw)

    def _loadTaniguchi2010Counts(self, validation_data_raw):
        # Load taniguichi Xie Science 2010 dataset
        taniguichi_dataset = validation_data_raw.taniguichi2010_table_6
        self.taniguichi2010counts = np.zeros(
            len(taniguichi_dataset),
            dtype=[
                ("monomerId", "U100"),
                ("gene_symbol", "U10"),
                ("b_number", "U10"),
                ("counts_ave", np.float32),
                ("gamma_shape_parameter", np.float32),
                ("gamma_scale_parameter", np.float32),
            ],
        )
        for idx, row in enumerate(taniguichi_dataset):
            self.taniguichi2010counts[idx]["gene_symbol"] = row["Gene_Name"]
            self.taniguichi2010counts[idx]["b_number"] = row["B_Number"]
            self.taniguichi2010counts[idx]["counts_ave"] = row["Mean_Protein"]
            self.taniguichi2010counts[idx]["gamma_shape_parameter"] = row["A_Protein"]
            self.taniguichi2010counts[idx]["gamma_scale_parameter"] = row["B_Protein"]
            # Add a monomerId to each row
            if row["Gene_Name"] in self.geneSymbolToMonomerId:
                self.taniguichi2010counts[idx]["monomerId"] = (
                    self.geneSymbolToMonomerId[row["Gene_Name"]]
                )

    def _loadHouser2015Counts(self, validation_data_raw):
        # Load Houser Wilke PLoSCB 2015 dataset
        houser_dataset = validation_data_raw.houser2015_javier_table
        self.houser2015counts = np.zeros(
            len(houser_dataset),
            dtype=[
                ("monomerId", "U100"),
                ("gene_symbol", "U10"),
                ("counts_ave_exponential", np.float32),
                ("sample16_t3", np.float32),
                ("sample17_t4", np.float32),
                ("sample18_t5", np.float32),
                ("sample19_t6", np.float32),
                ("sample20_t8", np.float32),
                ("sample21_t24", np.float32),
                ("sample22_t48", np.float32),
                ("sample23_t168", np.float32),
                ("sample24_336", np.float32),
                ("sample25_t3", np.float32),
                ("sample26_t4", np.float32),
                ("sample27_t5", np.float32),
                ("sample28_t6", np.float32),
                ("sample29_t8", np.float32),
                ("sample30_t24", np.float32),
                ("sample31_t48", np.float32),
                ("sample32_t168", np.float32),
                ("sample34_336", np.float32),
                ("sample97_t3", np.float32),
                ("sample98_t4", np.float32),
                ("sample99_t5", np.float32),
                ("sample100_t6", np.float32),
                ("sample101_t8", np.float32),
                ("sample102_t24", np.float32),
                ("sample103_t48", np.float32),
                ("sample104_t168", np.float32),
                ("sample105_336", np.float32),
            ],
        )

        for idx, row in enumerate(houser_dataset):
            if row["gene_symbol"] in self.geneSymbolToMonomerId:
                self.houser2015counts[idx]["monomerId"] = self.geneSymbolToMonomerId[
                    row["gene_symbol"]
                ]
            self.houser2015counts[idx]["gene_symbol"] = row["gene_symbol"]
            # Hour 6 is the most clearly exponential time point, this is in columns labeled as t6
            exponential_average = np.mean(
                [
                    float(row["sample19_t6"]),
                    float(row["sample28_t6"]),
                    float(row["sample100_t6"]),
                ]
            )
            # Only include proteins observed at least once in this time step
            if exponential_average:
                self.houser2015counts[idx]["counts_ave_exponential"] = (
                    exponential_average
                )
            # Otherwise, remove this row from the dataset
            else:
                np.delete(self.houser2015counts, idx, 0)

            # Load the rest of the data as-is
            for fieldName in row:
                if fieldName == "gene_symbol" or fieldName == "":
                    continue
                self.houser2015counts[idx][fieldName] = row[fieldName]

    def _loadWisniewski2014Counts(self, validation_data_raw, knowledge_base_raw):
        dataset = validation_data_raw.wisniewski2014_supp2
        rep1 = np.array([x["rep1"] for x in dataset])
        rep2 = np.array([x["rep2"] for x in dataset])
        rep3 = np.array([x["rep3"] for x in dataset])
        avg = np.mean((rep1, rep2, rep3), axis=0)
        geneIds = [x["EcoCycID"] for x in dataset]

        monomer_ids = []
        avg_counts_filtered = []

        for i, (gene_id, avg_count) in enumerate(zip(geneIds, avg)):
            if gene_id in self.geneIdToMonomerId:
                monomer_ids.append(self.geneIdToMonomerId[gene_id])
                avg_counts_filtered.append(avg_count)

        nEntries = len(monomer_ids)

        wisniewski2014Data = np.zeros(
            nEntries,
            dtype=[
                ("monomerId", "U50"),
                ("avgCounts", "f8"),
            ],
        )

        wisniewski2014Data["monomerId"] = monomer_ids
        wisniewski2014Data["avgCounts"] = avg_counts_filtered

        self.wisniewski2014Data = wisniewski2014Data

    def _loadSchmidt2015Counts(self, validation_data_raw):
        dataset = validation_data_raw.schmidt2015_javier_table

        geneIds = [x["EcoCycID"] for x in dataset]
        monomerIds = [self.geneIdToMonomerId[x] for x in geneIds]

        glucoseCounts = [x["Glucose"] for x in dataset]
        lb_counts = [x["LB"] for x in dataset]
        nEntries = len(geneIds)

        schmidt2015Data = np.zeros(
            nEntries,
            dtype=[
                ("monomerId", "U50"),
                ("glucoseCounts", "f8"),
                ("LB_counts", "f8"),
            ],
        )

        schmidt2015Data["monomerId"] = monomerIds
        schmidt2015Data["glucoseCounts"] = glucoseCounts
        schmidt2015Data["LB_counts"] = lb_counts

        self.schmidt2015Data = schmidt2015Data

    def _load_li(self, validation_data_raw):
        monomers = []
        rich_rates = []
        minimal_rates = []
        for line in validation_data_raw.li_protein_synthesis_rates_2014:
            gene = line["Gene"]
            if (symbol := self.geneSymbolToMonomerId.get(gene)) is not None:
                monomers.append(symbol)
                rich_rates.append(int(str(line["MOPS complete"]).strip("[]")))
                minimal_rates.append(int(str(line["MOPS minimal"]).strip("[]")))

        self.li_2014 = np.zeros(
            len(monomers),
            dtype=[
                ("monomer", f"U{max([len(monomer) for monomer in monomers])}"),
                ("rich_rate", "f8"),
                ("minimal_rate", "f8"),
            ],
        )

        self.li_2014["monomer"] = monomers
        self.li_2014["rich_rate"] = rich_rates
        self.li_2014["minimal_rate"] = minimal_rates


class ReactionFlux(object):
    """ReactionFlux"""

    def __init__(self, validation_data_raw, knowledge_base_raw):
        self._loadToya2010Fluxes(validation_data_raw)

    def _loadToya2010Fluxes(self, validation_data_raw):
        # Load Toya 2010 Biotech Prog central carbon metabolism C13 flux dataset
        toya_dataset = validation_data_raw.toya_2010_central_carbon_fluxes
        self.toya2010fluxes = np.zeros(
            len(toya_dataset),
            dtype=[
                ("reactionID", "U100"),
                ("reactionFlux", Unum),
                ("reactionFluxStdev", Unum),
            ],
        )
        for idx, row in enumerate(toya_dataset):
            self.toya2010fluxes[idx]["reactionID"] = row["reactionID"]
            self.toya2010fluxes[idx]["reactionFlux"] = row["flux"]
            self.toya2010fluxes[idx]["reactionFluxStdev"] = row[
                "flux standard deviation"
            ]


class EssentialGenes(object):
    """EssentialGenes"""

    def __init__(self, validation_data_raw):
        self._load_essential_genes(validation_data_raw)

    def _load_essential_genes(self, validation_data_raw):
        self.essential_genes = []
        self.essential_cistrons = []
        self.essential_proteins = []

        for row in validation_data_raw.essential_genes:
            self.essential_genes.append(row["FrameID"])
            self.essential_cistrons.append(row["rnaID"])
            self.essential_proteins.append(
                row["proteinID"] + "[" + row["proteinLoc"] + "]"
            )


class GeneFunctions(object):
    """GeneFunctions"""

    def __init__(self, validation_data_raw):
        self._loadGeneFunctions(validation_data_raw)

    def _loadGeneFunctions(self, validation_data_raw):
        self.geneFunctions = {}

        for row in validation_data_raw.geneFunctions:
            self.geneFunctions[row["FrameID"]] = row["Function"]


class MacromolecularGrowthRateModulation(object):
    """MacromolecularGrowthRateModulation"""

    def __init__(self, validation_data_raw):
        self._load_macromolecular_growth_rate_modulation(validation_data_raw)

    def _load_macromolecular_growth_rate_modulation(self, validation_data_raw):
        dataset = validation_data_raw.macromolecular_growth_rate_modulation
        # List of data value names extracted from the raw flat file
        values = list(dataset[0].keys())
        for value in values:
            # Store numerical value of raw data into a temporary ndarray
            temp = np.zeros(len(dataset))
            for idx, row in enumerate(dataset):
                y = row[value]
                temp[idx] = Unum.coerceToUnum(y).asNumber()
            # Set each ndarray as the corresponding attribute, while tagging the data's original units onto the ndarray
            setattr(self, value, Unum(Unum.coerceToUnum(dataset[0][value])._unit, temp))
