import requests
import xmltodict
import pprint as pp
from wholecell.utils import units

# s = requests.Session() # create session
# # Post login credentials to session:
# s.post('https://websvc.biocyc.org/credentials/login/', data={'email':'', 'password': ''})
#
# # example entry
# # entity = 'PWY0-1356'
# entity = 'MIX0-76'
# req_str = f'https://brg-preview.ai.sri.com/getxml?id=ECOLI:{entity}&detail=full'
#
# r = s.get(req_str)
# if r.status_code != 200:
#     print(entity, r.status_code)
#
# o = xmltodict.parse(r.content)
#
# composition = o['ptools-xml']['growth-media']['composition']
# components_dict = {x['Compound']['@frameid'] : float(x['conc-m']['#text']) for x in composition}
# #pp.pprint(o['ptools-xml'])
#
#
# import ipdb; ipdb.set_trace()
# Media in ecocyc: BHI on metacyc, LB, M9, LB + glycerol (check),
# MOPS, M63, LB + HOMOPIPES (check), LB low salt (check), EZ defined rich medium,
# MES-LB (check), M9C (check), BHI agar (same as BHI but with agar),
# https://www.fda.gov/food/laboratory-methods-food/bam-media-m24-brain-heart-infusion-bhi-broth-and-agar#:~:text=To%20prepare%20brain%20heart%20infusion,concentration%20of%202%2D3%25.
# Not in ecocyc: MOPS + M9 or M9 + MOPS + tannic acid (?),
# Terrific Broth, DMEM, K medium, modified complex (read paper),
# glucose-limited minimal media in chemostat (check), Davis MM



# terrific broth: https://www.emdmillipore.com/US/en/product/Terrific-Broth-Novagen,EMD_BIO-71754?ReferrerURL=https%3A%2F%2Fwww.google.com%2F#:~:text=The%20composition%20of%20the%20Terrific,2.2%20g%20potassium%20phosphate%2C%20monobasic.

# Make a simplified version, where it's just the media components as lists
#SIMPLIFIED_MEDIA_COMPONENTS = {
#    "TB": {"amino acids", "yeast extract"}
#    "Davis MM": {"glucose", "citrate", "phosphate", "sulfate", "ammonium", "K+", "Na+", "Mg2+"},
#    "DMEM": {"Na+", "Ca2+", "Fe3+", "Mg2+", "K+", "nitrate", "sulfate", "phosphate", "Cl-", "amino acids", "vitamins", "choline", "inositol", "glucose"},
#    "K medium": {"Na+", "nitrate", "ammonium", "Cl-", "glycerophosphate", "selenous acid", "EDTA", "}
#                 }}








# TODO: convert tryptone, peptone, etc. into all 20 amino acids
# TODO: convert yeast extract, BHI, etc. into useful stuff:

#

MEDIA_RECIPE = {
    "TB": {"tryptone": 12 * units.g / units.L,
           "yeast extract": 24 * units.g / units.L,
           "dipotassium phosphate": 9.4 * units.g / units.L,
           "monopotassium phosphate": 2.2 * units.g / units.L
           },
    "Davis MM": {
        "glucose": 1.0 * units.g / units.L,
        "dipotassium phosphate": 2.0 * units.g / units.L,
        "monopotassium phosphate": 2.0 * units.g / units.L,
        "sodium citrate": 0.5 * units.g / units.L,
        "magnesium sulfate": 0.1 * units.g / units.L,
        "ammonium sulfate": 1.0 * units.g / units.L
    }, # pH = 7,
    "DMEM": {
        "calcium chloride": 0.265 * units.g / units.L,
        "iron (III) nitrate": 0.0001 * units.g / units.L,
        "magnesium sulfate": 0.09767 * units.g / units.L,
        "potassium chloride": 0.4 * units.g / units.L,
        "sodium bicarbonate": 3.7 * units.g / units.L,
        "sodium chloride": 6.4 * units.g / units.L,
        "monosodium phosphate": 0.109 * units.g / units.L,
        "arginine": 0.084 * units.g / units.L,
        "cystine": 0.0626 * units.g / units.L,
        "glutamine": 0.584 * units.g / units.L,
        "glycine": 0.03 * units.g / units.L,
        "histidine": 0.042 * units.g / units.L,
        "isoleucine": 0.105 * units.g / units.L,
        "leucine": 0.105 * units.g / units.L,
        "lysine": 0.146 * units.g / units.L,
        "methionine": 0.03 * units.g / units.L,
        "phenylalanine": 0.066 * units.g / units.L,
        "serine": 0.042 * units.g / units.L,
        "threonine": 0.095 * units.g / units.L,
        "tryptophan": 0.016 * units.g / units.L,
        "tyrosine": 0.12037 * units.g / units.L,
        "valine": 0.094 * units.g / units.L,
        "choline chloride": 0.004 * units.g / units.L,
        "folic acid": 0.004 * units.g / units.L,
        "inositol": 0.0072 * units.g / units.L,
        "niacinamide": 0.004 * units.g / units.L,
        "pantothenic acid": 0.004 * units.g / units.L,
        "pyridoxine": 0.00404 * units.g / units.L,
        "riboflavin": 0.0004 * units.g / units.L,
        "thiamine": 0.004 * units.g / units.L,
        "glucose": 1.0 * units.g / units.L
    },
    "K medium": {
        "sodium nitrate": 0.075 * units.g / units.L,
        "ammonium chloride": 0.00267 * units.g / units.L,
        "disodium glycerophosphate": 0.00216 * units.g / units.L,
        "selenous acid": 0.00000129 * units.g / units.L,
        "disodium edta": 0.00436 * units.g / units.L,
        "iron (III) chloride": 0.00315 * units.g / units.L,
        "manganese (II) chloride": 0.00018 * units.g / units.L,
        "copper (II) sulfate": 0.0000025 * units.g / units.L,
        "zinc (II) sulfate": 0.000022 * units.g / units.L,
        "cobalt (II) chloride": 0.00001 * units.g / units.L,
        "disodium molybdate": 0.0000063 * units.g / units.L,
        "cyanocobalamin": 0.0000005 * units.g / units.L,
        "biotin": 0.0000005 * units.g / units.L,
        "thiamine": 0.0001 * units.g / units.L,
        # TODO: "pH": 8.0,
    },
    "modified complex": {
        "glucose": 20 * units.g / units.L,
        "yeast extract": 5 * units.g / units.L,
        "tryptone": 5 * units.g / units.L,
        "dipotassium phosphate": 7 * units.g / units.L,
        "monopotassium phosphate": 5.5 * units.g / units.L,
        "cysteine": 0.5 * units.g / units.L,
        "ammonium sulfate": 1 * units.g / units.L,
        "magnesium sulfate": 0.25 * units.g / units.L, # TODO: actually a hydrate
        "calcium chloride": 0.021 * units.g / units.L,
        "cobalt (II) nitrate": 0.029 * units.g / units.L,
        "iron (II) ammonium sulfate": 0.039 * units.g / units.L,
        "nicotinic acid": 2 * units.mg / units.L,
        "disodium selenate": 0.172 * units.g / units.L,
        "nickel (II) chloride": 0.02 * units.g / units.L,
        "manganese (II) chloride": 0.5e-2 * units.g / units.L,
        "boric acid": 0.1e-2 * units.g / units.L,
        "aluminum (III) potassium sulfate": 0.01e-2 * units.g / units.L,
        "copper (II) chloride": 0.001e-2 * units.g / units.L,
        "disodium edta": 0.5e-2 * units.g / units.L,
        "ammonium heptamolybdate": 0.4 * units.mg / units.L
    },
    "glucose-limited minimal media": { # TODO: chemostat
        "disodium phosphate": 10.21 * units.g / units.L,
        "monopotassium phosphate": 3 * units.g / units.L,
        "ammonium sulfate": 1.77 * units.g / units.L,
        "magnesium chloride": 130 * units.mg / units.L, # TODO: hexahydrate, and much below
        "calcium carbonate": 80  * units.mg / units.L,
        "iron (III) chloride": 77  * units.mg / units.L,
        "manganese (II) chloride": 11  * units.mg / units.L,
        "copper (II) sulfate": 1.5  * units.mg / units.L,
        "copper (II) chloride": 1.3  * units.mg / units.L,
        "zinc (II) oxide": 4 * units.mg / units.L,
        "boric acid": 1.2 * units.mg / units.L,
        "disodium molybdate": 10 * units.mg / units.L,
        "disodium edta": 706.55 * units.mg / units.L,
        "glucose": 1 * units.g / units.L,
        # TODO: pH 6.7 (check?)
    }, # https://onlinelibrary.wiley.com/doi/10.1002/bem.21709
    "LB": {"chloride": 170 * units.mmol / units.L,
        "Na+": 170 * units.mmol / units.L,
        "tryptone": 10 * units.g / units.L,
        "yeast extract": 5 * units.g / units.L
        },
    "BHI": {
        "Na+": 121.2 * units.mmol / units.L,
        "chloride": 86 * units.mmol / units.L,
        "phosphate": 17.6 * units.mmol / units.L,
        "beef heart": 5 * units.g / units.L,
        "calf brains": 12.5 * units.g / units.L,
        "glucose": 2 * units.g / units.L,
        "peptone": 10 * units.g / units.L,
    },
}

MEDIA_COMPOSITION = {
    "M9": {
        "Na+": 103.7 * units.mmol / units.L,
        "phosphate": 69.454 * units.mmol / units.L,
        "chloride": 27.25 * units.mmol / units.L,
        "K+": 21.883 * units.mmol/units.L,
        "ammonium": 18.695 * units.mmol / units.L,
        "Mg2+": 2 * units.mmol / units.L,
        "sulfate": 2 * units.mmol / units.L,
    },
    "MOPS": {
        "chloride": 60.557 * units.mmol / units.L,
        "Na+": 50 * units.mmol / units.L,
        "MOPS": 40 * units.mmol / units.L,
        "boric acid": 24.732 * units.mmol / units.L,
        "ammonium": 9.5 * units.mmol / units.L,
        "tricine": 4 * units.mmol / units.L,
        "K+": 3.192 * units.mmol / units.L,
        "phosphate": 1.32 * units.mmol / units.L,
        "Mg2+": 528 * units.umol / units.L,
        "sulfate": 286.02 * units.umol / units.L,
        "Fe2+": 10 * units.umol / units.L,
        "Ca2+": 500e-3 * units.umol / units.L,
        "Mn2+": 80e-3 * units.umol / units.L,
        "Co2+": 30e-3 * units.umol / units.L,
        "Cu2+": 10e-3 * units.umol / units.L,
        "Zn2+": 10e-3 * units.umol / units.L,
        "heptamolybdate": 3e-3 * units.umol / units.L,
    },
    "M63": {
        "K+": 100 * units.mmol / units.L,
        "phosphate": 100 * units.mmol / units.L,
        "ammonium": 30 * units.mmol / units.L,
        "sulfate": 16.002 * units.mmol / units.L,
        "Mg2+": 1 * units.mmol / units.L,
        "Fe2+": 1.8 * units.umol / units.L,
    },
    "EZ Rich Defined Medium": {
        "chloride": 60.557 * units.mmol / units.L,
        "Na+": 50 * units.mmol / units.L,
        "MOPS": 40 * units.mmol / units.L,
        "boric acid": 24.732 * units.mmol / units.L,
        "glucose": 19.84 * units.mmol / units.L,
        "serine": 10 * units.mmol / units.L,
        "ammonium": 9.5 * units.mmol / units.L,
        "arginine": 5.2 * units.mmol / units.L,
        "tricine": 4 * units.mmol / units.L,
        "K+": 3.192 * units.mmol / units.L,
        "phosphate": 1.32 * units.mmol / units.L,
        "alanine": 800 * units.umol / units.L,
        "leucine": 800 * units.umol / units.L,
        "glycine": 800 * units.umol / units.L,
        "glutamate": 600 * units.umol / units.L,
        "glutamine": 600 * units.umol / units.L,
        "valine": 600 * units.umol / units.L,
        "Mg2+": 528 * units.umol / units.L,
        "proline": 400 * units.umol / units.L,
        "isoleucine": 400 * units.umol / units.L,
        "threonine": 400 * units.umol / units.L,
        "asparagine": 400 * units.umol / units.L,
        "aspartate": 400 * units.umol / units.L,
        "lysine": 400 * units.umol / units.L,
        "phenylalanine": 400 * units.umol / units.L,
        "sulfate": 286.02 * units.umol / units.L,
        "histidine": 200 * units.umol / units.L,
        "adenine": 200 * units.umol / units.L,
        "guanine": 200 * units.umol / units.L,
        "cytosine": 200 * units.umol / units.L,
        "uracil": 200 * units.umol / units.L,
        "methionine": 200 * units.umol / units.L,
        "tyrosine": 200 * units.umol / units.L,
        "cysteine": 100 * units.umol / units.L,
        "tryptophan": 100 * units.umol / units.L,
        "pantothenate": 20 * units.umol / units.L,
        "Ca2+": 10.5 * units.umol / units.L,
        "Fe2+": 10 * units.umol / units.L,
        "thiamine": 10 * units.umol / units.L,
        "4-aminobenzoate": 10 * units.umol / units.L,
        "4-hydroxybenzoate": 10 * units.umol / units.L,
        "2,3-dihydroxybenzoate": 10 * units.umol / units.L,
        "Mn2+": 80e-3 * units.umol / units.L,
        "Co2+": 30e-3 * units.umol / units.L,
        "Zn2+": 10e-3 * units.umol / units.L,
        "Cu2+": 10e-3 * units.umol / units.L,
        "heptamolybdate": 3e-3 * units.umol / units.L,
    },
}

MEDIA_COMPOSITION["M9C"] = {k:v for k, v in MEDIA_COMPOSITION["M9"].items()}
MEDIA_COMPOSITION["M9C"].update({"glucose": 22.203 * units.mmol / units.L,
                                                           "casamino acids": 4 * units.g / units.L})
MEDIA_RECIPE["LB + glycerol"] = {k:v for k, v in MEDIA_RECIPE["LB"].items()}
MEDIA_RECIPE["LB + glycerol"].update({"glycerol": None}) # TODO: glycerol 10% v/v, also temperature was 30C
MEDIA_RECIPE["LB low salt"] = {k:v for k, v in MEDIA_RECIPE["LB"].items()}
MEDIA_RECIPE["LB low salt"].update({ "Na+": 5 * units.g / units.L, "chloride": 5 * units.g / units.L,})
MEDIA_RECIPE["LB+HOMOPIPES"] = {k:v for k, v in MEDIA_RECIPE["LB"].items()}
MEDIA_RECIPE["LB+HOMOPIPES"].update({"HOMOPIPES": 50 * units.mmol / units.L,})
MEDIA_RECIPE["MES-LB"] = {k:v for k, v in MEDIA_RECIPE["LB"].items()}
MEDIA_RECIPE["MES-LB"].update({"MES": None})
MEDIA_RECIPE["BHI Agar"] = {k:v for k, v in MEDIA_RECIPE["BHI"].items()}

RECIPE_TO_COMPOSITION = {
    "calcium chloride": {"Ca2+": 1, "chloride": 2},
    "iron (III) nitrate": {"Fe3+": 1, "nitrate": 3},
    "magnesium sulfate": {"Mg2+": 1, "sulfate": 1},
    "potassium chloride": {"K+": 1, "chloride": 1},
    "sodium bicarbonate": {"Na+": 1, "bicarbonate": 1},
    "monosodium phosphate": {"Na+": 1, "phosphate": 1},
    "choline chloride": {"choline": 1, "chloride": 1},
    "sodium chloride": {"Na+": 1, "chloride": 1},
    "dipotassium phosphate": {"K+": 2, "phosphate": 1},
    "sodium citrate": {"Na+": 3, "citrate": 1},
    "ammonium sulfate": {"ammonium": 2, "sulfate": 1},
    "sodium nitrate": {"Na+": 1, "nitrate": 1},
    "ammonium chloride": {"ammonium": 1, "chloride": 1},
    "disodium glycerophosphate": {"Na+": 2, "glycerophosphate": 1},
    "disodium edta": {"Na+": 2, "edta": 1},
    "iron (III) chloride": {"Fe3+": 1, "chloride": 3},
    "manganese (II) chloride": {"Mn2+": 1, "chloride": 2},
    "copper (II) sulfate": {"Cu2+": 1, "sulfate": 1},
    "zinc (II) sulfate": {"Zn2+": 1, "sulfate": 1},
    "cobalt (II) chloride": {"Co2+": 1, "chloride": 2},
    "disodium molybdate": {"Na+": 2, "molybdate": 1},
    "monopotassium phosphate": {"K+": 1, "phosphate": 1},
    "disodium phosphate": {"Na+": 2, "phosphate": 1},
    "cobalt (II) nitrate": {"Co2+": 1, "nitrate": 2},
    "iron (II) ammonium sulfate": {"Fe2+": 1, "ammonium": 2, "sulfate": 2},
    "disodium selenate": {"Na+": 2, "selenate": 1},
    "nickel (II) chloride": {"Ni2+": 1, "chloride": 2},
    "aluminum (III) potassium sulfate": {"Al3+": 1, "K+": 1, "sulfate": 1},
    "copper (II) chloride": {"Cu2+": 1, "chloride": 2},
    "ammonium heptamolybdate": {"ammonium": 6, "heptamolybdate": 1},
    "calcium carbonate": {"Ca2+": 1, "bicarbonate": 1},
    "magnesium chloride": {"Mg2+": 1, "chloride": 2},
    "K2HPO4": {"K+": 2, "phosphate": 1},
    "KCl": {"K+": 1, "chloride": 1},
    "L-cysteine-HCl": {"cysteine": 1, "chloride": 1},
    "L-methionine": {"methionine": 1},
    "MgSO4": {"Mg2+": 1, "sulfate": 1},
    "MnCl2": {"Mn2+": 1, "chloride": 2},
    "Na2S-9H2O": {"Na+": 2, "sulfide": 1},
    "NaCl": {"Na+": 1, "chloride": 1},
    "acidified-sodium-nitrate": {"Na+": 1, "nitrate": 1, "low pH": None}, # TODO: pH
    "pantothenic acid": {"pantothenate": 1},
    "sodiumbenzoate": {"Na+": 1, "benzoate": 1},
    "yeast extract": {"glucose": None, "phosphate": None, "Na+": None, "K+": None, "chloride": None, "Mg2+": None, "Ca2+": None,
                      "ammonium": None,
                      "glycine": None, "alanine": None, "leucine": None, "methionine": None, "isoleucine": None, "valine": None,
                      "tryptophan": None, "proline": None, "tyrosine": None, "serine": None, "threonine": None, "cysteine": None,
                      "histidine": None, "lysine": None, "arginine": None, "glutamine": None, "asparagine": None, "aspartate": None,
                      "glutamate": None, "phenylalanine": None,
                      "adenine": None, "guanine": None, "cytosine": None, "uracil": None, "thymine": None,
                      "thiamine": None, "riboflavin": None, "niacinamide": None, "pantothenate": None, "pyridoxine": None, "biotin": None, "folic acid": None,
                      "cyanocobalamin": None, "choline": None, "Fe3+": None, "Mn2+": None, "Al3+": None, "Ni2+": None, "Cu2+": None, "Fe2+": None,
                      "Zn2+": None, "Co2+": None, "yeast extract other": None
                      }, # TODO: molybdate/heptamolybdate? boric acid, selenate/selenous acid? biarbonate? etc.
                        # TODO: probably should take glucose out maybe?
                        # TODO: ask about GSE21551, does the LB mahve glucose in it??
    "tryptone": {"glycine": None, "alanine": None, "leucine": None, "methionine": None, "isoleucine": None, "valine": None,
                      "tryptophan": None, "proline": None, "tyrosine": None, "serine": None, "threonine": None, "cysteine": None,
                      "histidine": None, "lysine": None, "arginine": None, "glutamine": None, "asparagine": None, "aspartate": None,
                      "glutamate": None, "phenylalanine": None, "tryptone-peculiar": None},
    "peptone": {"glycine": None, "alanine": None, "leucine": None, "methionine": None, "isoleucine": None, "valine": None,
                      "tryptophan": None, "proline": None, "tyrosine": None, "serine": None, "threonine": None, "cysteine": None,
                      "histidine": None, "lysine": None, "arginine": None, "glutamine": None, "asparagine": None, "aspartate": None,
                      "glutamate": None, "phenylalanine": None, "peptone-peculiar": None},
    "casamino acids": {"glycine": None, "alanine": None, "leucine": None, "methionine": None, "isoleucine": None, "valine": None,
                      "tryptophan": None, "proline": None, "tyrosine": None, "serine": None, "threonine": None, "cysteine": None,
                      "histidine": None, "lysine": None, "arginine": None, "glutamine": None, "asparagine": None, "aspartate": None,
                      "glutamate": None, "phenylalanine": None, "casamino-pecular": None},
    "amino acids": {"glycine": None, "alanine": None, "leucine": None, "methionine": None, "isoleucine": None, "valine": None,
                      "tryptophan": None, "proline": None, "tyrosine": None, "serine": None, "threonine": None, "cysteine": None,
                      "histidine": None, "lysine": None, "arginine": None, "glutamine": None, "asparagine": None, "aspartate": None,
                      "glutamate": None, "phenylalanine": None,},
    "aa": {"glycine": None, "alanine": None, "leucine": None, "methionine": None, "isoleucine": None, "valine": None,
                      "tryptophan": None, "proline": None, "tyrosine": None, "serine": None, "threonine": None, "cysteine": None,
                      "histidine": None, "lysine": None, "arginine": None, "glutamine": None, "asparagine": None, "aspartate": None,
                      "glutamate": None, "phenylalanine": None,},
    "beef heart": {"phosphate": None, "Na+": None, "K+": None, "chloride": None, "Mg2+": None, "Ca2+": None,
                      "ammonium": None,
                      "glycine": None, "alanine": None, "leucine": None, "methionine": None, "isoleucine": None, "valine": None,
                      "tryptophan": None, "proline": None, "tyrosine": None, "serine": None, "threonine": None, "cysteine": None,
                      "histidine": None, "lysine": None, "arginine": None, "glutamine": None, "asparagine": None, "aspartate": None,
                      "glutamate": None, "phenylalanine": None,
                      "adenine": None, "guanine": None, "cytosine": None, "uracil": None, "thymine": None,
                      "thiamine": None, "riboflavin": None, "niacinamide": None, "pantothenate": None, "pyridoxine": None, "biotin": None, "folic acid": None,
                      "cyanocobalamin": None, "choline": None, "Fe3+": None, "Mn2+": None, "Al3+": None, "Ni2+": None, "Cu2+": None, "Fe2+": None,
                      "Zn2+": None, "Co2+": None, "beef heart other": None
                      },
    "calf brains": {"glucose": None, "phosphate": None, "Na+": None, "K+": None, "chloride": None, "Mg2+": None, "Ca2+": None,
                      "ammonium": None,
                      "glycine": None, "alanine": None, "leucine": None, "methionine": None, "isoleucine": None, "valine": None,
                      "tryptophan": None, "proline": None, "tyrosine": None, "serine": None, "threonine": None, "cysteine": None,
                      "histidine": None, "lysine": None, "arginine": None, "glutamine": None, "asparagine": None, "aspartate": None,
                      "glutamate": None, "phenylalanine": None,
                      "adenine": None, "guanine": None, "cytosine": None, "uracil": None, "thymine": None,
                      "thiamine": None, "riboflavin": None, "niacinamide": None, "pantothenate": None, "pyridoxine": None, "biotin": None, "folic acid": None,
                      "cyanocobalamin": None, "choline": None, "Fe3+": None, "Mn2+": None, "Al3+": None, "Ni2+": None, "Cu2+": None, "Fe2+": None,
                      "Zn2+": None, "Co2+": None, "calf brain other": None
                      },
    "biofilm with R1drd19 plasmid": {"biofilm": None, "R1drd19 plasmid": None}
    }

# Possibly relevant molecules:
# glucose, citrate, inositol, glycerophosphate, glycerol, mannitol, gluconate, propylene glycol, fumarate, sucrose, lactate, arabinose, ethanol, formate, acetate, succinate
# phosphate, sulfate, ammonium, bicarbonate, nitrate, sulfide
# major ions
# trace metals (including Fe2+, Fe3+, zinc (II) oxide, etc.), including selenous acid, selenate, boric acid
# amino acids
# B vitamins and choline
# nucleotides
# questionable: hydroxyindole, isatin, indole, hydroxyurea, benzoate derivatives, DPD


class MediaComps(object):
    def __init__(self):
        self.media_comp = MEDIA_COMPOSITION
        self.recipe_to_comp = RECIPE_TO_COMPOSITION

        for recipe in MEDIA_RECIPE:
            MEDIA_COMPOSITION[recipe] = {}
            for x in MEDIA_RECIPE[recipe]:
                if x in RECIPE_TO_COMPOSITION:
                    MEDIA_COMPOSITION[recipe].update({y: None for y in RECIPE_TO_COMPOSITION[x]})  # TODO: do smth about concentrations
                else:
                    MEDIA_COMPOSITION[recipe].update({x: MEDIA_RECIPE[recipe][x]})

        all_compounds = {}
        for media in MEDIA_COMPOSITION:
            all_compounds.update(MEDIA_COMPOSITION[media])
        all_compounds = list(all_compounds.keys())

        self.all_compounds = all_compounds

    def convert_to_composition(self, components):
        compositions = []
        for comp in components:
            minicomp = []
            for x in comp:
                if x in RECIPE_TO_COMPOSITION:
                    minicomp.extend(list(RECIPE_TO_COMPOSITION[x].keys()))
                else:
                    minicomp.append(x)
            compositions.append(minicomp)

        return compositions


# TODO: agar or not agar?


