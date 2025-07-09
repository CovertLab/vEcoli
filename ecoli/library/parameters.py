import os
import pandas as pd
from collections import namedtuple
import numpy as np
from scipy import constants

from vivarium.library.topology import get_in, assoc_path
from vivarium.library.units import units

from wholecell.utils.filepath import ROOT_PATH
from ecoli.library.cell_wall.column_sampler import fit_strand_term_p


class Parameter:
    def __init__(self, value, source="", canonicalize=None, note="", latex_source=""):
        self.value = value
        self.source = source
        self.canonicalize = canonicalize or (lambda x: x)
        self.note = note
        self.latex_source = latex_source


class ParameterStore:
    def __init__(self, parameters, derivation_rules=None):
        self._parameters = parameters
        self.derive_parameters(derivation_rules or {})

    def get_parameter(self, path):
        param_obj = get_in(self._parameters, path)
        if not param_obj:
            raise RuntimeError(f"No parameter found at path {path}")
        return param_obj

    def get_value(self, path):
        param_obj = self.get_parameter(path)
        return param_obj.canonicalize(param_obj.value)

    def get(self, path):
        return self.get_value(path)

    def add(self, path, parameter):
        assert get_in(self._parameters, path) is None
        assoc_path(self._parameters, path, parameter)

    def derive_parameters(self, derivation_rules):
        for path, deriver in derivation_rules.items():
            new_param = deriver(self)
            self.add(path, new_param)


PARAMETER_DICT = {
    "ampicillin": {
        "permeability": {
            "outer": Parameter(
                0.28e-5 * units.cm / units.sec,
                "Kojima and Nikaido (2013)",
                note="This is total, not per-porin, permeability.",
                latex_source="kojima_permeation_2013",
            ),
        },
        "mic": Parameter(
            2 * units.micrograms / units.mL,
            "Mazzariol, Cornaglia, and Nikaido (2000)",
            lambda x: (
                # Divide by molecular weight from PubChem.
                x / (349.4 * units.g / units.mol)
            ).to(units.mM),
        ),
        "efflux": {
            "vmax": Parameter(
                0.069 * units.nmol / units.mg / units.sec,
                "Kojima and Nikaido (2013)",
                latex_source="kojima_permeation_2013",
            ),
            "km": Parameter(
                2.16e-3 * units.mM,
                "Kojima and Nikaido (2013)",
                latex_source="kojima_permeation_2013",
            ),
            "n": Parameter(
                1.9 * units.count,
                "Kojima and Nikaido (2013)",
                latex_source="kojima_permeation_2013",
            ),
        },
        "hydrolysis": {
            "kcat": Parameter(
                6.5 / units.sec,
                "Mazzariol, Cornaglia, and Nikaido (2000)",
                latex_source="mazzariol_contributions_2000",
            ),
            "km": Parameter(
                0.9e-3 * units.mM,
                "Mazzariol, Cornaglia, and Nikaido (2000)",
                latex_source="mazzariol_contributions_2000",
            ),
            "n": Parameter(
                1 * units.count,
                "Mazzariol, Cornaglia, and Nikaido (2000)",
                latex_source="mazzariol_contributions_2000",
            ),
        },
        "pbp_binding": {
            "K_A (micromolar)": {
                "PBP1A": Parameter(0.7 * units.uM, "Kocaoglu and Carlson 2015"),
                "PBP1B": Parameter(1.27 * units.uM, "Catherwood et al. 2020"),
            }
        },
        "molar_mass": Parameter(349.406 * units.g / units.mol),
    },
    "cephaloridine": {
        "permeability": {
            "outer": Parameter(
                (52.6e-5 + 4.5e-5) * units.cm / units.sec,
                "Nikaido, Rosenberg, and Foulds (1983)",
                note="This is total, not per-porin, permeability",
                latex_source="nikaido_porin_1983",
            ),
        },
        # Cell-wide permeability with only one porin present.
        "porin_specific_permeability": {
            "outer": {
                "ompf": Parameter(
                    52.6e-5 * units.cm / units.sec,
                    "Nikaido, Rosenberg, and Foulds (1983)",
                    latex_source="nikaido_porin_1983",
                ),
                "ompc": Parameter(
                    4.5e-5 * units.cm / units.sec,
                    "Nikaido, Rosenberg, and Foulds (1983)",
                    latex_source="nikaido_porin_1983",
                ),
            },
        },
        "mic": Parameter(
            0.5 * units.micrograms / units.mL,
            "Rolinson (1980)",
            lambda x: (
                # Divide by molecular weight from PubChem.
                x / (415.5 * units.g / units.mol)
            ).to(units.mM),
        ),
        "efflux": {
            "vmax": Parameter(
                1.82 * units.nmol / units.mg / units.sec,
                "Nagano and Nikaido (2009)",
                latex_source="nagano_kinetic_2009",
            ),
            "km": Parameter(
                0.288 * units.mM,
                "Nagano and Nikaido (2009)",
                latex_source="nagano_kinetic_2009",
            ),
            "n": Parameter(
                1.75 * units.count,
                "Nagano and Nikaido (2009)",
                latex_source="nagano_kinetic_2009",
            ),
        },
        "hydrolysis": {
            "kcat": Parameter(
                130 / units.sec,
                "Galleni et al. (1988)",
            ),
            "km": Parameter(
                0.17 * units.mM,
                "Galleni et al. (1988)",
            ),
            "n": Parameter(1 * units.count),
        },
        "pbp_binding": {
            "K_A": {
                "PBP1A": Parameter(0.25 * (units.ug / units.mL), "Curtis et al. 1979"),
                "PBP1B": Parameter(2.5 * (units.ug / units.mL), "Curtis et al. 1979"),
            }
        },
        "molar_mass": Parameter(415.488 * units.g / units.mol),
    },
    "tetracycline": {
        "permeability": {
            "outer_without_porins": Parameter(
                0.7e-7 * units.cm / units.sec,
                "Thanassi, Suh, and Nikaido (1995) p. 1004",
                latex_source="thanassi_role_1995",
            ),
            "outer_with_porins": Parameter(
                1e-5 * units.cm / units.sec,
                "Thanassi, Suh, and Nikaido (1995) p. 1005",
                latex_source="thanassi_role_1995",
            ),
            "inner": Parameter(
                3e-6 * units.cm / units.sec,
                "Thanassi, Suh, and Nikaido (1995) p. 1004",
                latex_source="thanassi_role_1995",
            ),
        },
        "charge": Parameter(1 * units.count),
        "efflux": {
            "vmax": Parameter(
                0.2 * units.nmol / units.mg / units.min,
                "Thanassi, Suh, and Nikaido (1995) p. 1004",
                latex_source="thanassi_role_1995",
            ),
            "km": Parameter(
                200 * units.uM,
                "Thanassi, Suh, and Nikaido (1995) p. 1004",
                latex_source="thanassi_role_1995",
            ),
            "n": Parameter(
                1 * units.count,
            ),
        },
        "mic": Parameter(
            1.5 * units.mg / units.L,
            """Middle of values reported by Thanassi, Suh, and Nikaido (1995)
            p. 1005, Cuisa et al. (2022), and Prochnow et al. (2019)""",
            latex_source="thanassi_role_1995",
            canonicalize=lambda x: (
                # Divide by molecular weight from PubChem.
                x / (444.4 * units.g / units.mol)
            ).to(units.mM),
        ),
        "mass": Parameter(
            444.4 * units.g / units.mol,
            "Tetracycline PubChem",
            note="https://pubchem.ncbi.nlm.nih.gov/compound/Tetracycline",
        ),
    },
    "shape": {
        "periplasm_fraction": Parameter(
            0.2,
            "Stock et al. (1977)",
        ),
        "initial_cell_mass": Parameter(
            1170 * units.fg,
            "Model",
        ),
        "initial_cell_volume": Parameter(
            1.2 * units.fL,
            "Model",
        ),
        "initial_outer_area": Parameter(
            4.52 * units.um**2,
            "Model",
        ),
        "average_cell_mass": Parameter(
            1640.6570410485792 * units.fg,
            "Simulation 0a2cd6816d36d408470445ff654371f07cd3f9f8",
        ),
        "average_dry_mass": Parameter(
            492.7365227132813 * units.fg,
            "Simulation 0a2cd6816d36d408470445ff654371f07cd3f9f8",
        ),
        "average_cell_volume": Parameter(
            1.4915064009532537 * units.fL,
            "Simulation 0a2cd6816d36d408470445ff654371f07cd3f9f8",
        ),
        # TODO: Recompute average surface area with correct equation
        "average_outer_area": Parameter(
            6.227824991612169 * units.um**2,
            "Simulation 0a2cd6816d36d408470445ff654371f07cd3f9f8",
        ),
    },
    "cell_wall": {
        "strand_length_data": Parameter(
            os.path.join(
                ROOT_PATH,
                "reconstruction/ecoli/flat/cell_wall/murein_strand_length_distribution.csv",
            ),
            "Obermann, W., & Höltje, J. (1994).",
        ),
        "upper_mean": Parameter(
            45, "Vollmer, W., Blanot, D., & De Pedro, M. A. (2008)."
        ),
        "critical_radius": Parameter(
            20 * units.nm,
            "Daly, K. E., Huang, K. C., Wingreen, N. S., & Mukhopadhyay, R. (2011).",
        ),
        "cell_radius": Parameter(0.5 * units.um, "Cell shape process"),
        "disaccharide_height": Parameter(
            1.03 * units.nm, "Vollmer, W., & Höltje, J.-V. (2004)."
        ),
        "disaccharide_width": Parameter(
            1.4 * units.nm,
            "Turner, R. D., Mesnage, S., Hobbs, J. K., & Foster, S. J. (2018).",
        ),
        "inter_strand_distance": Parameter(0.6 * units.nm, "Parameter scan"),
        "max_expansion": Parameter(3, "Koch, A. L., & Woeste, S. (1992)."),
    },
    "concs": {
        "initial_pump": Parameter(
            6.7e-4 * units.mM,
            "Simulation c33d8283af0bed4a6a598774ac5d8aec19d169bf",
        ),
        "initial_hydrolase": Parameter(
            7.1e-4 * units.mM,
            "Simulation c33d8283af0bed4a6a598774ac5d8aec19d169bf",
        ),
        # AcrAB-TolC: TRANS-CPLX-201[m]
        "average_pump": Parameter(
            0.0007157472280240362 * units.mM,
            "Simulation 0a2cd6816d36d408470445ff654371f07cd3f9f8",
        ),
        # Beta-lactamase: EG10040-MONOMER[p]
        "average_hydrolase": Parameter(
            0.0008351114340588106 * units.mM,
            "Simulation 0a2cd6816d36d408470445ff654371f07cd3f9f8",
        ),
    },
    "counts": {
        # ompC: CPLX0-7533[o]
        # ompF: CPLX0-7534[o]
        "initial_ompf": Parameter(
            18975 * units.count,
            "Simulation c33d8283af0bed4a6a598774ac5d8aec19d169bf",
        ),
        "initial_ompc": Parameter(
            5810 * units.count,
            "Simulation c33d8283af0bed4a6a598774ac5d8aec19d169bf",
        ),
        "average_ompf": Parameter(
            26303.986572174563 * units.count,
            "Simulation 0a2cd6816d36d408470445ff654371f07cd3f9f8",
        ),
        "average_ompc": Parameter(
            7288.019395747855 * units.count,
            "Simulation 0a2cd6816d36d408470445ff654371f07cd3f9f8",
        ),
    },
    "lysis_initiation": {"mean_lysis_time": Parameter(192.8, "Wong and Amir 2019")},
    "avogadro": constants.N_A / units.mol,
    "outer_potential": Parameter(
        -0.0215 * units.volt,
        "Sen, Hellman, and Nikaido (1988) p. 1184",
        latex_source="sen_porin_1988",
        note="Donnan potential",
    ),
    "inner_potential": Parameter(
        0 * units.volt,
        note="Assume no bias for diffusion from periplasm to cytoplasm",
        # -0.1185 * units.volt,
        # "Felle et al. (1980) p. 3587",
        # note="Assume -140 mV resting potential"
    ),
    "faraday_constant": Parameter(
        constants.value("Faraday constant") * units.C / units.mol
    ),
    "gas_constant": Parameter(
        constants.R * units.J / units.mol / units.K,
    ),
    "temperature": Parameter(
        298 * units.K,
    ),
}

DERIVATION_RULES = {
    ("shape", "initial_periplasm_volume"): lambda params: Parameter(
        (
            params.get(("shape", "initial_cell_volume"))
            * params.get(("shape", "periplasm_fraction"))
        ),
    ),
    ("shape", "initial_cytoplasm_volume"): lambda params: Parameter(
        (
            params.get(("shape", "initial_cell_volume"))
            * (1 - params.get(("shape", "periplasm_fraction")))
        ),
    ),
    ("shape", "average_periplasm_volume"): lambda params: Parameter(
        (
            params.get(("shape", "average_cell_volume"))
            * params.get(("shape", "periplasm_fraction"))
        ),
    ),
    ("shape", "average_cytoplasm_volume"): lambda params: Parameter(
        (
            params.get(("shape", "average_cell_volume"))
            * (1 - params.get(("shape", "periplasm_fraction")))
        ),
    ),
    ("shape", "initial_inner_area"): lambda params: Parameter(
        (
            np.cbrt((1 - params.get(("shape", "periplasm_fraction")))) ** 2
            * params.get(("shape", "initial_outer_area"))
        ),
    ),
    ("ampicillin", "efflux", "kcat"): lambda params: Parameter(
        (
            params.get(("ampicillin", "efflux", "vmax"))
            / params.get(("concs", "average_pump"))
            * params.get(("shape", "average_dry_mass"))
            / params.get(("shape", "average_periplasm_volume"))
        )
    ),
    ("ampicillin", "per_porin_permeability", "outer", "ompf"): lambda params: Parameter(
        (
            params.get(("ampicillin", "permeability", "outer"))
            / (
                params.get(("counts", "average_ompf"))
                / params.get(("shape", "average_outer_area"))
            )
        ),
    ),
    ("cephaloridine", "efflux", "kcat"): lambda params: Parameter(
        (
            params.get(("cephaloridine", "efflux", "vmax"))
            / params.get(("concs", "average_pump"))
            * params.get(("shape", "average_dry_mass"))
            / params.get(("shape", "average_periplasm_volume"))
        )
    ),
    (
        "cephaloridine",
        "per_porin_permeability",
        "outer",
        "ompf",
    ): lambda params: Parameter(
        (
            params.get(
                ("cephaloridine", "porin_specific_permeability", "outer", "ompf")
            )
            / (
                params.get(("counts", "average_ompf"))
                / params.get(("shape", "average_outer_area"))
            )
        ),
    ),
    (
        "cephaloridine",
        "per_porin_permeability",
        "outer",
        "ompc",
    ): lambda params: Parameter(
        (
            params.get(
                ("cephaloridine", "porin_specific_permeability", "outer", "ompc")
            )
            / (
                params.get(("counts", "average_ompc"))
                / params.get(("shape", "average_outer_area"))
            )
        ),
    ),
    (
        "cephaloridine",
        "pbp_binding",
        "K_A (micromolar)",
        "PBP1A",
    ): lambda params: Parameter(
        (
            params.get(("cephaloridine", "pbp_binding", "K_A", "PBP1A"))
            / params.get(("cephaloridine", "molar_mass"))
        ).to("micromolar")
    ),
    (
        "cephaloridine",
        "pbp_binding",
        "K_A (micromolar)",
        "PBP1B",
    ): lambda params: Parameter(
        (
            params.get(("cephaloridine", "pbp_binding", "K_A", "PBP1B"))
            / params.get(("cephaloridine", "molar_mass"))
        ).to("micromolar")
    ),
    ("tetracycline", "efflux", "kcat"): lambda params: Parameter(
        (
            params.get(("tetracycline", "efflux", "vmax"))
            / params.get(("concs", "average_pump"))
            * params.get(("shape", "average_dry_mass"))
            / params.get(("shape", "average_cytoplasm_volume"))
        )
    ),
    ("tetracycline", "permeability", "outer", "ompf"): lambda params: Parameter(
        (
            params.get(("tetracycline", "permeability", "outer_with_porins"))
            - params.get(("tetracycline", "permeability", "outer_without_porins"))
        ),
    ),
    (
        "tetracycline",
        "per_porin_permeability",
        "outer",
        "ompf",
    ): lambda params: Parameter(
        (
            params.get(("tetracycline", "permeability", "outer", "ompf"))
            / (
                params.get(("counts", "average_ompf"))
                / params.get(("shape", "average_outer_area"))
            )
        ),
    ),
    ("cell_wall", "strand_term_p"): lambda params: Parameter(
        fit_strand_term_p(
            pd.read_csv(
                params.get(("cell_wall", "strand_length_data")),
            ),
            params.get(("cell_wall", "upper_mean")),
        )
    ),
}

param_store = ParameterStore(PARAMETER_DICT, DERIVATION_RULES)


TableRow = namedtuple("TableRow", ["param_name", "param", "units", "description"])
TABLES = {
    "Trans-membrane diffusion parameters": (
        TableRow(
            "P_{amp,outer}",
            param_store.get_parameter(("ampicillin", "permeability", "outer")),
            "cm/sec",
            "Outer membrane permeability to ampicillin.",
        ),
        TableRow(
            "P_{tet,outer,np}",
            param_store.get_parameter(
                ("tetracycline", "permeability", "outer_without_porins")
            ),
            "cm/sec",
            "Outer membrane permeability to tetracycline without porins.",
        ),
        TableRow(
            "P_{tet,outer,p}",
            param_store.get_parameter(
                ("tetracycline", "permeability", "outer_with_porins")
            ),
            "cm/sec",
            "Outer membrane permeability to tetracycline with porins.",
        ),
        TableRow(
            "P_{tet,inner}",
            param_store.get_parameter(("tetracycline", "permeability", "inner")),
            "cm/sec",
            "Inner membrane permeability to tetracycline.",
        ),
        TableRow(
            "\\Delta \\phi",
            param_store.get_parameter(("outer_potential",)),
            "mV",
            "Donnan potential across the outer membrane.",
        ),
    ),
    "Antibiotic export parameters": (
        TableRow(
            "[E]_0",
            param_store.get_parameter(("concs", "initial_pump")),
            "mM",
            "Initial concentration of AcrAB-TolC.",
        ),
        TableRow(
            "m_0",
            param_store.get_parameter(("shape", "initial_cell_mass")),
            "fg",
            "Initial mass of the cell.",
        ),
        TableRow(
            "V_{p,0}",
            param_store.get_parameter(("shape", "initial_periplasm_volume")),
            "fL",
            "Initial volume of the periplasm.",
        ),
        TableRow(
            "v_{max,e,amp}",
            param_store.get_parameter(("ampicillin", "efflux", "vmax")),
            "nmol/mg/sec",
            "Maximum rate of ampicillin export.",
        ),
        TableRow(
            "K_{M,e,amp}",
            param_store.get_parameter(("ampicillin", "efflux", "km")),
            "mM",
            "Michaelis constant for ampicillin export.",
        ),
        TableRow(
            "n_{e,amp}",
            param_store.get_parameter(("ampicillin", "efflux", "n")),
            "",
            "Hill coefficient for ampicillin export.",
        ),
        TableRow(
            "v_{max,e,tet}",
            param_store.get_parameter(("tetracycline", "efflux", "vmax")),
            "nmol/mg/sec",
            "Maximum rate of tetracycline export.",
        ),
        TableRow(
            "K_{M,e,tet}",
            param_store.get_parameter(("tetracycline", "efflux", "km")),
            "mM",
            "Michaelis constant for tetracycline export.",
        ),
        TableRow(
            "n_{e,tet}",
            param_store.get_parameter(("tetracycline", "efflux", "n")),
            "",
            "Hill coefficient for tetracycline export.",
        ),
    ),
    "Antibiotic hydrolysis parameters": (
        TableRow(
            "k_{cat,h,amp}",
            param_store.get_parameter(("ampicillin", "hydrolysis", "kcat")),
            "1/s",
            "Rate constant for ampicillin hydrolysis.",
        ),
        TableRow(
            "K_{M,h,amp}",
            param_store.get_parameter(("ampicillin", "hydrolysis", "km")),
            "mM",
            "Michaelis const. for ampicillin hydrolysis.",
        ),
        TableRow(
            "n_{h,amp}",
            param_store.get_parameter(("ampicillin", "hydrolysis", "n")),
            "",
            "Hill coefficient for ampicillin hydrolysis.",
        ),
    ),
}


def test_simple():
    param_dict = {
        "a": {
            "b": Parameter(
                1,
            ),
        },
    }
    param_store = ParameterStore(param_dict)
    assert param_store.get(("a", "b")) == 1


def test_canonicalize():
    param_dict = {
        "a": {
            "b": Parameter(1, canonicalize=lambda x: x * 2),
        },
    }
    param_store = ParameterStore(param_dict)
    assert param_store.get(("a", "b")) == 2


def test_derivation():
    param_dict = {
        "a": {
            "b": Parameter(
                1,
            ),
        },
    }
    derivation_rules = {
        ("a", "c"): lambda params: Parameter(
            params.get(("a", "b")) * 2,
            canonicalize=lambda x: x * 3,
        )
    }
    param_store = ParameterStore(param_dict, derivation_rules)
    assert param_store.get(("a", "c")) == 6


def main():
    for table_name, rows in TABLES.items():
        print(
            r"\begin{tabular}{|"
            r"p{0.15\columnwidth}"
            r"p{0.3\columnwidth}"
            r"p{0.5\columnwidth}"
            r"|}"
        )
        print("\t" + r"\hline")
        print("\t" + r"\multicolumn{3}{|c|}{%s} \\" % table_name)
        print("\t" + r"\hline")
        print("\t" + r"Parameter & Value & Description \\")
        print("\t" + r"\hline")
        for row in rows:
            description = row.description
            if description.endswith("."):
                description = description[:-1]
            if row.param.latex_source:
                description += r" \cite{%s}" % row.param.latex_source
            description += "."
            value_str = "{:.2e}".format(row.param.value.to(row.units).magnitude)
            if "e" in value_str:
                base, exponent = value_str.split("e")
                exponent = int(exponent)
                if "." in base:
                    base = base.rstrip("0").rstrip(".")
                if exponent:
                    value_str = "%s \\times 10^{%s}" % (base, exponent)
                else:
                    value_str = base
            value_str = f"${value_str}$"
            row_str = "$%s$ & %s %s & %s" % (
                row.param_name,
                value_str,
                row.units,
                description,
            )
            print("\t" + r"%s\\" % row_str)
        print("\t" + r"\hline")
        print(r"\end{tabular}")
        print()


if __name__ == "__main__":
    main()
