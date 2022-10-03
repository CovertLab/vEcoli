import pandas as pd
from collections import namedtuple
from scipy import constants

from vivarium.library.topology import get_in, assoc_path
from vivarium.library.units import units

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
                latex_source="kojima2013permeation",
            ),
        },
        "mic": Parameter(
            2 * units.micrograms / units.mL,
            "Mazzariol, Cornaglia, and Nikaido (2000)",
            lambda x: (
                # Divide by molecular weight from PubChem.
                x
                / (349.4 * units.g / units.mol)
            ).to(units.mM),
        ),
        "efflux": {
            "vmax": Parameter(
                0.069 * units.nmol / units.mg / units.sec,
                "Kojima and Nikaido (2013)",
                latex_source="kojima2013permeation",
            ),
            "km": Parameter(
                2.16e-3 * units.mM,
                "Kojima and Nikaido (2013)",
                latex_source="kojima2013permeation",
            ),
            "n": Parameter(
                1.9 * units.count,
                "Kojima and Nikaido (2013)",
                latex_source="kojima2013permeation",
            ),
        },
        "hydrolysis": {
            "kcat": Parameter(
                6.5 / units.sec,
                "Mazzariol, Cornaglia, and Nikaido (2000)",
                latex_source="mazzariol2000contributions",
            ),
            "km": Parameter(
                0.9e-3 * units.mM,
                "Mazzariol, Cornaglia, and Nikaido (2000)",
                latex_source="mazzariol2000contributions",
            ),
            "n": Parameter(
                1 * units.count,
                "Mazzariol, Cornaglia, and Nikaido (2000)",
                latex_source="mazzariol2000contributions",
            ),
        },
        "pbp_binding": {
            "K_A": {
                "PBP1A": Parameter(1.4 * (units.ug / units.mL), "Curtis et al. 1979"),
                "PBP1B": Parameter(3.9 * (units.ug / units.mL), "Curtis et al. 1979"),
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
                latex_source="nikaido1983porin",
            ),
        },
        # Cell-wide permeability with only one porin present.
        "porin_specific_permeability": {
            "outer": {
                "ompf": Parameter(
                    52.6e-5 * units.cm / units.sec,
                    "Nikaido, Rosenberg, and Foulds (1983)",
                    latex_source="nikaido1983porin",
                ),
                "ompc": Parameter(
                    4.5e-5 * units.cm / units.sec,
                    "Nikaido, Rosenberg, and Foulds (1983)",
                    latex_source="nikaido1983porin",
                ),
            },
        },
        "mic": Parameter(
            0.5 * units.micrograms / units.mL,
            "Rolinson (1980)",
            lambda x: (
                # Divide by molecular weight from PubChem.
                x
                / (415.5 * units.g / units.mol)
            ).to(units.mM),
        ),
        "efflux": {
            "vmax": Parameter(
                1.82 * units.nmol / units.mg / units.sec,
                "Nagano and Nikaido (2009)",
                latex_source="nagano2009kinetic",
            ),
            "km": Parameter(
                0.288 * units.mM,
                "Nagano and Nikaido (2009)",
                latex_source="nagano2009kinetic",
            ),
            "n": Parameter(
                1.75 * units.count,
                "Nagano and Nikaido (2009)",
                latex_source="nagano2009kinetic",
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
                latex_source="thanassi1995role",
            ),
            "outer_with_porins": Parameter(
                1e-5 * units.cm / units.sec,
                "Thanassi, Suh, and Nikaido (1995) p. 1005",
                latex_source="thanassi1995role",
            ),
            "inner": Parameter(
                3e-6 * units.cm / units.sec,
                "Thanassi, Suh, and Nikaido (1995) p. 1004",
                latex_source="thanassi1995role",
            ),
            # 'accumulation_factor': Parameter(
            #     8.4,
            #     """15- to 17-fold gross accumulation [1, 2]
            #     reduced by 37% to account for bound fraction [2]
            #     then reduced by 17% to account for higher Mg2+ in medium [3]
            #     [1] https://doi.org/10.1128/jb.177.4.998-1007.1995
            #     [2] https://doi.org/10.1007/BF00408069
            #     [3] https://doi.org/10.1128/AAC.35.1.53
            #     Current tetracycline accumulation models do not match
            #     experimental accumulation data. This factor abstracts
            #     away the mechanism of accumulation with the goal of
            #     achieving an accurate steady state tetracycline conc."""
            # ),
        },
        "charge": Parameter(1 * units.count),
        "efflux": {
            "vmax": Parameter(
                0.2 * units.nmol / units.mg / units.min,
                "Thanassi, Suh, and Nikaido (1995) p. 1004",
                latex_source="thanassi1995role",
            ),
            "km": Parameter(
                200 * units.uM,
                "Thanassi, Suh, and Nikaido (1995) p. 1004",
                latex_source="thanassi1995role",
            ),
            "n": Parameter(
                1 * units.count,
            ),
        },
        "mic": Parameter(
            2.5 * units.micromolar,
            "Thanassi, Suh, and Nikaido (1995) p. 1005",
            latex_source="thanassi1995role",
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
        "initial_area": Parameter(
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
        "average_area": Parameter(
            6.227824991612169 * units.um**2,
            "Simulation 0a2cd6816d36d408470445ff654371f07cd3f9f8",
        ),
    },
    "cell_wall": {
        "strand_length_data": Parameter(
            "data/cell_wall/murein_strand_length_distribution.csv",
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
        "inter_strand_distance": Parameter(1.2 * units.nm, "Parameter scan"),
        "max_expansion": Parameter(3, "Koch, A. L., & Woeste, S. (1992)."),
        "peptidoglycan_unit_area": Parameter(
            2.5 * units.nm**2,
            "Wientjes, F. B., Woldringh, C. L., & Nanninga, N. (1991).",
        ),
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
    'lysis_initiation': {
        'mean_lysis_time': Parameter(
            192.8,
            "Wong and Amir 2019"
        )
    },
    "avogadro": constants.N_A / units.mol,
    "donnan_potential": Parameter(
        -0.0215 * units.volt,
        "Sen, Hellman, and Nikaido (1988) p. 1184",
        latex_source="sen1988porin",
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
                / params.get(("shape", "average_area"))
            )
        ),
    ),
    (
        "ampicillin",
        "pbp_binding",
        "K_A (micromolar)",
        "PBP1A",
    ): lambda params: Parameter(
        (
            params.get(("ampicillin", "pbp_binding", "K_A", "PBP1A"))
            / params.get(("ampicillin", "molar_mass"))
        ).to("micromolar")
    ),
    (
        "ampicillin",
        "pbp_binding",
        "K_A (micromolar)",
        "PBP1B",
    ): lambda params: Parameter(
        (
            params.get(("ampicillin", "pbp_binding", "K_A", "PBP1B"))
            / params.get(("ampicillin", "molar_mass"))
        ).to("micromolar")
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
                / params.get(("shape", "average_area"))
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
                / params.get(("shape", "average_area"))
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
                / params.get(("shape", "average_area"))
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
            "P_{cep,outer}",
            param_store.get_parameter(("cephaloridine", "permeability", "outer")),
            "cm/sec",
            "Outer membrane permeability to cephaloridine.",
        ),
        TableRow(
            "P_{amp,outer}",
            param_store.get_parameter(("ampicillin", "permeability", "outer")),
            "cm/sec",
            "Outer membrane permeability to ampicillin.",
        ),
        TableRow(
            "P_{tet,outer}",
            param_store.get_parameter(
                ("tetracycline", "permeability", "outer_without_porins")
            ),
            "cm/sec",
            "Outer membrane permeability to tetracycline without porins.",
        ),
        TableRow(
            "P_{tet,inner}",
            param_store.get_parameter(("tetracycline", "permeability", "inner")),
            "cm/sec",
            "Inner membrane permeability to tetracycline.",
        ),
        TableRow(
            "\Delta \phi",
            param_store.get_parameter(("donnan_potential",)),
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
            "v_{max,e,cep}",
            param_store.get_parameter(("cephaloridine", "efflux", "vmax")),
            "nmol/mg/sec",
            "Maximum rate of cephaloridine export.",
        ),
        TableRow(
            "K_{M,e,cep}",
            param_store.get_parameter(("cephaloridine", "efflux", "km")),
            "mM",
            "Michaelis constant for cephaloridine export.",
        ),
        TableRow(
            "n_{e,cep}",
            param_store.get_parameter(("cephaloridine", "efflux", "n")),
            "",
            "Hill coefficient for cephaloridine export.",
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
            "Michaelis constant for ampicillin hydrolysis.",
        ),
        TableRow(
            "n_{h,amp}",
            param_store.get_parameter(("ampicillin", "hydrolysis", "n")),
            "",
            "Hill coefficient for ampicillin hydrolysis.",
        ),
        TableRow(
            "k_{cat,h,cep}",
            param_store.get_parameter(("cephaloridine", "hydrolysis", "kcat")),
            "1/s",
            "Rate constant for cephaloridine hydrolysis.",
        ),
        TableRow(
            "K_{M,h,cep}",
            param_store.get_parameter(("cephaloridine", "hydrolysis", "km")),
            "mM",
            "Michaelis constant for cephaloridine hydrolysis.",
        ),
        TableRow(
            "n_{h,cep}",
            param_store.get_parameter(("cephaloridine", "hydrolysis", "n")),
            "",
            "Hill coefficient for cephaloridine hydrolysis.",
        ),
    ),
}


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
                exponent = exponent.strip("+-0")
                base = base.strip("0")
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
