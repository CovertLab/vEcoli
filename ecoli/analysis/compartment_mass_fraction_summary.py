import os
import numpy as np
from matplotlib import pyplot as plt
from six.moves import zip
from ecoli.composites.ecoli_master import run_ecoli
from ecoli.analysis.tablereader import TableReader

COLORS_256 = [  # From colorbrewer2.org, qualitative 8-class set 1
    [166, 206, 227],
    [31, 120, 180],
    [178, 223, 138],
    [255, 222, 0],
    [251, 154, 153],
    [227, 26, 28],
    [253, 191, 111],
    [255, 127, 0],
    [202, 178, 214],
    [106, 61, 154],
    [51, 160, 44]
]

COLORS = [
    [colorValue / 255. for colorValue in color]
    for color in COLORS_256
]


class Plot:

    def __init__(self, data):
        self.data = data
        self.do_plot(self.data)

    def do_plot(self, data):
        mass = TableReader("Mass", data)
        main_reader = TableReader("Main", data)

        cell = mass.readColumn("cellMass")
        projection = mass.readColumn("projection_mass")
        cytosol = mass.readColumn("cytosol_mass")
        extracellular = mass.readColumn("extracellular_mass")
        membrane = mass.readColumn("membrane_mass")
        outer_membrane = mass.readColumn("outer_membrane_mass")
        periplasm = mass.readColumn("periplasm_mass")
        pilus = mass.readColumn("pilus_mass")
        inner_membrane = mass.readColumn("inner_membrane_mass")
        flagellum = mass.readColumn("flagellum_mass")

        time_vals = main_reader.readColumn('time')
        initialTime = time_vals[0]
        t = (main_reader.readColumn("time") - initialTime) / 60.

        masses = np.vstack([
            projection,
            cytosol,
            extracellular,
            membrane,
            outer_membrane,
            periplasm,
            pilus,
            inner_membrane,
            flagellum,
        ]).T
        fractions = (masses / cell[:, None]).mean(axis=0)

        mass_labels = ["Projection", "Cytosol", "Extracellular", "Membrane",
                       "Outer Membrane", "Periplasm", "Pilus", "Inner Membrane",
                       "Flagellum"]
        legend = [
                     '{} ({:.3e})'.format(label, fraction)
                     for label, fraction in zip(mass_labels, fractions)
                 ] + ['Total cell mass']

        plt.figure(figsize=(8.5, 11))
        plt.gca().set_prop_cycle('color', COLORS)

        plt.plot(t, masses / masses[0, :], linewidth=2)
        plt.plot(t, cell / cell[0], color='k', linestyle=':')

        plt.title("Cell mass by compartment (average fraction of total cell mass in parentheses)")
        plt.xlabel("Time (min)")
        plt.ylabel("Mass (normalized by t = 0 min)")
        plt.legend(legend, loc="best")

        fig = plt.figure(figsize=(8.5, 11))
        plt.gca().set_prop_cycle('color', COLORS)

        plt.plot(t, masses / masses[0, :], linewidth=2)
        plt.plot(t, cell / cell[0], color='k', linestyle=':')

        plt.title(
            "Biomass components (average fraction of total dry mass in parentheses)")
        plt.xlabel("Time (min)")
        plt.ylabel("Mass (normalized by t = 0 min)")
        plt.legend(legend, loc="best")

        # TODO -- this can go into a general method in an analysis base class
        plt.tight_layout()
        out_dir = 'out/analysis/'
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_dir + 'compartmentMassFractionSummary.png')
        plt.close("all")
        return fig


def run_plot():
    data = run_ecoli(total_time=10)
    Plot(data)


# python ecoli/analysis/compartment_mass_fraction_summary.py
if __name__ == "__main__":
    run_plot()
