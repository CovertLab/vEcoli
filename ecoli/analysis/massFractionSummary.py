from __future__ import absolute_import, division, print_function

import os

import numpy as np
from matplotlib import pyplot as plt

#from wholecell.io.tablereader import TableReader
#from wholecell.analysis.analysis_tools import exportFigure
#from models.ecoli.analysis import singleAnalysisPlot
from six.moves import zip

from ecoli.composites.ecoli_master import run_ecoli
from ecoli.analysis.tablereader import TableReader


COLORS_256 = [  # From colorbrewer2.org, qualitative 8-class set 1
    [228, 26, 28],
    [55, 126, 184],
    [77, 175, 74],
    [152, 78, 163],
    [255, 127, 0],
    [255, 255, 51],
    [166, 86, 40],
    [247, 129, 191]
]

COLORS = [
    [colorValue/255. for colorValue in color]
    for color in COLORS_256
]


class Plot(object):


    def do_plot(self, data):
        mass = TableReader("Mass", self.data)
        main_reader = TableReader("Main", self.data)

        cell = mass.readColumn("dryMass")
        protein = mass.readColumn("proteinMass")
        tRna = mass.readColumn("tRnaMass")
        rRna = mass.readColumn("rRnaMass")
        mRna = mass.readColumn("mRnaMass")
        dna = mass.readColumn("dnaMass")
        smallMolecules = mass.readColumn("smallMoleculeMass")

        #initialTime = main_reader.readAttribute("initialTime")
        t = (main_reader.readColumn("time", ) - initialTime) / 60.

        masses = np.vstack([
            protein,
            rRna,
            tRna,
            mRna,
            dna,
            smallMolecules,
        ]).T
        fractions = (masses / cell[:, None]).mean(axis=0)

        mass_labels = ["Protein", "rRNA", "tRNA", "mRNA", "DNA", "Small Mol.s"]
        legend = [
            '{} ({:.3f})'.format(label, fraction)
            for label, fraction in zip(mass_labels, fractions)
        ] + ['Total dry mass']

        plt.figure(figsize=(8.5, 11))
        plt.gca().set_prop_cycle('color', COLORS)

        plt.plot(t, masses / masses[0, :], linewidth=2)
        plt.plot(t, cell / cell[0], color='k', linestyle=':')

        plt.title(
            "Biomass components (average fraction of total dry mass in parentheses)")
        plt.xlabel("Time (min)")
        plt.ylabel("Mass (normalized by t = 0 min)")
        plt.legend(legend, loc="best")

        plt.tight_layout()
        #exportFigure(plt, plotOutDir, plotOutFileName, metadata)
        plt.close("all")

    def __init__(self, data):
        import ipdb; ipdb.set_trace()
        self.data = data
        self.do_plot(self.data)


def test_plot():

    data = run_ecoli(total_time=4)
    import ipdb; ipdb.set_trace()
    Plot(data)


if __name__ == "__main__":
    #import ipdb; ipdb.set_trace()
    test_plot()
