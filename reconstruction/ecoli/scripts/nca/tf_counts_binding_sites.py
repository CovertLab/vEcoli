import argparse
import csv
import os
import re
import sys
import time
from typing import Dict, List, Optional, Set, Tuple, cast
import math

import matplotlib.pyplot as plt
import numpy as np
import scipy
from rnaseq_utils import *

# Midpoint of oriC, from dna_sites.tsv flat file
ORIC_SITE = (3925744+3925975)/2
# Length of K12 genome
GENOME_LENGTH = 4608319
# Replisome elongation rate, from parameters.tsv flat file
REPL_RATE = 967
C_PERIOD = GENOME_LENGTH / (REPL_RATE*2*60)
# D_period, from parameters.tsv flate file
D_PERIOD = 20



def get_promoter_coords():
    filename = '../../../../../../devViv/vivarium-ecoli/reconstruction/ecoli/scripts/nca/All-promoters-of-E.-coli-K-12-substr.-MG1655.txt'
    promoter_names = []
    coords = []
    with open(os.path.join(BASE_DIR, filename), 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            promoter_names.append(line.split()[0])
            if len(line.split()) > 1:
                coords.append(int(line.split()[1]))
            else:
                coords.append(0)

    return {p:c for p, c in zip(promoter_names, coords)}

def get_binding_sites():
    def tf_name_switch(name):
        if name[:11] == 'dna-binding':
            return name.split(' ')[-1]
        elif name.split(' ')[0] == 'phosphorylated':
            return name.split(' ')[-1]
        elif name == 'nrdr transcriptional repressor':
            return 'nrdr'
        elif name == 'ntrc phosphorylated dimer':
            return 'ntrc'
        elif name == 'h-ns':
            return 'hns'
        elif '-' in name:
            return name.split('-')[0]
        return name

    filename = "../../../../../../devViv/vivarium-ecoli/reconstruction/ecoli/scripts/nca/All-transcription-factor-binding-sites-of-E.-coli-K-12-substr.-MG1655.txt"
    names = []
    coords = []
    regs = []
    tf_names = []
    promoter_names = []
    promoter_coord_dict = get_promoter_coords()

    with open(os.path.join(BASE_DIR, filename), 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            name, coord, reg = line.split('\t')
            if reg[-1] == '\n':
                reg = reg[:-1]

            name_split = name.split(' DNA-binding-site')

            #name_split = name.split(' ')
            tf_name = name_split[0]
            promoter_name = name_split[-1].split(' ')[-1]
            tf_names.append(tf_name_switch(tf_name.lower()))
            promoter_names.append(promoter_name)
            names.append(name)
            if coord != '':
                if coord[-2:] == 'd0':
                    coords.append(float(coord[:-2]))
                else:
                    coords.append(float(coord))
            else:
                promoter = name.split(' ')[-1]
                if promoter in promoter_coord_dict:
                    coords.append(promoter_coord_dict[promoter])
                else:
                    coords.append(0) # TODO: do something about these?
            regs.append(reg)

    def coords_to_gene_dosage(coords, doubling_time):
        coords = np.array(coords)
        right_side = (np.abs(coords - ORIC_SITE) < GENOME_LENGTH/2)
        relative_coords = np.zeros_like(coords)
        relative_coords[right_side] = np.abs(coords[right_side] - ORIC_SITE)
        relative_coords[~right_side] = GENOME_LENGTH - np.abs(
            coords[~right_side] - ORIC_SITE)
        relative_coords = relative_coords * 2 / GENOME_LENGTH
        gene_dosages = 2 ** (
                    ((1 - relative_coords) * C_PERIOD + D_PERIOD) / doubling_time)

        return gene_dosages

    gene_dosages = coords_to_gene_dosage(coords, 60)

    tf_names = np.array(tf_names)
    unique_tfs = np.unique(tf_names)
    tf_counts = [np.sum(gene_dosages[np.where(tf_names == x)[0]]) for x in unique_tfs]

    return unique_tfs, tf_counts
    # At a doubling time of 40, TF counts are present in some amount. If the TF
    # is supposed to be actively binding DNA, then probably there should be
    # more than the amount of binding sites that there are?
    # If the TF is not supposed to be binding DNA, but what if the ligand comes,
    # then there should be enough so that it is able to bind DNA and respond
    # quickly, unless that TF does like a positive feedback thing on itself
    # but that'll also take more time. Ofc also if it is like a growth-rate
    # responsive molecule might be different. But so in general, if TF is much
    # less than # of binding sites that's interesting, and also perhaps if it's
    # much more?
    # There's the question of, does one TF monomer bind to one binding site?
    # Or does a multimer of TF bind to one binding site? Or does a multimer
    # of TF bind to multiple binding sites? If some binding sites are mutually
    # exclusive, just count them as one, though not sure if that occurs at all.
    # TODO: get proteome counts, and do some comparisons?
    # TODO: around 300 of them don't have coords, but we could get their promoters
    # from promoter_names then try to get promoters that way?
    # Also, some promoters are in parantheses like (BS0smth).

def get_proteome_counts():
    proteome_file = os.path.join(BASE_DIR, '../../../../../../devViv/vivarium-ecoli/reconstruction/ecoli/scripts/nca/copied_schmidt2015_javier_table.tsv')
    gene_names = []
    glucose_counts = []
    with open(proteome_file, 'r') as f:
        csv_reader = csv.reader(f, delimiter='\t')
        next(csv_reader)
        for line in csv_reader:
            gene_names.append(line[1].lower())
            glucose_counts.append(float(line[2]))

    return gene_names, glucose_counts
#     # From proteome experimental dataset, get the counts under certain conditions.
#     #
#

def get_autoregulated_tfs():
    regulator_names = []
    autoregulate = []

    def detect_autoreg(tf, genes):
        for gene in genes:
            if tf in gene:
                if gene[0:3] == "+/-":
                    return 2
                if gene[0] == "+":
                    return 1
                if gene[0] == "-":
                    return -1
                return -2
        return 0

    with open(os.path.join(BASE_DIR, '../../../../../../devViv/vivarium-ecoli/reconstruction/ecoli/scripts/nca/ECOLI-regulatory-network.txt'), 'r') as f:
        current_regulator = ''
        for line in f.readlines():
            if line.startswith('#'):
                continue
            if line[0] != ' ':
                regulator = line.split()[0]
                if regulator[-1] == "*":
                    if regulator == 'rspR*':
                        regulator_names.append('ydfh')
                    else:
                        regulator_names.append(regulator[:-1].lower())
                current_regulator = regulator
            else:
                genes = line.split()
                autoregulate.append(detect_autoreg(current_regulator, genes))

    return {r: d for r, d in zip(regulator_names, autoregulate)}

def get_tf_direction():
    regulator_names = []
    direction = []

    def detect_direction(genes):
        # TODO: filter for only transcriptional regulation
        dirs = []
        for gene in genes:
            if gene[:3] == "+/-":
                dirs.append(2)
            elif gene[0] == "+":
                dirs.append(1)
            elif gene[0] == "-":
                dirs.append(0)
        dirs = np.unique(dirs)
        if np.array_equal(dirs, np.array([0])):
            return 0
        if np.array_equal(dirs, np.array([1])):
            return 1
        return 2

    with open(os.path.join(BASE_DIR, '../../../../../../devViv/vivarium-ecoli/reconstruction/ecoli/scripts/nca/ECOLI-regulatory-network.txt'), 'r') as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            if line[0] != ' ':
                regulator = line.split()[0]
                if regulator[-1] == "*":
                    if regulator == 'rspR*':
                        regulator_names.append('ydfh')
                    else:
                        regulator_names.append(regulator[:-1].lower())
            else:
                genes = line.split()
                direction.append(detect_direction(genes))

    return {r: d for r, d in zip(regulator_names, direction)}

def make_plot():
    tf_names, tf_site_counts = get_binding_sites()
    gene_names, glucose_counts = get_proteome_counts()
    gene_to_count = {g: c for g, c in zip(gene_names, glucose_counts)}
    tf_to_site_count = {t: c for t, c in zip(tf_names, tf_site_counts)}
    tf_in_proteome = []
    for gene in tf_names:
        tf_in_proteome.append((gene in gene_names))

    plot_tf_names = np.array(tf_names)[tf_in_proteome]
    plot_tf_site_counts = np.array(tf_site_counts)[tf_in_proteome]
    plot_glucose_counts = np.array([gene_to_count[x] for x in plot_tf_names])
    for i in np.where(plot_glucose_counts==0)[0]:
            plot_glucose_counts[i] = 0.1

    color_dict = {0: 'b', 1: 'r', -1: 'g', 2: 'y'}

    tf_autoreg = get_autoregulated_tfs()
    plot_tf_colors = []
    excluded_tfs = []
    for tf in plot_tf_names:
        if tf in tf_autoreg:
            plot_tf_colors.append(color_dict[tf_autoreg[tf]])
        else:
            plot_tf_colors.append('k')
            excluded_tfs.append(tf)
    plot_tf_colors = np.array(plot_tf_colors)

    direction_color_dict = {0: 'b', 1: 'r', 2: 'y'}
    tf_direction = get_tf_direction()
    plot_tf_dir_colors = [direction_color_dict[tf_direction[tf]] for tf in plot_tf_names]

    c_to_label = {'b': 'no autoreg.',
                   'r': 'positive autoreg.',
                   'g': 'negative autoreg.',
                   'y': 'pos. and neg. autoreg.'}
    fig, axs = plt.subplots(5, figsize=(5, 25))
    for c in np.unique(plot_tf_colors):
        is_c = (plot_tf_colors == c)
        axs[0].scatter(np.log2(plot_tf_site_counts[is_c]), np.log2(plot_glucose_counts[is_c]),
                       s=5, c=c, label=c_to_label[c])
    axs[0].legend()
    for i, txt in enumerate(plot_tf_names):
        if txt == 'purr':
            axs[0].annotate('PurR',
                            (np.log2(plot_tf_site_counts[i]), np.log2(plot_glucose_counts[i])),
                            size=10, c='g')

    axs[0].plot([0, 12], [0, 12])
    axs[0].set_title("TF counts vs genomic binding sites")
    axs[0].set_xlabel("log2(binding site counts), gene dosage adjusted")
    axs[0].set_ylabel("Counts in minimal media (1hr doubling time)")

    axs[1].scatter(np.log2(plot_tf_site_counts), np.log2(plot_glucose_counts),
                   s=5, c=plot_tf_dir_colors)
    for i, txt in enumerate(plot_tf_names):
        axs[1].annotate(txt, (np.log2(plot_tf_site_counts[i]),
        np.log2(plot_glucose_counts[i])), size=10)
    axs[1].set_title("Counts vs binding sites. Blue: repressor, Red: activator, Yellow: dual function")
    axs[1].set_xlabel("log2(binding site counts), gene dosage adjusted")
    axs[1].set_ylabel("Counts in minimal media (1hr doubling time)")
    axs[1].plot([0, 12], [0, 12])

    # axs[1].scatter(np.log2(plot_glucose_counts), np.log2(plot_tf_site_counts / plot_glucose_counts), s=0.5)
    # axs[1].set_title("Site-count ratio vs minimal media counts")
    # axs[2].scatter(np.log2(plot_tf_site_counts), np.log2(plot_tf_site_counts / plot_glucose_counts), s=0.5)
    # axs[2].set_title("Site-count ratio vs binding site counts")
    axs[2].hist(np.log2(plot_tf_site_counts))
    axs[2].set_title("TF site counts histogram")
    axs[2].set_xlabel("log2(TF binding site), gene dosage adjusted")
    axs[2].set_ylabel("Frequency")

    axs[3].hist(np.log2(plot_glucose_counts))
    axs[3].set_title("Protein counts histogram")
    axs[3].set_xlabel("log2(protein counts in minimal media), 1hr dt")
    axs[3].set_ylabel("Frequency")

    axs[4].hist(np.log2(plot_tf_site_counts / plot_glucose_counts))
    axs[4].set_title("Site:protein count ratio histogram")
    axs[4].set_xlabel("log2(Site count / protein count), 1hr dt, gene dosage adjusted")
    axs[4].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "TF_site_count"))
    plt.close('all')

    # TODO: whether the remaining 98 TFs are present in the proteome?
    # TODO: finish getting all coords maybe?
    # TODO: there's the factor of monomer:binding site ratio, which probably
    # TODO: do something about TF synonyms!
    # goes from 1/4 to 4.
    # for them. Then think about what to do about multimeric TF thing, and
    # multiple binding sites thing, and other things before making the actual
    # plot. And think about what'd you expect on the actual plot.
    # Maybe try plotting without accounting for gene dosage first as well?
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    make_plot()
    # So from the TF binding sites file, we're trying to get for each individual
    # TF, how many binding sites does it have and at what gene dosage.
    # So we just need which TF is the one acting in each binding site,
    # and also its coordinate (direclty given, or via the gene it's acting on).
    # And we also need which gene or TU the binding site is ahead of, because
    # that might be a better connection.
    # Or we could also want to say like how many transcription units does
    # it control, because like for example lacI repressor can bind to
    # four lacI sites at once??
# we just need which TF