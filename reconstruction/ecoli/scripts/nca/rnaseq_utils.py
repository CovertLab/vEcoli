import os

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import csv
import scipy.cluster as cluster
import scipy.spatial as spatial
from explore_rnaseq import OUTPUT_DIR, BASE_DIR

# interesting notes: ada does +/- autoregulation, weird, ask Markus?
# araC also does some +/- regulation
# bglG does attenuation on its 3 genes (postiive, so when bglG active, it turns on its own)
# bolA has some +/- stuff (it says it binds, but doesn't say what direction)
# chbR has +/- tho it seems legit (maybe coming from a ligand?)
# dicA does positive autoregulation
# dksA to rpoS translational, might have some other ones that are translational
# fecI is a sigma factor
# gadE, gadX are positively autoregulated
# hipA, hipB affects relA, so affects ppGpp, also is a mech for persister cells, they r toxin/antitoxin
# hprR and cusR bind the same dna sequence & cooperate, one is h202 one is cu, interesting
# hyfR is a sigma54-dependent transcription regulator
# idnR has + autoregulation

TRANSL_REGULATORS = ['accA', 'accD', #accA, accD translationally regulate accA/D, while FadR transcriptionally regulates it.
# FadR also regulates accB/C, and accB transcriptionally regulates accB/C. Any logic for the difference?
    'acnB', # aconitase, it translationally upregulates itself
    'deaD', #+ldtD  +mntR*  +sdiA*  +uvrY*
    'hfq', # p global, also maybe shud include cuz affects srnas? idk ask markus?
    ]
SRNAS = ['agrB', 'ameF', 'arcZ', # affects rpoS (sigma stress), and arcB 2CS stuff
        'arrS', # does gadE (acid-resistance)
        'asflhD', # does flhD
        'azuR', # does gadE and cadA
        'C0293', # does fadR, hcaR, but regulators
        'chiX',
        'cpxP', # DOES NOT affect mRNA stability or transcription (so same level?)
        'cyaR', # promotes degradation of a few genes
        'dicF', # ftsZ and some others, didn't mention degradation
        'dsrA',
        'fimR2',
        'fnrS',
        'gadY',
        'glmZ',
        'glnZ',
        'istR', # I think doesn't have mRNA stability, so might be same level?


    ]
DELETED_TFS = [
    'fis',
    'hns',
    'hupA', # ask whether should include this? it doesn't regulate that many genes
    'hupB',
    'ihfA',
    'ihfB', # ask whether to include this?


    ]


class RnaseqUtils(object):
    def __init__(self):
        pass

    def run_group_test(self, group_A, group_B, sample_table):
        score = self.group_similarity(group_A, group_B)
        _, p = self.group_bootstrap(score, len(group_A['gr']), len(group_B['gr']), sample_table)

        return score, p

    def group_similarity(self, group_A, group_B):
        total_score = 0
        for i in range(len(group_A['gr'])):
            sample = {}
            for key in group_A:
                sample[key] = group_A[key][i]
            score = self.single_similarity(sample, group_B)
            total_score += score

        return total_score

    def single_similarity(self, sample_values, group_values):
        score = 0
        for key in sample_values:
            if key == 'gr':
                score += np.sum(1 - np.abs(group_values[key] - sample_values[key])/2)
            else:
                score += np.sum(np.array([group_values == sample_values[key]], dtype=int))

        return score

    def group_bootstrap(self, score, group_A_size, group_B_size, all_samples, size=1000):
        scores = []
        for i in range(size):
            sample_idxs = np.random.choice(np.arange(len(all_samples['gr'])), group_A_size + group_B_size)
            group_A_idxs = sample_idxs[:group_A_size]
            group_B_idxs = sample_idxs[group_A_size:]

            group_A_sample = {}
            group_B_sample = {}
            for key in all_samples:
                group_A_sample[key] = all_samples[key][group_A_idxs]
                group_B_sample[key] = all_samples[key][group_B_idxs]

            scores.append(self.group_similarity(group_A_sample, group_B_sample))

        return scores, np.sum(np.array([np.array(scores) < score], dtype=int)) / len(scores)

    def single_bootstrap(self, score, group_size, all_samples, size=1000):
        scores = []
        for i in range(size):
            sample_idxs = np.random.choice(np.arange(len(all_samples['gr'])), group_size+1)
            group_idxs = sample_idxs[1:]
            single_idxs = sample_idxs[0]

            group_sample = {}
            single_sample = {}
            for key in all_samples:
                group_sample[key] = all_samples[key][group_idxs]
                single_sample[key] = all_samples[key][single_idxs]

            scores.append(self.single_similarity(single_sample, group_sample))

        return scores, np.sum(np.array([np.array(scores) < score], dtype=int)) / len(scores)


class PlotUtils(object):
    def __init__(self):
        pass

    def n_bins(self, array):
        return int(np.ceil(
            (np.max(array) - np.min(array)) / (2 * stats.iqr(array) / len(array) ** (1 / 3))))


class Rankings(object):
    def __init__(self):
        pass

    def rank_by_moments(self, expression, gene_names, plot_file, write_file):
        stds = []
        skews = []
        kurts = []

        for i, exp in enumerate(expression):
            stds.append(np.std(exp))
            skews.append(stats.skew(exp))
            kurts.append(stats.kurtosis(exp))

        stds = np.array(stds)
        skews = np.array(skews)
        kurts = np.array(kurts)

        fig, axs = plt.subplots(6, figsize=(5, 30))
        axs[0].scatter(stds, skews, s=0.5)
        axs[0].set_title("Stds vs. skews")
        axs[0].set_xlabel("Std")
        axs[0].set_ylabel("Skewness")

        axs[1].scatter(stds[kurts<50], kurts[kurts<50], s=0.5)
        axs[1].set_title("Stds vs. kurts")
        axs[1].set_xlabel("Std")
        axs[1].set_ylabel("Kurtosis")

        axs[2].scatter(skews[kurts<50], kurts[kurts<50], s=0.5)
        axs[2].set_title("Skews vs kurts")
        axs[2].set_xlabel("Skewness")
        axs[2].set_ylabel("Kurtosis")

        def n_bins(scores):
            return int(np.ceil((np.max(scores) - np.min(scores)) / (2 * stats.iqr(scores) / len(scores) ** (1 / 3))))

        axs[3].hist(stds, bins=n_bins(stds))
        axs[3].set_title("Stds")
        axs[3].set_xlim(0)

        axs[4].hist(skews, bins=n_bins(skews))
        axs[4].set_title("Skewness")

        axs[5].hist(kurts[kurts<50], bins=n_bins(kurts[kurts<50]))
        axs[5].set_title("Kurtosis")

        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close('all')

        rank_stds = np.argsort(stds)
        rank_skews = np.argsort(skews)
        rank_kurts = np.argsort(kurts)

        with open(write_file, 'w') as f:
            writer = csv.writer(f, delimiter='\t', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            header = ['genes_rank_std', "std_rank_std", "skew_rank_std", "kurt_rank_std",
                    "genes_rank_skew", "std_rank_skew", "skew_rank_skew", "kurt_rank_skew",
                    "genes_rank_kurt", "std_rank_kurt", "skew_rank_kurt", "kurt_rank_kurt"]
            writer.writerow(header)
            for i in range(len(stds)):
                writer.writerow([gene_names[rank_stds][i], stds[rank_stds][i], skews[rank_stds][i], kurts[rank_stds][i],
                                 gene_names[rank_skews][i], stds[rank_skews][i], skews[rank_skews][i], kurts[rank_skews][i],
                                 gene_names[rank_kurts][i], stds[rank_kurts][i], skews[rank_kurts][i], kurts[rank_kurts][i]])

        import ipdb;
        ipdb.set_trace()


    def rank_by_std_prob_plot(self, expression, gene_names, plot_file, write_file):
        def calc_prob_r(exp):
            order = np.argsort(exp)
            percentiles = stats.norm.cdf(exp[order], loc=np.mean(exp), scale=np.std(exp))
            return np.abs(stats.pearsonr(percentiles, np.arange(len(exp)))[0])

        stds = []
        abs_prob_rs = []
        for exp in expression:
            stds.append(np.std(exp))
            abs_prob_rs.append(calc_prob_r(exp))

        stds = np.array(stds)
        abs_prob_rs = np.array(abs_prob_rs)

        fig, axs = plt.subplots(3, figsize=(5, 15))
        axs[0].scatter(stds, abs_prob_rs, s=0.5)
        axs[0].set_title("Stds vs abs(pearson r)")

        def n_bins(scores):
            return int(np.ceil((np.max(scores) - np.min(scores)) / (2 * stats.iqr(scores) / len(scores) ** (1 / 3))))

        axs[1].hist(stds, bins=n_bins(stds))
        axs[1].set_title("Stds")
        axs[1].set_xlim(0)

        axs[2].hist(abs_prob_rs, bins=n_bins(abs_prob_rs))
        axs[2].set_title("Absolute prob r")

        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close('all')

        rank_stds = np.argsort(stds)
        rank_rs = np.argsort(abs_prob_rs)[::-1]
        with open(write_file, 'w') as f:
            writer = csv.writer(f, delimiter='\t', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            header = ['genes_rank_std', "std_rank_std", "r_rank_std",
                      'genes_rank_r, "std_rank_r', 'r_rank_r']
            writer.writerow(header)
            for i in range(len(stds)):
                writer.writerow([gene_names[rank_stds][i], stds[rank_stds][i], abs_prob_rs[rank_stds][i],
                                 gene_names[rank_rs][i], stds[rank_rs][i], abs_prob_rs[rank_rs][i]])

        import ipdb; ipdb.set_trace()

class ClassifyScore(object):
    def __init__(self):
        pass

    def get_accuracy(self, groupA, groupB, features):
        # TODO: make it work for growth rate
        groupA_features = np.array([groupA[f] for f in features]).T
        groupB_features = np.array([groupB[f] for f in features]).T
        unique_features = np.unique(np.concatenate((groupA_features, groupB_features), axis=0), axis=0)
        classify = {}

        correct = 0
        incorrect = 0
        for value in unique_features:
            x_A = np.sum(np.all(groupA_features == value, axis=1), dtype=int)
            x_B = np.sum(np.all(groupB_features == value, axis=1), dtype=int)
            correct += max(x_A, x_B)
            incorrect += min(x_A, x_B)
            value = tuple(value)
            classify[value] = (x_A, x_B)

        return correct / (correct + incorrect), classify

    def bootstrap(self, score, groupA_size, groupB_size, all_samples, features, size=5000):
        scores = []
        for i in range(size):
            sample_idxs = np.random.choice(np.arange(len(all_samples['gr'])), groupA_size + groupB_size)
            groupA_idxs = sample_idxs[:groupA_size]
            groupB_idxs = sample_idxs[groupA_size:]

            groupA_sample = {}
            groupB_sample = {}
            for key in features:
                groupA_sample[key] = all_samples[key][groupA_idxs]
                groupB_sample[key] = all_samples[key][groupB_idxs]
            scores.append(self.get_accuracy(groupA_sample, groupB_sample, features)[0])
        scores = np.array(scores)

        return scores, np.sum(np.array([scores > score], dtype=int)) / len(scores)

    def run_test(self, groupA, groupB, sample_table, features):
        score, classify = self.get_accuracy(groupA, groupB, features)
        _, p = self.bootstrap(score, len(groupA['gr']), len(groupB['gr']), sample_table, features)

        return score, p, classify

    def plot_samples(self, groupA, groupB, info):
        def convert(sample, info):
            converted = {}
            for k, v in sample.items():
                converted[k] = np.array(info[k])[np.array(v)]

            return converted

        groupA_info = convert(groupA, info)
        groupB_info = convert(groupB, info)

        # So we first want to order the data so that:
        #

class EcocycReg(object):
    def __init__(self, base_dir, exclude_sigma=False):
        # TODO: include regulation direction. So every time with an input gene,
        # we can get the updated name, then look at what genes ecocyc
        # says regulates it, and in what direction. :)
        self.base_dir = base_dir
        regulators = []
        regulatees = []
        sigma_factors = ['rpoD', 'rpoE', 'rpoN', 'rpoH', 'rpoS', 'fliA']
        with open(os.path.join(self.base_dir, '../../../../../../devViv/vivarium-ecoli/reconstruction/ecoli/scripts/nca/ECOLI-regulatory-network.txt'), 'r') as f:
            is_sigma = False
            for line in f.readlines():
                if line.startswith('#'):
                    continue
                if line[0] != ' ':
                    regulator = line.split()[0]
                    if regulator[-1] == "*":
                        if exclude_sigma:
                            if regulator[:-1] not in sigma_factors:
                                regulators.append(regulator[:-1])
                                is_sigma = False
                            else:
                                is_sigma = True
                        else:
                            regulators.append(regulator[:-1])
                else:
                    if exclude_sigma & is_sigma:
                        continue
                    genes = line.split()
                    format_genes = []

                    for gene in genes:
                        if gene[-1] == "*":
                            gene = gene[:-1]
                        if gene[:3] == '+/-':
                            format_genes.append((gene[3:], '+/-'))
                        elif gene[0] == '+':
                            format_genes.append((gene[1:], '+'))
                        elif gene[0] == '-':
                           format_genes.append((gene[1:], '-'))
                        else:
                            format_genes.append((gene, 'n/a'))
                    regulatees.append(format_genes)

        self.regulators = regulators
        self.regulation = {r: g for r, g in zip(regulators, regulatees)}
        self.all_regulated_genes = set()
        for g in regulatees:
            self.all_regulated_genes.update([v[0] for v in g])
        self.all_regulated_genes = list(self.all_regulated_genes)

        gene_to_regulators = {g: [] for g in self.all_regulated_genes}
        for g in self.all_regulated_genes:
            for k in self.regulation:
                reg_genes = [v[0] for v in self.regulation[k]]
                if g in reg_genes:
                    idx = reg_genes.index(g)
                    gene_to_regulators[g].append((k, self.regulation[k][idx][1]))
        self.gene_to_regulators = gene_to_regulators

        gene_synonyms = []
        all_genes = []
        synonyms_file = os.path.join(self.base_dir, "../../../../../../devViv/vivarium-ecoli/reconstruction/ecoli/scripts/nca/All-genes-of-E.-coli-K-12-substr.-MG1655.txt")
        with open(synonyms_file, 'r') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            for line in csv_reader:
                all_genes.append(line[0])
                if line[1] != '':
                    synonyms = line[1].split(" // ")
                    synonyms.append(line[0])
                    gene_synonyms.append(synonyms)

        self.all_genes = all_genes
        self.gene_synonyms = gene_synonyms
    # TODO: fix duplication, what if there r genes where the same javi's repo gene is a synonym for multiple genes in ecocyc?



    def find_autoregulated_genes(self):
        neg_autoreg = []
        pos_autoreg = []
        both_autoreg = []
        for regulator in self.regulation:
            for (gene, dir) in self.regulation[regulator]:
                if gene == regulator:
                    if dir == '-':
                        neg_autoreg.append(regulator)
                    elif dir == '+':
                        pos_autoreg.append(regulator)
                    elif dir == '+/-':
                        both_autoreg.append(regulator)

        return neg_autoreg, pos_autoreg, both_autoreg

    def classify_regulation(self, genes):
        raw_regulators = []
        processed_regulators = []
        for gene in genes:
            regulators = [x[0] for x in self.gene_to_regulators[gene]]
            raw_regulators.append(regulators)
            if 'rpoD' in regulators:
                processed = [x for x in regulators if x != 'rpoD']
                processed_regulators.append(processed)
            else:
                processed_regulators.append(regulators)

        return np.array([len(x) for x in processed_regulators]), processed_regulators

    def get_categories(self, gene_names):
        regulation = self.find_gene_regulation(gene_names)
        categories = {
            "no_reg": [],
            "rpoD_only": [],
            "rpoD_autoreg_only": [],
            "one_other": []
        }
        import ipdb; ipdb.set_trace()
        #for key, value in regulation.items():
            #if value
        # TODO: get enrichment of certin regulators?

    def update_synonyms(self, gene_names):
        updated_gene_names = []
        duplicated_genes = []

        for gene in gene_names:
            update_name = gene
            if gene not in self.all_genes:
                for syns in self.gene_synonyms:
                    if gene in syns:
                        update_name = syns[-1]
                        continue
            updated_gene_names.append(update_name)

        unique_names, counts = np.unique(updated_gene_names, return_counts=True)
        for u, c in zip(unique_names, counts):
            if c > 1:
                idxs = np.where(np.array(updated_gene_names)==u)[0]
                duplicated_genes.append([gene_names[i] for i in idxs])

        return updated_gene_names, duplicated_genes

    def find_missing_genes(self, gene_names):
        missing_genes = []
        for name in gene_names:
            if name not in self.all_genes:
                missing_genes.append(name)

        return missing_genes

    def ecocyc_to_ecomac_names(self, names_to_convert, ecomac_names):
        converted_names = []
        for gene in names_to_convert:
            if gene in ecomac_names:
                converted_names.append(gene)
                continue
            in_synonyms = False
            for synonyms in self.gene_synonyms:
                if gene in synonyms:
                    in_synonyms = True
                    in_ecomac = False
                    for name in synonyms:
                        if name in ecomac_names:
                            converted_names.append(name)
                            in_ecomac = True
                            break
                    if in_ecomac is not True:
                        converted_names.append("")
                    break
            if in_synonyms is not True:
                converted_names.append("")

        return converted_names

    def curate_gene_names(self, gene_names):
        updated_gene_names, duplicated_genes = self.update_synonyms(gene_names)
        missing_genes = self.find_missing_genes(updated_gene_names)
        return updated_gene_names, duplicated_genes, missing_genes

    def find_gene_regulation(self, gene_names):
        #TODO: fix duplicated gene names

        # NOTE: Gene_names should be ALL ecomac names
        updated_names, dupl_genes, missing_genes = self.curate_gene_names(gene_names)
        regulation = {}
        gene_to_update = {gene: update for gene, update in zip(gene_names, updated_names)}

        for gene in gene_names:
            update = gene_to_update[gene]
            if update in missing_genes:
                regulation[gene] = "N/A"
            else:
                if update in self.gene_to_regulators:
                    reg_info = self.gene_to_regulators[update]
                    names = [x[0] for x in reg_info]
                    converted_names = self.ecocyc_to_ecomac_names(
                        names, gene_names
                        )
                    regulation[gene] = [(x, y[1]) for x,y in zip(
                        converted_names, reg_info
                            )]
                else:
                    regulation[gene] = []

        return regulation


    def categorize_genes(self, gene_names):
        categories = []
        synonyms_file = os.path.join(self.base_dir, "../../../../../../devViv/vivarium-ecoli/reconstruction/ecoli/scripts/nca/All-genes-of-E.-coli-K-12-substr.-MG1655.txt")
        gene_synonyms = []
        with open(synonyms_file, 'r') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            next(csv_reader)
            for line in csv_reader:
                if line[1] != '':
                    synonyms = line[1].split(" // ")
                    synonyms.append(line[0])
                    gene_synonyms.append(synonyms)
                else:
                    gene_synonyms.append([line[0]])


        # So we have a file for all genes and their synonyms.
        # So our incoming gene names are mostly one of these synonyms, tho some aren't
        # And we want to compare to self.all_regulated_genes, which is probably
        # consisting of the gene names.
        # And so we can say, for each incoming gene name, get the synonyms,
        # and see if the gene is in all_regulated_genes
        for gene in gene_names:
            #for n in names:
            #    if n in self.all_regulated_genes:
            #        gene = n
            #        continue
            updated_name = ''
            for synonyms in gene_synonyms:
                if gene in synonyms:
                    updated_name = synonyms[-1]
                    continue

            if updated_name == '':
                categories.append(-1)
            else:
                if updated_name in self.all_regulated_genes:
                    if len(self.gene_to_regulators[updated_name]) == 1:
                        if updated_name in self.gene_to_regulators[updated_name]:
                            categories.append(1)
                        else:
                            categories.append(2)
                    else:
                        if updated_name in self.gene_to_regulators[updated_name]:
                            categories.append(3)
                        else:
                            categories.append(4)
                else:
                    categories.append(0)
        categories = np.array(categories)
        gene_categorized = {x: [gene_names[y] for y in np.where(categories==x)[0]] for x in [-1, 0, 1, 2, 3, 4]}

        # Autoregulated in our list: [['frmR', 'yaiN'], ['ybjK', 'rcdA'], ['dicA', 'ftsT'], ['yebK', 'hexR(P.a.)'], ['yfeC'], ['yhaJ'], ['selB', 'fdhA']]
        # Autoregulated from ecocyc: [['yfeC'], ['citR'], ['yhaJ'], ['nimR'], ['rcdA'], ['yebK'], ['frmR'], ['dicA'], ['selB']]
        # So overlapping are: all but citR, nimR, but those have other synonyms.
        # TODO: curate the 70 category -1, i.e. gene names that are listed in the repo but that are not present in ecocyc gene_name or synonym,
        # TODO: why are there 167 category 3 here and 165 from ecocyc's? maybe there are like 2 genes here that are synonyms & correspond to the same gene on ecocyc, so you double count?
        # TODO: some sort of curation to make sure there's a one-to-one correspondence between a gene here and a gene that ecocyc has (unless in cases where there shouldn't be)
        return gene_categorized
        # Overall using the gene_names symbols and Ecocyc synonyms: 70 are category -1, 1383 are category 0, 9 are category 1,  735 are category 2, 167 are category 3, 1825 are category 4
        # Ecocyc's own data:
        # around 43 + (whatever non-overlapping genes) 0, 9 are 1, 811 are 2, 165 are 3, 1992 are 4.
        # 0 category will include un-regulated genes, and also uncharacterized genes, so will need to do smth about that.
        # TODO: curate these by showing translational regulation
        # TODO: Then, get the statistics for candidates_low_z, try maybe using some metric also for some counts above some z-value,
        #  aand also overall category statistics for all genes according to sigma, prob_r, and max z/tail thing!

        import ipdb; ipdb.set_trace()

    def compare_category_statistics(self, all_genes, subset):
        all_genes_category = self.categorize_genes(all_genes)
        subset_category = self.categorize_genes(subset)
        stats_all = [len(all_genes_category[x]) for x in [0, 1, 2, 3, 4]]
        stats_subset = [len(subset_category[x]) for x in [0, 1, 2, 3, 4]]
        stats_all_freq = np.array(stats_all) / np.sum(stats_all)
        stats_error = np.sqrt(np.sum(stats_subset) * np.multiply(stats_all_freq, 1-stats_all_freq))

        chisquare = stats.chisquare(stats_subset, np.array(stats_all) * np.sum(stats_subset) / np.sum(stats_all), ddof=4)
        import ipdb; ipdb.set_trace()
        return stats_subset, stats_all, chisquare, stats_error



class MakeHeatmaps(object):
    def __init__(self):
        pass

    def heatmap(self, expression, output_name, ordering):
        # If range is from 0 to 20, and want 0.1, then that's 200 bins
        expression = expression[np.array(ordering)[::-1], :]
        bins = np.linspace(-12, 12, 401)
        histograms = []
        for exp in expression:
            exp = np.array(exp) - np.mean(exp)
            hist, _ = np.histogram(exp, bins=bins)
            log_hist = np.log10(hist + 1)
            histograms.append(log_hist)

        histograms = np.array(histograms)

        fig, ax = plt.subplots(figsize=(10, 200))
        ax.imshow(histograms, aspect="auto")
        ax.set_xlabel("Expression value centered around mean (mean is 100, from mean - 12 to mean + 12)")
        ax.set_ylabel("Genes, clustered order")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, output_name))
        plt.close('all')
        import ipdb; ipdb.set_trace()

class SimilarityMetrics(object):
    def __init__(self):
        pass

    def lowest_distance(self, expA, expB, metric="JS"):
        lowest_distance = 1
        histB, _ = np.histogram(expB - np.mean(expB), bins=np.linspace(-12, 12, num=401))
        probB = histB / np.sum(histB)
        # histA, _ = np.histogram(expA - np.mean(expA), bins=np.linspace(-12, 12, num=401))
        # probA = histA / np.sum(histA)
        # return self.JS_metric(probA, probB)
        for i in np.concatenate((np.arange(np.mean(expA), np.max(expA)-12, -0.06),
                                 np.arange(np.mean(expA)+0.06, np.min(expA)+12, 0.06))):
            histA, _ = np.histogram(expA, bins=np.linspace(i-12, i+12, num=401))
            probA = histA / np.sum(histA)
            if metric=="JS":
                js = self.JS_metric(probA, probB)
                lowest_distance = min(lowest_distance, js)
            elif metric=="Heilinger":
                heilinger = self.Heilinger_metric(probA, probB)
                lowest_distance = min(lowest_distance, heilinger)

        return lowest_distance
        # TODO: whether I can use a difference between shape-shifted and non-shifted as a metric for two-peak??

    def JS_metric(self, probA, probB):
        #probA, probB = self._to_prob(expA, expB)
        return spatial.distance.jensenshannon(probA, probB)

    def Heilinger_metric(self, probA, probB):
        heilinger = np.dot(np.sqrt(probA), np.sqrt(probB))
        return np.sqrt(1 - heilinger)
    # TODO: I could try displacing the probas to get the best match?

    def make_matrix(self, exps, metric="JS"):
        matrix = np.zeros((len(exps), len(exps)))
        for i in range(len(exps)):
            for j in range(i, len(exps)):
                matrix[i, j] = self.lowest_distance(exps[i, :], exps[j, :], metric=metric)
                matrix[j, i] = matrix[i, j]

        return matrix

    def cluster_genes(self, exps, names, output, metric="JS"):
        pdist = spatial.distance.pdist(exps, metric=lambda u, v:self.lowest_distance(u, v, metric=metric))
        linkage = cluster.hierarchy.linkage(pdist)
        fig, ax = plt.subplots(1, figsize=(20, 200))
        dn = cluster.hierarchy.dendrogram(linkage, labels=names, ax=ax, orientation="left", color_threshold=0)
        leaves = dn["leaves"]
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, output))
        return leaves

class HistogramPreparations(object):
    def hist_of_hist(self, expression, output_name):
        bins = np.linspace(0, 20, 401)
        histograms = np.zeros(1500)
        for exp in expression:
            hist, _ = np.histogram(exp, bins=bins)
            hist_of_hist, _ = np.histogram(hist, bins=[x for x in range(1501)])
            histograms += hist_of_hist

        zeros = int(histograms[0])
        histograms[0] = 0
        non_zero = max(histograms.nonzero()[0])


        fig, ax = plt.subplots(1, figsize=(15, 5))
        ax.bar([x for x in range(non_zero + 10)], histograms[:non_zero+10], 1)
        ax.set_title("Histogram of histograms of gene exp per gene, zeros = "+ str(zeros))
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Occurrence across all genes and expression values")
        plt.savefig(os.path.join(OUTPUT_DIR, output_name))
        plt.close('all')

class HistogramAlignment(object):
    def __init__(self):
        self.spacing = 20
        self.range = 24
        self.gap_penalty = -0.1
        self.aa_score = lambda u, v: 0.75 - np.abs((u-v)**3 / (u+v)**2) / 80
        self.bb_score = 0
        self.ab_score = -0.5
        # TODO: do I need a gap-A and gap-B score? try it with gap-A and gap-B scores. Then it'll
        # make ones where there's not matching, worse than currently, where it's mostly coming
        # from the extra potential for score in the wider one.
        # Does gap to A being small, ever allow it to cheat out of some bad scenario with an ok score? (
        # On the other hand, does it ever cause a good-matching scenario to become a bad-matching one (slightly diff stds?)
        # TODO: make align with sequences to see how they're aligning
        # TODO: normalize scores

        # TODO: how do I want it to be different than simply comparing whenver smth is present? 1. gaps, 2. actual magnitudes tell us when smth is important,
        # 3. anything else? Think!
        # To optimize, I can at least shave off the ends of the widest of the two.
        # TODO: can I shave off the end of each as well?

    def make_hists(self, expression):
        bins = np.linspace(-12, 32, 44 * self.spacing + 1)
        hists = np.zeros((len(expression), self.range * self.spacing))
        for i, exp in enumerate(expression):
            hist, _ = np.histogram(exp, bins=bins)
            max_value = np.argmax(hist)
            # if not isinstance(max_value, int):
            #     max_value = max_value[0]
            centered_hist = hist[max_value - 12 * self.spacing: max_value + 12 * self.spacing]
            hists[i, :] = centered_hist

        return hists

    def align_score_only(self, histA, histB):
        dim = self.spacing * self.range + 1
        align_matrix = np.zeros((dim, dim))
        align_matrix[0, :] = np.linspace(0, -1 * (dim-1) * self.gap_penalty, dim)
        align_matrix[:, 0] = np.linspace(0, -1 * (dim-1) * self.gap_penalty, dim)

        for i in range(1, dim):
            for j in range(1, dim):
                m_score = align_matrix[i-1, j-1] + self.score(histA[i-1], histB[j-1])
                x_score = align_matrix[i-1, j] + self.gap_penalty
                y_score = align_matrix[i, j-1] + self.gap_penalty
                align_matrix[i, j] = max(m_score, x_score, y_score)

        return align_matrix[-1, -1]

    def score(self, v1, v2):
        if v1 == 0:
            if v2 == 0:
                return self.bb_score
            else:
                return self.ab_score
        else:
            if v2 == 0:
                return self.ab_score
            else:
                return self.aa_score(v1, v2)

    def get_scores(self, expression):
        hists = self.make_hists(expression)
        length = len(expression)
        scores = np.zeros((length, length))
        for i in range(length):
            for j in range(i, length):
                scores[i, j] = self.align_score_only(hists[i, :], hists[j, :])

        return scores

    def normalize_scores(self, scores):
        length = np.shape(scores)[0]
        for i in range(length):
            for j in range(i, length):
                scores[i, j] = scores[i, j] / (scores[i, i] + scores[j, j])
                scores[j, i] = scores[i, j]

    def cluster_dendrogram(self, expression, gene_names, plot_name):
        scores = self.get_scores(expression)
        self.normalize_scores(scores)

        pdist = spatial.distance.pdist([[x] for x in range(len(expression))], lambda i, j: 1 - scores[i[0], j[0]])
        linkage = cluster.hierarchy.linkage(pdist, method='ward')

        fig, ax = plt.subplots(1, figsize=(len(gene_names), len(gene_names)))
        dn1 = cluster.hierarchy.dendrogram(linkage, ax=ax, labels=gene_names, orientation="right")
        ax.set_title("Dendrogram")
        plt.savefig(plot_name)
        plt.close('all')
        return scores, dn1


class PlotComponents(object):
    # So: given a gene and a range, I'll say 1. Print out ids of them,
    # and other useful info like the GSEs and stuff
    # 2. What are the things that are enriched or not enriched in this?
    # 3.Plot histograms for those

    def __init__(self, components, samples):
        self.components = np.array(components)
        self.samples = np.array(samples)
        self.n_total = np.shape(samples)[0]
        self.total_freqs = np.sum(self.samples, axis=0)
        self.frac_cutoff = 0.75
        self.prob_cutoff = 0.95

        self.bins_per_unit = 10
        self.n_bins = lambda x: int(np.ceil((np.max(x) - np.min(x)) * self.bins_per_unit))
        self.components_dir = os.path.join(OUTPUT_DIR, 'component_plots')
        if not os.path.exists(self.components_dir):
            os.mkdir(self.components_dir)

    def exclude_component(self, expression, components_to_exclude):
        components_idxs = np.isin(self.components, components_to_exclude)
        contains_exclude = self.samples[:, components_idxs].astype(bool)
        exclude_mask = np.all(~contains_exclude, axis=1)

        new_components = self.components[~components_idxs]
        new_samples = self.samples[exclude_mask, :]
        new_samples = new_samples[:, ~components_idxs]

        return expression[:, exclude_mask], new_components, new_samples

    def get_subset(self, expression, interval):
        subset_idxs = []
        for i, x in enumerate(expression):
            if x > interval[0]:
                if x < interval[1]:
                    subset_idxs.append(i)

        return np.array(subset_idxs)

    def get_enriched(self, subset_idxs):
        subset_freqs = np.sum(self.samples[subset_idxs], axis=0)
        num_subset = len(subset_idxs)
        enriched_idxs = []

        # TODO: only testing for enrichment rn, can test for de-enrichment later
        for i, (subset, total) in enumerate(zip(subset_freqs, self.total_freqs)):
            hypergeom = stats.hypergeom(self.n_total, total, num_subset)
            confidence_interval = hypergeom.interval(self.prob_cutoff)
            if subset > confidence_interval[1]:
                enriched_idxs.append(i)
            elif subset/total > self.frac_cutoff:
                enriched_idxs.append(i)

        return enriched_idxs

    def get_combo_enriched(self, subset_idxs):
        subset_samples = self.samples[subset_idxs]
        enriched_pairs = []
        for i in range(len(self.components)):
            for j in range(i+1, len(self.components)):
                has_both = np.sum(np.logical_and(subset_samples[:, i],
                                                subset_samples[:, j]))
                total_has_both = np.sum(np.logical_and(self.samples[:, i],
                                                       self.samples[:, j]))
                hypergeom = stats.hypergeom(self.n_total, total_has_both, has_both)
                confidence_interval = hypergeom.interval(1 - 1e-6)
                if has_both > confidence_interval[1]:
                    enriched_pairs.append((self.components[i], self.components[j]))
                elif has_both / total_has_both > self.frac_cutoff:
                    enriched_pairs.append((self.components[i], self.components[j]))

        import ipdb; ipdb.set_trace()

    def plot_exp_control_components(self, expression, plot_name, gene_names,
                                    exp_comp, control_comp,
                        vectorized_samples=None):
        if vectorized_samples is None:
            vectorized_samples = self.samples
        fig, axs = plt.subplots(nrows=4, ncols=2 * len(expression),
                                figsize=(10*len(expression), 10))
        for i, comp in enumerate([exp_comp, control_comp]):
            idx = np.where(self.components==comp)[0][0]
            includes_compound = vectorized_samples[:, idx].astype(bool)
            for j, exp in enumerate(expression):
                axs[i, 2*j].hist(exp[includes_compound], bins=self.n_bins(exp[includes_compound]))
                axs[i, 2*j].set_title("Has" + str(comp) + ", " + str(gene_names[j]))

                axs[i, 2*j+1].hist(exp[~includes_compound], bins=self.n_bins(exp[~includes_compound]))
                axs[i, 2*j+1].set_title("Does not have" + comp + "," + str(gene_names[j]))

        includes_1 = vectorized_samples[:, np.where(self.components==exp_comp)[0][0]].astype(bool)
        includes_2 = vectorized_samples[:, np.where(self.components==control_comp)[0][0]].astype(bool)
        has_1_not_2 = np.logical_and(includes_1, ~includes_2)
        has_2_not_1 = np.logical_and(~includes_1, includes_2)
        for j, exp in enumerate(expression):
            axs[2, 2 * j].hist(exp, bins=self.n_bins(exp))
            axs[2, 2 * j].set_title("Total " + str(gene_names[j]))

            axs[3, 2 * j].hist(exp[has_1_not_2], bins=self.n_bins(
                exp[has_1_not_2]))
            axs[3, 2 * j].set_title("Has " + str(exp_comp) + " not "
                                                  + str(control_comp) + " " + str(
                gene_names[j]))

            axs[3, 2 * j + 1].hist(exp[has_2_not_1], bins=self.n_bins(
                exp[has_2_not_1]))
            axs[3, 2 * j + 1].set_title("Has " + str(control_comp) + " not "
                                                  + str(exp_comp) + " " + str(
                gene_names[j]))

        plt.tight_layout()
        plt.savefig(os.path.join(self.components_dir, plot_name))
        plt.close('all')

    def plot_all_components(self, expression, plot_name):
        fig, axs = plt.subplots(nrows=len(self.components), ncols=2,
                                figsize=(6, 3 * len(self.components)))
        for idx, comp in enumerate(self.components):
            includes_compound = self.samples[:, idx].astype(bool)

            axs[idx, 0].hist(expression[includes_compound], bins=self.n_bins(expression))
            axs[idx, 0].set_title("Has" + comp)

            axs[idx, 1].hist(expression[~includes_compound], bins=self.n_bins(expression))
            axs[idx, 1].set_title("Does not have" + comp)
        plt.tight_layout()
        plt.savefig(os.path.join(self.components_dir, plot_name))
        plt.close('all')

    def plot_enriched(self, expression, interval, plot_name, combos=1):
        subset_idxs = self.get_subset(expression, interval)
        import ipdb; ipdb.set_trace()
        if combos == 1:
            enriched_idxs = self.get_enriched(subset_idxs)
        elif combos == 2:
            enriched_idxs = self.get_combo_enriched(subset_idxs)

        fig, axs = plt.subplots(len(enriched_idxs)+1, figsize=(5, 5 * len(enriched_idxs)+1), sharex=True)

        axs[0].hist(expression, bins=self.n_bins(expression))
        axs[0].set_title("All")
        if interval[1] != np.inf:
            axs[0].plot([interval[1], interval[1]], [0, 100])
        if interval[0] != 0:
            axs[0].plot([interval[0], interval[0]], [0, 100])

        for i, idx in enumerate(enriched_idxs):
            # self.samples is like 2000 by 100. So we'll say that, for each
            #
            includes_compound = self.samples[:, idx].astype(bool)
            compound_expression = expression[includes_compound]
            axs[i+1].hist(compound_expression, bins=self.n_bins(expression))
            axs[i+1].set_title(self.components[idx])
            if interval[1] != np.inf:
                axs[i+1].plot([interval[1], interval[1]], [0, 100])
            if interval[0] != 0:
                axs[i+1].plot([interval[0], interval[0]], [0, 100])

        plt.tight_layout()
        plt.savefig(os.path.join(self.components_dir, plot_name))
        plt.close('all')



        # class CompareComponents(object):
        #     def __init__(self, components, overall):
        #         self.components = np.array(components)
        #         self.all_treatments = overall
        #         self.n_total = np.shape(self.all_treatments)[0]
        #         self.treatment_freqs = np.sum(self.all_treatments, axis=0)
        #
        #     def compare(self, subset):
        #         subset_freqs = np.sum(subset, axis=0)
        #         # first_statistics = np.sum(subset, axis=0) / self.treatment_freqs * self.n_total / np.shape(subset)[0]
        #         # What I'm doing is, a subset of e.g. 100 out of 2000, see whether it is enriched in a certain property which is present x% overall.
        #
        #         # is_enriched = []
        #         # is_depleted = []
        #         cdfs = []
        #         for subset_freq, total_freq in zip(subset_freqs, self.treatment_freqs):
        #             hypergeom = stats.hypergeom(self.n_total, total_freq, np.shape(subset)[0])
        #             cdf = hypergeom.cdf(subset_freq)
        #             cdfs.append(cdf)
        #             # confidence_interval = hypergeom.interval(1 - 0.05/150)
        #             # is_enriched.append((subset_freq > confidence_interval[1]))
        #             # is_depleted.append((subset_freq < confidence_interval[0]))
        #         # enriched = self.components[first_statistics > 1]
        #         # depleted = self.components[first_statistics < 1]
        #
        #         # enriched_components = self.components[is_enriched]
        #         # depleted_components = self.components[is_depleted]
        #         fraction_treatment = subset_freqs / self.treatment_freqs
        #         # import ipdb; ipdb.set_trace()
        #         # return enriched_components, depleted_components, fraction_treatment
        #         return cdfs, fraction_treatment
        #         # Another thing is to look at each component, and just look whether it deviates significantly from overall. Hmm.
        #         # TODO: depleted will contain most things.

class PeakDetection(object):
    def __init__(self):
        # TODO: how to find parameters for how to detect peaks?
        # What I want it to do, is return me genes with only one peak,
        # genes with only two peaks. And maybe have some level of significance for it.
        # And also like, how low the valley goes (so just opposite of peak)?
        # A higher window, makes less peaks.
        # A higher number of bins, makes less peaks (probably maybe?).

        # Test we can do: let's just say for now, we're fixing a window and
        # n_bins for all genes. Then, we'll see how many genes end up having
        # 1 peak, 2 peaks, 3 peaks, etc. And compare for different window
        # and bin sizes to see what is reasonable?

        # Now, we want an identification of a class of one-peak genes,
        # and a class of two-peak genes. A class of one-peak genes,
        # that is if they have a low enough 95% interval.
        # A class of two-peak genes, that is if they have kind of clearly
        # one big thing with lowish 95% interval, and another bump with
        # a lowish 95% interval.
        # We could try just fitting two gaussians? But I could see problems
        # with that, if they don't look like gaussians.
        # Maybe, we take the point that is minimum between them, and then
        # take 95% confidence intervals on both sides? The problem is what
        # if there's a sudden drop. Well then you could do moving
        # average maybe?
        # Let's try just taking the minimum without any smoothing for now.
        # Now, can this act as a good way to find out two-peak category?
        # Well, let's see how two-peak genes look based on their widths?

        # TODO: so perhaps a two-peak criteria is, two peaks separated by at least
        # some amount, and their widths are below some threshold?
        # TODO: make plots for 1. the widths (a scatterplot of higher peak on x
        # vs lower peak on y, maybe note the ones where the peak heights r
        # p close to one another? or instead of that, like note the
        # possible differences in peak area? THIS peak area might be better),
        # 2. for the inter-peak distances,
        # and others to understand it?
        # We can get the "population standard deviation" for each peak,
        # and this might be a way to normalize for say, peaks with more
        # data points might tend to be wider, and "erroneously" cross the
        # threshold.
        self.window = 10
        self.bins_per_unit = 10
        self.n_bins = lambda x: int(np.ceil((np.max(x) - np.min(x)) * self.bins_per_unit))
        self.to_coord = lambda x, exp: np.min(exp) + x/self.bins_per_unit
        self.to_width = lambda x: x / self.bins_per_unit
        self.peak_dir = os.path.join(OUTPUT_DIR, 'peak_finder')

    def plot_genes(self, expression, gene_names, plot_name, num_regulators=None):
        # fig, axs = plt.subplots(len(expression), figsize=(5, 5*len(expression)), sharex=True)
        if num_regulators is not None:
            sort = np.argsort(num_regulators)
            gene_names = gene_names[sort]
            num_regulators = num_regulators[sort]
            expression = expression[sort]
        #
        # for i, exp in enumerate(expression):
        #     n_bins = self.n_bins(exp)
        #     axs[i].hist(exp, bins=n_bins)
        #     if num_regulators is not None:
        #         axs[i].set_title(gene_names[i]+str(num_regulators[i]))
        #     else:
        #         axs[i].set_title(gene_names[i])
        #
        # plt.tight_layout()
        # plt.savefig(os.path.join(self.peak_dir, plot_name))
        # plt.close('all')

        if num_regulators is not None:
            unique = np.unique(num_regulators)
            fig, axs = plt.subplots(nrows=len(unique), ncols=2, figsize=(5*2, 5*len(unique)))
            standard_devs = [np.std(exp) for exp in expression]
            bins = np.linspace(0, np.max(standard_devs), 20)
            for i, n in enumerate(unique):
                idxs = np.where(np.array(num_regulators) == n)[0]
                chosen_standard_devs = [standard_devs[idx] for idx in idxs]
                axs[i, 0].hist(chosen_standard_devs, bins=bins)
                axs[i, 0].set_title(str(n)+" regulators, standard deviation")
                axs[i, 0].set_xlabel("Overall standard deviation of expression")
                axs[i, 0].set_ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(os.path.join(self.peak_dir, plot_name+"stats"))
        plt.close('all')



    def detect_peaks(self, exp, plot_name=None, find_standard_devs=False):
        n_bins = self.n_bins(exp)
        hist, _ = np.histogram(exp, bins=n_bins)
        length = len(hist)
        peaks = []
        for i, n in enumerate(hist):
            is_peak = True
            if n == 0:
                is_peak = False
            for j in range(max(0, i - self.window), min(length, i + self.window)):
                if hist[j] > hist[i]:
                    is_peak = False
            if is_peak:
                # TODO: temporary fix for problem of having multiple same-height peaks
                # in a narrow region, is just to only add the leftmost peak?
                if len(peaks) > 0:
                    if i > peaks[-1][0] + self.window:
                        peaks.append((i, n))
                else:
                    peaks.append((i,n))

        if plot_name:
            fig, ax = plt.subplots(1, figsize=(5, 5))
            ax.hist(exp, bins=n_bins)

            plt.tight_layout()
            plt.savefig(os.path.join(self.peak_dir, plot_name))
            plt.close('all')

        # def is_valley(i):
        #     is_valley = True
        #     for j in range(max(0, i - self.window), min(length, i + self.window)):
        #         if hist[j] < hist[i]:
        #             is_valley = False
        #     return is_valley

        if find_standard_devs:
            # TODO: one possible problem with this method is that if the two
            # bumps are overlapping, the widths will seem smaller because
            # we're cutting off from the min point in between, so assuming
            # the two peaks get to their finishes (which would not be true
            # if they substantially overlap)
            # Actually, we're just using population standard deviation
            # instead of widths.
            peaks_with_ends = [x[0] for x in peaks]
            peaks_with_ends.append(n_bins-1)
            peaks_with_ends.insert(0, 0)

            standard_devs = []
            areas = []
            for i, peak in enumerate(peaks_with_ends[:-1]):
                if i > 0:
                    # Find ends of peaks
                    # TODO: maybe make this more robust to low-count noise,
                    # maybe by accepting any index having the lowest 2 or 3
                    # counts on the histogram as an acceptable end?
                    # If the minimum is indeed somewhere in the valley though,
                    # and it's fairly smooth, it's probably fine, because even
                    # if you were to stretch the end more, that would just
                    # increase the threshold for 95%. Basically, you're trying
                    # to capture the bulk of the bump, which hopefully is
                    # included with this scheme?? Not sure if this is true though.
                    # TODO: make the left end for the absolute left end just the end of the histogram,
                    # and same thing for the right end.
                    # TODO: simplify the code to just get the ends and then calculate
                    # so you don't re-calculate half of the ends for each gene
                    if i == 1:
                        left_end = 0
                    else:
                        left_end = np.argmin(hist[peaks_with_ends[i-1]:peak+1]) + peaks_with_ends[i-1]
                    if i == len(peaks_with_ends)-2:
                        right_end = n_bins - 1
                    else:
                        right_end = peaks_with_ends[i+1] - np.argmin(hist[peak:peaks_with_ends[i+1]+1][::-1])
                    # Find width by progressively taking away from left or right end
                    # (primitive dynamic programming?)
                    total = np.sum(hist[left_end:right_end+1])
                    areas.append(total)

                    tails_sum = 0
                    # TODO: I can see a bad case where, if there's a tiny bump
                    # far off to the right, but there's a zero somewhere in the
                    # end of a big bump, then the left_end of the tiny bump
                    # will be put there, and so the tiny bump will erroneously
                    # include some of the tail, and so will seem bigger than
                    # it is. But ig it is also an ambiguous case because
                    # if the tiny bump was closer to the tail,
                    # you might think it makes sense to include some of the
                    # tail. TODO: maybe you could say smth like if there's
                    # a huge clear gap of 0 between two bumps, then set your
                    # left end in there?
                    # TODO: THINK: maybe just get rid of this whole deleting tails
                    # thing for calculating standard deviation?
                    while tails_sum < total * 0.05:
                        # TODO: a cleaner way to do this? Basically, if the
                        # tail forms a big part of the total, i.e. the
                        # bump is small, then don't take it away
                        if hist[left_end] > total * 0.1:
                            break
                        elif hist[right_end] > total*0.1:
                            break

                        if hist[left_end] >= hist[right_end]:
                            tails_sum += hist[left_end]
                            left_end += 1
                        else:
                            tails_sum += hist[right_end]
                            right_end -= 1

                    # TODO: whether doing the std while trimming off the 5% tails,
                    # whether that might decrease things unexpectedly or like
                    # get rid of outliers which we might care about? Or is that ok
                    # bc we're already considering at the level of peaks,
                    # and just want to get at the overall shape? And maybe the
                    # outliers wouldn't even do much?
                    sigma = stats.tstd(exp, limits=(self.to_coord(left_end, exp),
                    self.to_coord(right_end+1, exp)), inclusive=(True, True))
                    standard_devs.append(self.to_width(sigma))
                    # TODO: widths might still be useful to have just to see
                    # what the program is saying is the limits of the peaks?
                    #widths.append(self.to_coord(right_end, exp) - self.to_coord(left_end, exp))

            standard_devs = np.nan_to_num(standard_devs)
            peak_coords = [(self.to_coord(x[0], exp), y) for x, y in zip(peaks, areas)]
            return peak_coords, standard_devs
            # I'll say, for every peak, to the left and right, find the minimum.
            # Then, between these minimums, find width of 95% interval.
            # ends = []
            # for p, _ in peaks:
            #     right_end = n_bins
            #     left_end = 0
            #     for i in range(p, n_bins):
            #         if is_valley(i):
            #             right_end = i
            #             break
            #
            #     for i in range(p, 0, -1):
            #         if is_valley(i):
            #             left_end = i
            #             break
            #
            #     ends.append((self.to_coord(left_end, exp), self.to_coord(right_end, exp)))

            return peaks

        return peaks

    def detect_peaks_multigene(self, expression, plot_names=None, find_standard_devs=False):
        all_peaks = []
        for i, exp in enumerate(expression):
            peaks = self.detect_peaks(exp, plot_name=plot_names[i],
                                      find_standard_devs=find_standard_devs)
            all_peaks.append(peaks)
            # if len(peaks) == 1:
            #     one_peak_idxs.append(i)
            # elif len(peaks) == 2:
            #     two_peak_idxs.append(i)


        return all_peaks

    def param_testing(self, expression, windows, plot_name):
        num_ones_list = []
        num_twos_list = []
        for w in windows:
            self.window = w
            one, two, _ = self.detect_peaks_many(expression)
            num_ones = len(one)
            num_twos = len(two)
            num_ones_list.append(num_ones)
            num_twos_list.append(num_twos)

        fig, axs = plt.subplots(2, figsize=(5, 10))
        axs[0].plot(windows, num_ones_list)
        axs[0].set_title("Num of one-peak genes vs window size")
        axs[1].plot(windows, num_twos_list)
        axs[1].set_title("Num of two-peak genes vs window size")
        plt.tight_layout()
        plt.savefig(os.path.join(self.peak_dir, plot_name))
        plt.close('all')
        # It seems that between 8 and 14, the number of two-peaks is relatively stable?
        # Although it increases steadily for one-peak. So 10 is a reasonable guess for now?
        # TODO: test for different bin sizes.


    def gene_data(self, symbols, expression, plot_name, record_name):
        # For all genes, record their peak locations, peak areas, and standard devs.
        # For two-peak genes, plot the distance between the peaks.
        # For two-peak genes, plot the widths of the two peaks, for now
        # just saying the peak on the left is the y-axis, peak on the right
        # is the x-axis. (or also can lump together, or plot by the peak heights,
        # or plot by the peak areas, etc.)

        num_peaks = []
        all_peaks = []
        all_areas = []
        all_standard_devs = []

        for i, exp in enumerate(expression):
            peaks, standard_devs = self.detect_peaks(exp, find_standard_devs=True)
            num_peaks.append(len(peaks))
            all_peaks.append([round(p[0], 2) for p in peaks])
            all_areas.append([p[1] for p in peaks])
            all_standard_devs.append([round(w, 4) for w in standard_devs])

        sort = np.argsort(num_peaks)
        num_peaks = np.array(num_peaks)[sort]
        all_peaks = [all_peaks[i] for i in sort]
        all_standard_devs = [all_standard_devs[i] for i in sort]
        all_areas = [all_areas[i] for i in sort]
        genes = np.array(symbols)[sort]

        write_file = os.path.join(self.peak_dir, record_name)
        with open(write_file, 'w') as f:
            writer = csv.writer(f, delimiter='\t', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            header = ['gene', 'num_peaks', 'peaks', 'areas', 'standard_devs']
            writer.writerow(header)
            for gene, num, peak, area, standard_devs in zip(genes, num_peaks, all_peaks, all_areas, all_standard_devs):
                writer.writerow([gene, num, peak, area, standard_devs])


        one_peak_idxs = np.where(num_peaks==1)[0]
        one_peak_standard_devs = [all_standard_devs[i] for i in one_peak_idxs]

        two_peak_idxs = np.where(num_peaks==2)[0]
        two_peaks = [all_peaks[i] for i in two_peak_idxs]
        two_peak_dist = [np.abs(x[0] - x[1]) for x in two_peaks]
        two_peak_areas = [all_areas[i] for i in two_peak_idxs]
        two_peak_standard_devs = [all_standard_devs[i] for i in two_peak_idxs]



        max_standard_dev = np.max(two_peak_standard_devs)
        fig, axs = plt.subplots(6, figsize=(5, 30))
        axs[0].scatter([x[0] for x in two_peak_standard_devs], [x[1] for x in two_peak_standard_devs], s=0.5)
        axs[0].set_xlim(0, max_standard_dev+0.01)
        axs[0].set_ylim(0, max_standard_dev+0.01)
        axs[0].set_title("Two-peak standard deviations")
        axs[0].set_xlabel("Left peak")
        axs[0].set_ylabel("Right peak")
        axs[0].plot([0, max_standard_dev+0.01], [0, max_standard_dev + 0.01], lw=0.1)

        axs[1].hist(two_peak_dist)
        axs[1].set_title("Histogram of distance between peaks")
        axs[1].set_xlabel("Distance (log2)")
        axs[1].set_ylabel("Frequency")

        num_samples = np.shape(expression)[1]
        axs[2].scatter([x[0] for x in two_peak_areas], [x[1] for x in two_peak_areas], s=0.5)
        axs[2].set_xlim(0, num_samples+100)
        axs[2].set_ylim(0, num_samples+100)
        axs[2].plot([0, num_samples+100], [0, num_samples+100], lw=0.1)
        axs[2].plot([0, num_samples], [num_samples, 0], lw=0.1)
        axs[2].plot()
        axs[2].set_title("Two-peak areas")
        axs[2].set_xlabel("Left peak")
        axs[2].set_ylabel("Right peak")

        two_peak_standard_devs_sorted = []
        colors = []
        for i, areas in enumerate(two_peak_areas):
            # The x axis plots the higher area one
            if areas[0] < areas[1]:
                two_peak_standard_devs_sorted.append(two_peak_standard_devs[i][::-1])
            else:
                two_peak_standard_devs_sorted.append(two_peak_standard_devs[i])
            # TODO: figure out this cutoff based on histogram for peak area differences?
            if np.abs(areas[0] - areas[1]) < 500:
                colors.append('r')
            else:
                colors.append('b')
        axs[3].scatter([x[0] for x in two_peak_standard_devs_sorted],
                       [x[1] for x in two_peak_standard_devs_sorted], s=0.5, c=colors)
        axs[3].set_xlim(0, max_standard_dev+0.01)
        axs[3].set_ylim(0, max_standard_dev+0.01)
        axs[3].plot([0, max_standard_dev+0.01], [0, max_standard_dev+0.01], lw=0.1)
        axs[3].set_title("Two-peak standard deviations")
        axs[3].set_xlabel("Bigger peak")
        axs[3].set_ylabel("Smaller peak")

        axs[4].hist(num_peaks, bins=[x for x in range(10)])
        axs[4].set_title("Histograms of number of peaks")
        axs[4].set_xlabel("Number of peaks")
        axs[4].set_ylabel("Number of genes")

        axs[5].hist(one_peak_standard_devs)
        axs[5].set_title("Histogram of one-peak standard deviations")
        axs[5].set_xlabel("Standard deviation")
        axs[5].set_xlim(0)
        axs[5].set_ylabel("Number of genes")

        # For the areas, this is saying is the left peak fatter, or is the right peak fatter?
        # if left peak was fatter, then we'd see more dots on the
        # Does there tend to be a fat and skinny or about even?
        # if there tends to be fat and skinny, then it'll be focused on the lower right
        # or upper left. If they're about even, it'll be more in the center.
        # Supposedly, everything falls approximately on a shifted y=-x line. They might fall over it.
        # If they do, we could just do a histogram instead.

        # For the standard devs, that's saying, what is left standard dev vs
        # right standard dev? If they're both low, then that could be a good candidate (?).
        # If they're both high, that's a gene that's more spread out and appears to have two peaks
        # but might be more complicated. If one is high and one is low, then if the high one
        # has a low area, that means it's like mainly one peak, but then has a large tail.
        # If the high one has a high area, then that means there's a lil concentrated section
        # out somewhere, but mainly it's p spread out but appearing as one peak.
        # So maybe it's particularly important that the high-area one has a low standard dev?

        # For standard devs of high vs lower area. If both are low, then that could be a good candidate.
        # I do expect to see the lower areas to have lower standard dev in general,
        # which would produce a bias to be lower in the y axis, so more to bottom.
        # If high area standard dev is high, while low area is low, then that
        # supposedly means it's a fat thing with a lil tail blotch, but the fat
        # thing may be more complicated than that.
        # If high area standard dev is low, while low area is high, then that
        # supposedly means it's a nice peak followed by a long thing. Which is
        # interesting too.
        # If we just look at the high area ones, it's interesting to see
        # how widely spread out those are. A lot of low values would mean
        # a lot of nice peaks with some sort of tail or other peak, which would
        # be encouraging. A lot of high values would either mean our window
        # is too large, or there's a lot of genes that are just kinda widely
        # spread out in a fat bell curve.
        # If we just look at the low area ones, it's interesting to see how widely
        # spread out those are. Low values might mean just a few points, or it might
        # mean a p tight bump. This means that there's a concentrated region,
        # far from the bulk of points, that some extra points are located, which
        # is kinda interesting. Unless there's a huge tail to the large one that this
        # is an accidental bump on, but ig that's more of a subjective thing.
        # If it's a high value, then that means there's a region
        # of points away from the bulk of points, that's highly spread out.
        # It could just be a very long tail for sure, or it could be an extra
        # region of different activity but which has lots of variety and so
        # be more complicated and possibly underrepresented (?).
        # It's encouraging to see that the higher peak area ones tend to be
        # on the lower standard dev side, like around 0.5, becuase it means
        # truly higher standard dev ones are probably gone to more numbers of
        # peaks, etc.?

        # TODO: might want to do smth with actual widths because that has a
        # nice interpretable meaning? whereas we might not know what a certain
        # standard deviation means, and it might correspond to different widths
        # depending on the number of samples, etc.?

        # TODO: might want to get at the minimum between peaks or smth like that?
        # Similar to interpeak distance, trying to get at how separate peaks are?

        # TODO: make a histogram of the difference between peak areas

        plot_file = os.path.join(self.peak_dir, plot_name)
        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close('all')

        import ipdb; ipdb.set_trace()

class coliNetData(object):
    def __init__(self):
        data_dir = os.path.join(BASE_DIR, 'ColiNet-1.1')
        names_dir = os.path.join(data_dir, 'coliInterFullNames.txt')
        regs_dir = os.path.join(data_dir, 'coliInterFullVec.txt')

        self.names_dict = {}
        operons = []
        tfs = []
        reg_direction = []
        with open(names_dir, 'r') as f:
            for line in f.readlines():
                num, operon = line.split()
                self.names_dict[int(num)] = operon

        with open(regs_dir, 'r') as f:
            for line in f.readlines():
                operon, tf, direction = line.split()
                operons.append(int(operon))
                tfs.append(int(tf))
                reg_direction.append(int(direction))

        self.reg_data = {"operons": np.array(operons),
            "tfs": np.array(tfs), "reg_dir": np.array(reg_direction)}


    def get_simple_reg(self):
        # If an operon has only one regulator, then it appears once in "operons".
        # If a tf only regulates one gene, then it appears in "tfs".
        single_reg_operon = []
        single_target_tf = []
        one_to_one_reg = []
        single_reg_no_autoreg = []

        all_regs = [tuple(x) for x in zip(self.reg_data['tfs'],
                                            self.reg_data['operons'])]

        for gene_idx in self.names_dict.keys():
            reg_idxs = np.where(self.reg_data['operons'] == gene_idx)[0]
            tf_idxs = np.where(self.reg_data['tfs'] == gene_idx)[0]
            if len(reg_idxs) == 1:
                # Append a tuple (Operon in question, TF that regulates it)
                single_reg_operon.append((gene_idx,
                self.reg_data['tfs'][reg_idxs[0]]))

            if len(tf_idxs) == 1:
                # Append a tuple (TF in question, Operon it regulates)
                single_target_tf.append((gene_idx,
                self.reg_data['operons'][tf_idxs[0]]))

        for (x, y) in single_reg_operon:
            if (y, x) in single_target_tf:
                one_to_one_reg.append((self.names_dict[x], self.names_dict[y]))
            if (y, y) not in all_regs:
                single_reg_no_autoreg.append((self.names_dict[x], self.names_dict[y]))


        import ipdb; ipdb.set_trace()
        return single_reg_operon, single_target_tf, one_to_one_reg

    #