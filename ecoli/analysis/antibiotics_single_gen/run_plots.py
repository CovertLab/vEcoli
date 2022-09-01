from ecoli.analysis.antibiotics_single_gen.mRNA_counts import Plot as PlotMrnaCounts
from ecoli.analysis.antibiotics_single_gen.monomer_counts import Plot as PlotMonomerCounts
from ecoli.analysis.antibiotics_single_gen.mRNA_synth_prob import Plot as PlotSynthProbs


if __name__ == "__main__":
    exp_ids = [
        # 3.375 uM tet.
        # 0 uM tet.
        # Baseline
    ]
    labels = [
        # 3.375 uM tet.
        # 0 uM tet.
        # Baseline
    ]
    savgol_args = {
        'window_length': 50,
        'polyorder': 3
    }
    PlotMrnaCounts(exp_ids, labels, out_file='out/analysis/mrna_counts_savgol.png', savgol_args=savgol_args)
    PlotMrnaCounts(exp_ids, labels, out_file='out/analysis/mrna_counts_savgol_no_norm.png', savgol_args=savgol_args, norm=False)
    PlotMrnaCounts(exp_ids, labels, out_file='out/analysis/mrna_counts.png')
    PlotMrnaCounts(exp_ids, labels, out_file='out/analysis/mrna_counts_no_norm.png', norm=False)
    PlotMonomerCounts(exp_ids[4:], labels[4:], out_file='out/analysis/monomer_counts.png')
    PlotSynthProbs(exp_ids, labels, out_file='out/analysis/synth_probs_savgol.png', savgol_args=savgol_args)
    PlotSynthProbs(exp_ids, labels, out_file='out/analysis/synth_probs_savgol_no_norm.png', savgol_args=savgol_args, norm=False)
    PlotSynthProbs(exp_ids, labels, out_file='out/analysis/synth_probs.png')
    PlotSynthProbs(exp_ids, labels, out_file='out/analysis/synth_probs_no_norm.png', norm=False)
