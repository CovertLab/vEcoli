from ecoli.analysis.antibiotics_single_gen.mRNA_counts import Plot as PlotMrnaCounts
from ecoli.analysis.antibiotics_single_gen.monomer_counts import Plot as PlotMonomerCounts
from ecoli.analysis.antibiotics_single_gen.mRNA_synth_prob import Plot as PlotSynthProbs

exp_ids = [
    # 3.375 uM tet.
    "2022-08-15_17-44-04_988625+0000",
    "2022-08-15_18-33-06_362648+0000",
    "2022-08-15_21-49-24_819030+0000",
    "2022-08-17_00-30-10_058639+0000",
    # 0 uM tet.
    "2022-08-15_23-00-34_973906+0000",
    "2022-08-16_16-00-40_928301+0000",
    "2022-08-16_18-48-44_621220+0000",
    "2022-08-16_19-42-37_994009+0000",
    # Baseline
    "2022-08-16_21-09-39_797440+0000",
    "2022-08-16_22-04-54_797907+0000",
    "2022-08-16_23-04-23_500547+0000",
    "2022-08-17_00-21-24_422276+0000"
]
labels = [
    "3.375 uM tet", "3.375 uM tet", 
    "3.375 uM tet", "3.375 uM tet",
    "0 uM tet", "0 uM tet", 
    "0 uM tet", "0 uM tet",
    "base", "base", "base", "base"
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
