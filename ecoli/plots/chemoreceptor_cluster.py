import os

from matplotlib import pyplot as plt


def plot_receptor_output(output, settings, out_dir='out', filename='response'):
    ligand_vec = output['external']['MeAsp']  # TODO -- configure ligand name
    receptor_activity_vec = output['internal']['chemoreceptor_activity']
    n_methyl_vec = output['internal']['n_methyl']
    time_vec = output['time']
    aspect_ratio = settings.get('aspect_ratio', 1)
    fontsize = settings.get('fontsize', 12)

    # plot results
    cols = 1
    rows = 3
    width = 3
    height = width / aspect_ratio
    plt.figure(figsize=(width, height))
    plt.rc('font', size=fontsize)

    ax1 = plt.subplot(rows, cols, 1)
    ax2 = plt.subplot(rows, cols, 2)
    ax3 = plt.subplot(rows, cols, 3)

    ax1.plot(time_vec, ligand_vec, 'steelblue')
    ax2.plot(time_vec, receptor_activity_vec, 'steelblue')
    ax3.plot(time_vec, n_methyl_vec, 'steelblue')

    ax1.set_xticklabels([])
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.tick_params(right=False, top=False)
    ax1.set_ylabel("external ligand \n (mM) ", fontsize=fontsize)

    ax2.set_xticklabels([])
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.tick_params(right=False, top=False)
    ax2.set_ylabel("cluster activity \n P(on)", fontsize=fontsize)

    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.tick_params(right=False, top=False)
    ax3.set_xlabel("time (s)", fontsize=12)
    ax3.set_ylabel("average \n methylation", fontsize=fontsize)

    fig_path = os.path.join(out_dir, filename)
    plt.subplots_adjust(wspace=0.7, hspace=0.3)
    plt.savefig(fig_path + '.png', bbox_inches='tight')
