import os
import matplotlib.pyplot as plt


def plot_variable_receptor(output, out_dir='out', filename='motor_variable_receptor'):
    receptor_activities = output['chemoreceptor_activity']
    CheY_P_vec = output['CheY_P']
    ccw_motor_bias_vec = output['ccw_motor_bias']
    ccw_to_cw_vec = output['ccw_to_cw']

    # plot results
    cols = 1
    rows = 2
    plt.figure(figsize=(5 * cols, 2 * rows))

    ax1 = plt.subplot(rows, cols, 1)
    ax2 = plt.subplot(rows, cols, 2)

    ax1.scatter(receptor_activities, CheY_P_vec, c='b')
    ax2.scatter(receptor_activities, ccw_motor_bias_vec, c='b', label='ccw_motor_bias')
    ax2.scatter(receptor_activities, ccw_to_cw_vec, c='g', label='ccw_to_cw')

    ax1.set_xticklabels([])
    ax1.set_ylabel("CheY_P", fontsize=10)
    ax2.set_xlabel("receptor activity \n P(on) ", fontsize=10)
    ax2.set_ylabel("motor bias", fontsize=10)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig_path = os.path.join(out_dir, filename)
    plt.subplots_adjust(wspace=0.7, hspace=0.1)
    plt.savefig(fig_path + '.png', bbox_inches='tight')
