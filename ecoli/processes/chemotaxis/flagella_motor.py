"""
========================
Flagella Motor Processes
========================

``FlagellaMotor`` :term:`process` generates thrust and torque based on number of
flagella and their individual motor states.

References:
 CheY phosphorylation model from:
    `Kollmann, M., Lovdok, L., Bartholome, K., Timmer, J., & Sourjik, V. (2005).
    Design principles of a bacterial signalling network. Nature.`
 Veto model of motor activity from:
    `Mears, P. J., Koirala, S., Rao, C. V., Golding, I., & Chemla, Y. R. (2014).
    Escherichia coli swimming is robust against variations in flagellar number. Elife.`
 Rotational state of an individual flagellum from:
    `Sneddon, M. W., Pontius, W., & Emonet, T. (2012). Stochastic coordination of multiple
    actuators reduces latency and improves chemotactic response in bacteria.`

"""

import os
import copy
import random
import math

import numpy as np

from vivarium.core.process import Process
from vivarium.core.composition import simulate_process, PROCESS_OUT_DIR
from vivarium.core.emitter import timeseries_from_data
from vivarium.plots.simulation_output import plot_simulation_output

# plots
from matplotlib import colors, pyplot as plt
from matplotlib.patches import Patch


NAME = "flagella_motor"


class FlagellaMotor(Process):
    """Flagellar motor activity

    :term:`Ports`:
     * **internal_counts**
     * **flagella**
     * **internal**
     * **boundary**
     * **membrane**

    """

    name = NAME
    expected_pmf = (
        -140
    )  # PMF ~170mV at pH 7, ~140mV at pH 7.7 (Berg H, E. coli in motion, 2004, pg 113)
    expected_flagella = 4
    expected_thrust = 0.5  # (pN) Hughes MP & Morgan H. (1999) Measurement of bacterial flagellar thrust.
    defaults = {
        "time_step": 0.01,
        "n_flagella": expected_flagella,
        #  CheY phosphorylation parameters
        "k_y": 100.0,  # 1/uM/s
        "k_z": 30.0,  # / [CheZ],
        "gamma_y": 0.1,  # rate constant
        "k_s": 0.45,  # scaling coefficient
        "adapt_precision": 8,  # scales CheY_P to cluster activity
        # rotational state of individual flagella
        # parameters from Sneddon, Pontius, and Emonet (2012)
        "omega": 1.3,  # (1/s) characteristic motor switch time
        "g_0": 40,  # (k_B*T) free energy barrier for CCW-->CW
        "g_1": 40,  # (k_B*T) free energy barrier for CW-->CCW
        "K_D": 3.06,  # binding constant of CheY-P to base of the motor
        # added parameters
        "ccw_to_cw_leak": 0.2,  # 1/s rate of spontaneous switch to cw
        # motile force parameters
        "flagellum_thrust": expected_thrust / math.log(expected_flagella + 1),
        "tumble_jitter": 60.0,
        "tumble_scaling": 0.5 / expected_pmf,  # scale to expected PMF
        "run_scaling": 1.0 / expected_pmf,  # scale to expected PMF
        # initial state
        "initial_state": {
            "internal": {
                # response regulator proteins
                "CheY": 2.59,
                "CheY_P": 2.59,  # (uM) mean concentration of CheY-P
                "CheZ": 1.0,  # (uM) phosphatase
                "CheA": 1.0,  # (uM)
                # sensor activity
                "chemoreceptor_activity": 1 / 3,
                # cell motile state
                "motile_state": 1,  # 1 for tumble, 0 for run
            },
            "membrane": {
                "PMF": expected_pmf,
            },
            "boundary": {
                "thrust": 0,
                "torque": 0,
            },
        },
        # seed for unique ID generation
        "seed": 0,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.random_state = np.random.RandomState(seed=self.parameters["seed"])

    def ports_schema(self):
        ports = [
            "internal_counts",
            "flagella",
            "internal",
            "boundary",
            "membrane",
        ]
        schema = {port: {} for port in ports}

        # internal_counts of flagella (n_flagella)
        schema["internal_counts"]["flagella"] = {
            "_default": self.parameters["n_flagella"],
            "_emit": True,
        }

        # flagella
        schema["flagella"] = {
            "_divider": "split_dict",
            "*": {"_default": 1, "_updater": "set", "_emit": True},
        }

        # internal
        state_emit = [
            "chemoreceptor_activity",
            "motile_state",
            "CheA",
            "CheY_P",
            "CheY",
        ]
        state_set_updater = [
            "motile_state",
            "CheA",
            "CheY_P",
            "CheY",
        ]
        for state, default in self.parameters["initial_state"]["internal"].items():
            schema["internal"][state] = {"_default": default}
            if state in state_emit:
                schema["internal"][state].update({"_emit": True})
            if state in state_set_updater:
                schema["internal"][state].update({"_updater": "set"})

        # boundary (thrust and torque)
        for state, default in self.parameters["initial_state"]["boundary"].items():
            schema["boundary"][state] = {
                "_default": default,
                "_emit": True,
                "_updater": "set",
            }

        # membrane
        for state in ["PMF", "protons_flux_accumulated"]:
            schema["membrane"][state] = {
                "_default": self.parameters["initial_state"]["membrane"].get(state, 0.0)
            }

        return schema

    def next_update(self, timestep, states):
        # get flagella sub-compartments and current flagella protein counts
        flagella = states["flagella"]
        n_flagella = states["internal_counts"]["flagella"]

        # proton motive force
        PMF = states["membrane"]["PMF"]

        # internal states
        internal = states["internal"]
        P_on = internal["chemoreceptor_activity"]
        CheY_0 = internal["CheY"]
        CheY_P_0 = internal["CheY_P"]

        # parameters
        adapt_precision = self.parameters["adapt_precision"]
        k_y = self.parameters["k_y"]
        k_s = self.parameters["k_s"]
        k_z = self.parameters["k_z"]
        gamma_y = self.parameters["gamma_y"]

        ## Kinase activity
        # relative steady-state concentration of phosphorylated CheY.
        # CheZ combined in k_z as dephosphorylation rate of CheY-P
        new_CheY_P = (
            adapt_precision * k_y * k_s * P_on / (k_y * k_s * P_on + k_z + gamma_y)
        )
        dCheY_P = new_CheY_P - CheY_P_0

        # TODO -- add an assert here instead
        CheY_P = max(new_CheY_P, 0.0)  # keep value positive
        CheY = max(CheY_0 - dCheY_P, 0.0)  # keep value positive

        ## Update flagella subcompartments
        # check number of flagella proteins, compare with sub-compartments add/delete accordingly
        flagella_update = {}
        new_flagella = int(n_flagella) - len(flagella)
        if new_flagella < 0:
            flagella_update["_delete"] = []
            remove = random.sample(list(flagella.keys()), abs(new_flagella))
            for flagella_id in remove:
                flagella_update["_delete"].append((flagella_id,))

        elif new_flagella > 0:
            flagella_update["_add"] = []
            new_flagella_indexes = [
                num for num in self.random_state.randint(0, 2**63, new_flagella)
            ]
            for index in range(new_flagella):
                flagella_id = new_flagella_indexes[index]
                flagella_update["_add"].append(
                    {"key": str(flagella_id), "state": random.choice([-1, 1])}
                )

        # update individual flagella states
        for flagella_id, motor_state in flagella.items():
            new_motor_state = self.update_flagellum(motor_state, CheY_P, timestep)
            flagella_update.update({flagella_id: new_motor_state})

        ## get cell motile state.
        # if any flagella is rotating CW, the cell tumbles.
        # flagella motor state: -1 for CCW, 1 for CW
        # motile state: -1 for run, 1 for tumble, 0 for no state
        if any(state == 1 for state in flagella_update.values()):
            motile_state = 1
            [thrust, torque] = self.tumble(n_flagella, PMF)
        elif len(flagella_update) > 0:
            motile_state = -1
            [thrust, torque] = self.run(n_flagella, PMF)
        else:
            motile_state = 0
            thrust = 0
            torque = 0

        return {
            "flagella": flagella_update,
            "internal": {"motile_state": motile_state, "CheY_P": CheY_P, "CheY": CheY},
            "boundary": {"thrust": thrust, "torque": torque},
        }

    def update_flagellum(self, motor_state, CheY_P, timestep):
        """
        calculate the rotational state of a individual flagellum

        .. note::
            TODO -- normal, semi, curly states from Sneddon
        """
        g_0 = self.parameters["g_0"]  # (k_B*T) free energy barrier for CCW-->CW
        g_1 = self.parameters["g_1"]  # (k_B*T) free energy barrier for CW-->CCW
        K_D = self.parameters["K_D"]  # binding constant of CheY-P to base of the motor
        omega = self.parameters["omega"]  # (1/s) characteristic motor switch time

        # free energy barrier
        delta_g = g_0 / 4 - g_1 / 2 * (CheY_P / (CheY_P + K_D))

        # switching frequency
        CW_to_CCW = omega * math.exp(delta_g)
        CCW_to_CW = omega * math.exp(-delta_g)
        CW_bias = CCW_to_CW / (CCW_to_CW + CW_to_CCW)
        CCW_bias = CW_to_CCW / (CW_to_CCW + CCW_to_CW)

        # don't let ccw_to_cw get under leak value
        if CW_bias < self.parameters["ccw_to_cw_leak"]:
            CW_bias = self.parameters["ccw_to_cw_leak"]

        # flagella motor state: -1 for CCW, 1 for CW
        if motor_state == -1:
            # switch probability as function of the time step
            switch_rate = -math.log(1 - CW_bias)
            prob_switch = 1 - math.exp(-switch_rate * timestep)
            if np.random.random(1)[0] <= prob_switch:
                new_motor_state = 1
            else:
                new_motor_state = -1

        elif motor_state == 1:
            # switch probability as function of the time step
            switch_rate = -math.log(1 - CCW_bias)
            prob_switch = 1 - math.exp(-switch_rate * timestep)
            if np.random.random(1)[0] <= prob_switch:
                new_motor_state = -1
            else:
                new_motor_state = 1

        return new_motor_state

    def tumble(self, n_flagella, PMF):
        """
        thrust scales with log(n_flagella) because only the thickness of the bundle is affected
        """
        thrust = (
            self.parameters["tumble_scaling"]
            * PMF
            * self.parameters["flagellum_thrust"]
            * math.log(n_flagella + 1)
        )
        torque = random.normalvariate(0, self.parameters["tumble_jitter"])
        return [thrust, torque]

    def run(self, n_flagella, PMF):
        """
        thrust scales with log(n_flagella) because only the thickness of the bundle is affected
        """
        thrust = (
            self.parameters["run_scaling"]
            * PMF
            * self.parameters["flagellum_thrust"]
            * math.log(n_flagella + 1)
        )
        torque = 0.0
        return [thrust, torque]


# test functions
def get_chemoreceptor_activity_timeline(
    total_time=2, time_step=0.01, rate=1.0, initial_value=1.0 / 3.0
):
    val = copy.copy(initial_value)
    timeline = [(0, {("internal", "chemoreceptor_activity"): initial_value})]
    t = 0
    while t < total_time:
        val += random.choice((-1, 1)) * rate * time_step
        if val < 0:
            val = 0
        if val > 1:
            val = 1
        timeline.append((t, {("internal", "chemoreceptor_activity"): val}))
        t += time_step
    return timeline


def test_variable_flagella(out_dir="out", return_data=False):
    total_time = 30
    time_step = 0.01
    initial_flagella = 2

    # make timeline
    timeline = get_chemoreceptor_activity_timeline(
        total_time=total_time,
        time_step=time_step,
        rate=2.0,
    )
    timeline_flagella = [
        (10, {("internal_counts", "flagella"): initial_flagella + 1}),
        (20, {("internal_counts", "flagella"): initial_flagella + 2}),
    ]
    timeline.extend(timeline_flagella)

    # run simulation
    process_config = {"n_flagella": initial_flagella}
    process = FlagellaMotor(process_config)
    settings = {
        "return_raw_data": True,
        "timeline": {"timeline": timeline, "time_step": time_step},
    }
    raw_data = simulate_process(process, settings)

    if return_data:
        return raw_data


def plot_activity(
    data,
    settings=None,
    out_dir="out",
    filename="motor_control",
):
    if not settings:
        settings = {}
    if settings.get("data_type") == "timeseries":
        timeseries = data
    else:
        timeseries = timeseries_from_data(data)
    aspect_ratio = settings.get("aspect_ratio", 1)
    fontsize = settings.get("fontsize", 12)

    CheY_vec = timeseries["internal"]["CheY"]
    CheY_P_vec = timeseries["internal"]["CheY_P"]
    motile_state_vec = timeseries["internal"]["motile_state"]
    thrust_vec = timeseries["boundary"]["thrust"]
    flagella_activity = timeseries.get("flagella", {})
    time_vec = timeseries["time"]

    # make flagella activity grid
    flagella_ids = list(flagella_activity.keys())
    flagella_indexes = {}
    next_index = 0
    activity_grid = np.zeros((len(flagella_ids), len(time_vec)))
    total_CW = np.zeros((len(time_vec)))
    for time_index, (time, time_data) in enumerate(data.items()):
        time_flagella = time_data.get("flagella", {})

        for flagella_id, rotation_states in time_flagella.items():
            # get flagella_index by order of appearance
            if flagella_id not in flagella_indexes:
                flagella_indexes[flagella_id] = next_index
                next_index += 1
            flagella_index = flagella_indexes[flagella_id]

            modified_rotation_state = 0
            CW_rotation_state = 0
            if rotation_states == -1:
                modified_rotation_state = 1
            elif rotation_states == 1:
                modified_rotation_state = 2
                CW_rotation_state = 1

            activity_grid[flagella_index, time_index] = modified_rotation_state
            total_CW += np.array(CW_rotation_state)

    # grid for cell state
    motile_state_grid = np.zeros((1, len(time_vec)))
    motile_state_grid[0, :] = motile_state_vec

    # set up colormaps
    # cell motile state
    cmap1 = colors.ListedColormap(["steelblue", "lightgray", "darkorange"])
    bounds1 = [-1, -1 / 3, 1 / 3, 1]
    norm1 = colors.BoundaryNorm(bounds1, cmap1.N)
    motile_legend_elements = [
        Patch(facecolor="steelblue", edgecolor="k", label="Run"),
        Patch(facecolor="darkorange", edgecolor="k", label="Tumble"),
        Patch(facecolor="lightgray", edgecolor="k", label="N/A"),
    ]

    # rotational state
    cmap2 = colors.ListedColormap(["lightgray", "steelblue", "darkorange"])
    bounds2 = [0, 0.5, 1.5, 2]
    norm2 = colors.BoundaryNorm(bounds2, cmap2.N)
    rotational_legend_elements = [
        Patch(facecolor="steelblue", edgecolor="k", label="CCW"),
        Patch(facecolor="darkorange", edgecolor="k", label="CW"),
        Patch(facecolor="lightgray", edgecolor="k", label="N/A"),
    ]

    # plot results
    cols = 1
    rows = 4
    width = 3
    height = width / aspect_ratio
    plt.figure(figsize=(width, height))
    plt.rc("font", size=fontsize)

    # define subplots
    ax1 = plt.subplot(rows, cols, 1)
    ax3 = plt.subplot(rows, cols, 2)
    ax4 = plt.subplot(rows, cols, 3)
    ax5 = plt.subplot(rows, cols, 4)

    # plot Che-P state
    ax1.plot(time_vec, CheY_vec, label="CheY")
    ax1.plot(time_vec, CheY_P_vec, label="CheY_P")
    ax1.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax1.set_xticks([])
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.set_xlim(time_vec[0], time_vec[-1])
    ax1.set_ylabel("concentration \n (uM)")

    # plot flagella states in a grid
    if len(activity_grid) > 0:
        ax3.imshow(
            activity_grid,
            interpolation="nearest",
            aspect="auto",
            cmap=cmap2,
            norm=norm2,
            # extent=[-1,1,-1,1]
            extent=[time_vec[0], time_vec[-1], len(flagella_ids) + 0.5, 0.5],
        )
        plt.locator_params(axis="y", nbins=len(flagella_ids))
        ax3.set_yticks(list(range(1, len(flagella_ids) + 1)))
        ax3.set_xticks([])
        ax3.set_ylabel("flagella #")
        ax3.tick_params(axis="both", labelsize=fontsize)

        # legend
        ax3.legend(
            title="activity",
            handles=rotational_legend_elements,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )
    else:
        # no flagella
        ax3.set_axis_off()

    # plot cell motile state
    ax4.imshow(
        motile_state_grid,
        interpolation="nearest",
        aspect="auto",
        cmap=cmap1,
        norm=norm1,
        extent=[time_vec[0], time_vec[-1], 0, 1],
    )
    ax4.set_yticks([])
    ax4.set_xticks([])

    # legend
    ax4.legend(
        title="motile state",
        handles=motile_legend_elements,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    # plot motor thrust
    ax5.plot(time_vec, thrust_vec)
    ax5.set_xlim(time_vec[0], time_vec[-1])
    ax5.spines["right"].set_visible(False)
    ax5.spines["top"].set_visible(False)
    ax5.set_ylabel("thrust (pN)")
    ax5.set_xlabel("time (sec)")

    # save figure
    fig_path = os.path.join(out_dir, filename)
    plt.subplots_adjust(wspace=0.7, hspace=0.3)
    plt.savefig(fig_path + ".png", bbox_inches="tight")


def plot_motor_PMF(output, out_dir="out", figname="motor_PMF"):
    # motile_state = output["motile_state"]
    thrust = output["thrust"]
    # motile_torque = output["motile_torque"]
    PMF = output["PMF"]

    # plot results
    cols = 1
    rows = 1
    plt.figure(figsize=(10 * cols, 2 * rows))

    # define subplots
    ax1 = plt.subplot(rows, cols, 1)

    # plot motile_state
    ax1.plot(PMF, thrust)
    ax1.set_xlabel("PMF (mV)")
    ax1.set_ylabel("force")

    # save figure
    fig_path = os.path.join(out_dir, figname)
    plt.subplots_adjust(wspace=0.7, hspace=0.3)
    plt.savefig(fig_path + ".png", bbox_inches="tight")


def plot_signal_transduction(
    timeseries, plot_config, out_dir="out", filename="signal_transduction"
):
    if plot_config.get("ligand_id"):
        ligand_id = plot_config["ligand_id"]
        ligand = {ligand_id: timeseries["boundary"]["external"][ligand_id]}
    else:
        ligand = timeseries["boundary"]["external"]
    aspect_ratio = plot_config.get("aspect_ratio", 1)
    fontsize = plot_config.get("fontsize", 12)
    chemoreceptor_activity = timeseries["internal"]["chemoreceptor_activity"]
    CheY_P = timeseries["internal"]["CheY_P"]
    motile_state = timeseries["internal"]["motile_state"]
    time_vec = timeseries["time"]

    # grid for cell state
    motile_state_grid = np.zeros((1, len(time_vec)))
    motile_state_grid[0, :] = motile_state

    # set up colormaps
    # cell motile state
    cmap1 = colors.ListedColormap(["steelblue", "lightgray", "darkorange"])
    bounds1 = [-1, -1 / 3, 1 / 3, 1]
    norm1 = colors.BoundaryNorm(bounds1, cmap1.N)
    motile_legend_elements = [
        Patch(facecolor="steelblue", edgecolor="k", label="Run"),
        Patch(facecolor="darkorange", edgecolor="k", label="Tumble"),
        Patch(facecolor="lightgray", edgecolor="k", label="N/A"),
    ]

    # plot results
    cols = 1
    rows = 4
    width = 3
    height = width / aspect_ratio
    plt.figure(figsize=(width, height))
    plt.rc("font", size=12)

    ax1 = plt.subplot(rows, cols, 1)
    ax2 = plt.subplot(rows, cols, 2)
    ax3 = plt.subplot(rows, cols, 3)
    ax4 = plt.subplot(rows, cols, 4)

    for ligand_id, ligand_vec in ligand.items():
        ax1.plot(time_vec, ligand_vec, "steelblue")
    ax2.plot(time_vec, chemoreceptor_activity, "steelblue")
    ax3.plot(time_vec, CheY_P, "steelblue")

    # plot cell motile state
    ax4.imshow(
        motile_state_grid,
        interpolation="nearest",
        aspect="auto",
        cmap=cmap1,
        norm=norm1,
        extent=[time_vec[0], time_vec[-1], 0, 1],
    )

    ax1.set_xticklabels([])
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.tick_params(right=False, top=False)
    ax1.set_ylabel("external ligand \n (mM) ", fontsize=fontsize)

    ax2.set_xticklabels([])
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(right=False, top=False)
    ax2.set_ylabel("cluster activity \n P(on)", fontsize=fontsize)

    ax3.set_xticklabels([])
    ax3.spines["right"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax3.tick_params(right=False, top=False)
    ax3.set_ylabel("CheY-P", fontsize=fontsize)

    ax4.spines["right"].set_visible(False)
    ax4.spines["top"].set_visible(False)
    ax4.tick_params(right=False, top=False)
    ax4.set_xlabel("time (s)", fontsize=fontsize)
    ax4.set_ylabel("motile state", fontsize=fontsize)

    # legend
    ax4.legend(
        title="motile state",
        handles=motile_legend_elements,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    fig_path = os.path.join(out_dir, filename)
    plt.subplots_adjust(wspace=0.7, hspace=0.3)
    plt.savefig(fig_path + ".png", bbox_inches="tight")


def main():
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data = test_variable_flagella(out_dir, return_data=True)

    # plot
    plot_settings = {}
    timeseries = timeseries_from_data(data)
    plot_simulation_output(timeseries, plot_settings, out_dir)
    plot_activity(data, plot_settings, out_dir)


if __name__ == "__main__":
    main()
