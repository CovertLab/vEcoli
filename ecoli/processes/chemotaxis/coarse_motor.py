"""
======================
Coarse Motor Processes
======================
``MotorActivity`` models `E. coli` coarse motor activity, without explicit flagella.

References:
 Based on the model described in:
    `Vladimirov, N., Lovdok, L., Lebiedz, D., & Sourjik, V. (2008).
    Dependence of bacterial chemotaxis on gradient shape and adaptation rate.`
 CheY phosphorylation model from:
    `Kollmann, M., Lovdok, L., Bartholome, K., Timmer, J., & Sourjik, V. (2005).
    Design principles of a bacterial signalling network. Nature.`
 Motor switching model from:
    `Scharf, B. E., Fahrner, K. A., Turner, L., and Berg, H. C. (1998).
    Control of direction of flagellar rotation in bacterial chemotaxis. PNAS.`

An increase of attractant inhibits CheA activity (chemoreceptor_activity),
but subsequent methylation returns CheA activity to its original level.

TODO -- add CheB phosphorylation

"""

import os
import random
import math

import numpy as np
from numpy import linspace

import matplotlib.pyplot as plt

# vivarium-core imports
from vivarium.core.process import Process
from vivarium.core.composition import PROCESS_OUT_DIR


NAME = "coarse_motor"


class MotorActivity(Process):
    """
    Models changes to coarse motor activity, based on chemoreceptor_activity
    and current motor state.

    :term:`Ports`:
        * **internal**: includes variables ``ccw_motor_bias``, ``ccw_to_cw``, ``motile_state``, ``CheY_P``
        * **external**: includes variables ``thrust`` and ``torque``
    """

    name = NAME
    defaults = {
        "time_step": 0.1,
        #  CheY phosphorylation parameters
        # 'k_A': 5.0,  #
        "k_y": 100.0,  # 1/uM/s
        "k_z": 30.0,  # / CheZ,
        "gamma_Y": 0.1,  # rate constant
        "k_s": 0.45,  # scaling coefficient
        "adapt_precision": 3,  # scales CheY_P to cluster activity
        # motor
        "mb_0": 0.65,  # steady state motor bias (Cluzel et al 2000)
        "n_motors": 5,
        "cw_to_ccw": 0.83,  # 1/s (Block1983) motor bias, assumed to be constant
        "ccw_to_cw_leak": 0.25,  # rate of spontaneous transition to tumble
        # parameters for multibody physics
        "tumble_jitter": 120.0,
        # initial state
        "initial_state": {
            "internal": {
                # response regulator proteins
                "CheY_tot": 9.7,  # (uM) (mM) 9.7 uM = 0.0097 mM
                "CheY_P": 0.5,
                "CheZ": 0.01 * 100,  # (uM) phosphatase 100 uM = 0.1 mM
                "CheA": 0.01 * 100,  # (uM) 100 uM = 0.1 mM
                # sensor activity
                "chemoreceptor_activity": 1 / 3,
                # motor activity
                "ccw_motor_bias": 0.5,
                "ccw_to_cw": 0.5,
                "motile_state": 1,  # motile_state 1 for tumble, -1 for run
            },
            "external": {
                "thrust": 0,
                "torque": 0,
            },
        },
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

    def ports_schema(self):
        """create ``internal`` and ``external`` ports"""
        ports = ["internal", "external"]
        schema = {port: {} for port in ports}

        # external
        for state, default in self.parameters["initial_state"]["external"].items():
            schema["external"][state] = {
                "_default": default,
                "_emit": True,
                "_updater": "set",
            }

        # internal
        set_and_emit = ["ccw_motor_bias", "ccw_to_cw", "motile_state", "CheA", "CheY_P"]
        for state, default in self.parameters["initial_state"]["internal"].items():
            schema["internal"][state] = {"_default": default}
            if state in set_and_emit:
                schema["internal"][state].update({"_emit": True, "_updater": "set"})

        return schema

    def next_update(self, timestep, states):
        internal = states["internal"]
        P_on = internal["chemoreceptor_activity"]
        motile_state_current = internal["motile_state"]

        # parameters
        adapt_precision = self.parameters["adapt_precision"]
        k_y = self.parameters["k_y"]
        k_s = self.parameters["k_s"]
        k_z = self.parameters["k_z"]
        gamma_Y = self.parameters["gamma_Y"]
        mb_0 = self.parameters["mb_0"]
        cw_to_ccw = self.parameters["cw_to_ccw"]

        ## Kinase activity
        # relative steady-state concentration of phosphorylated CheY.
        CheY_P = (
            adapt_precision * k_y * k_s * P_on / (k_y * k_s * P_on + k_z + gamma_Y)
        )  # CheZ cancels out of k_z

        ## Motor switching
        # CCW corresponds to run. CW corresponds to tumble
        ccw_motor_bias = mb_0 / (CheY_P * (1 - mb_0) + mb_0)  # (1/s)
        ccw_to_cw = cw_to_ccw * (1 / ccw_motor_bias - 1)  # (1/s)
        # don't let ccw_to_cw get under leak value
        if ccw_to_cw < self.parameters["ccw_to_cw_leak"]:
            ccw_to_cw = self.parameters["ccw_to_cw_leak"]

        if motile_state_current == -1:  # -1 for run
            # switch to tumble (cw)?
            rate = -math.log(1 - ccw_to_cw)  # rate for probability function of time
            prob_switch = 1 - math.exp(-rate * timestep)
            if np.random.random(1)[0] <= prob_switch:
                motile_state = 1
                thrust, torque = tumble(self.parameters["tumble_jitter"])
            else:
                motile_state = -1
                thrust, torque = run()

        elif motile_state_current == 1:  # 1 for tumble
            # switch to run (ccw)?
            rate = -math.log(1 - cw_to_ccw)  # rate for probability function of time
            prob_switch = 1 - math.exp(-rate * timestep)
            if np.random.random(1)[0] <= prob_switch:
                motile_state = -1
                [thrust, torque] = run()
            else:
                motile_state = 1
                [thrust, torque] = tumble()

        return {
            "internal": {
                "ccw_motor_bias": ccw_motor_bias,
                "ccw_to_cw": ccw_to_cw,
                "motile_state": motile_state,
                "CheY_P": CheY_P,
            },
            "external": {"thrust": thrust, "torque": torque},
        }


def tumble(tumble_jitter=120.0):
    thrust = 0.2  # (pN)
    torque = random.normalvariate(0, tumble_jitter)
    return [thrust, torque]


def run():
    # average thrust = 0.5 pN according to:
    # Hughes MP & Morgan H. (1999) Measurement of bacterial flagellar thrust by negative dielectrophoresis.
    thrust = 0.5  # (pN)
    torque = 0.0
    return [thrust, torque]


def test_variable_receptor(return_data=False):
    motor = MotorActivity()
    state = motor.default_state()
    timestep = 1
    chemoreceptor_activity = linspace(0.0, 1.0, 501).tolist()
    CheY_P_vec = []
    ccw_motor_bias_vec = []
    ccw_to_cw_vec = []
    motile_state_vec = []
    for activity in chemoreceptor_activity:
        state["internal"]["chemoreceptor_activity"] = activity
        update = motor.next_update(timestep, state)
        CheY_P = update["internal"]["CheY_P"]
        ccw_motor_bias = update["internal"]["ccw_motor_bias"]
        ccw_to_cw = update["internal"]["ccw_to_cw"]
        motile_state = update["internal"]["motile_state"]

        CheY_P_vec.append(CheY_P)
        ccw_motor_bias_vec.append(ccw_motor_bias)
        ccw_to_cw_vec.append(ccw_to_cw)
        motile_state_vec.append(motile_state)

    # check ccw_to_cw bias is strictly increasing with increased receptor activity
    assert all(i <= j for i, j in zip(ccw_to_cw_vec, ccw_to_cw_vec[1:]))

    if return_data:
        return {
            "chemoreceptor_activity": chemoreceptor_activity,
            "CheY_P": CheY_P_vec,
            "ccw_motor_bias": ccw_motor_bias_vec,
            "ccw_to_cw": ccw_to_cw_vec,
            "motile_state": motile_state_vec,
        }


def plot_variable_receptor(output, out_dir="out", filename="motor_variable_receptor"):
    receptor_activities = output["chemoreceptor_activity"]
    CheY_P_vec = output["CheY_P"]
    ccw_motor_bias_vec = output["ccw_motor_bias"]
    ccw_to_cw_vec = output["ccw_to_cw"]

    # plot results
    cols = 1
    rows = 2
    plt.figure(figsize=(5 * cols, 2 * rows))

    ax1 = plt.subplot(rows, cols, 1)
    ax2 = plt.subplot(rows, cols, 2)

    ax1.scatter(receptor_activities, CheY_P_vec, c="b")
    ax2.scatter(receptor_activities, ccw_motor_bias_vec, c="b", label="ccw_motor_bias")
    ax2.scatter(receptor_activities, ccw_to_cw_vec, c="g", label="ccw_to_cw")

    ax1.set_xticklabels([])
    ax1.set_ylabel("CheY_P", fontsize=10)
    ax2.set_xlabel("receptor activity \n P(on) ", fontsize=10)
    ax2.set_ylabel("motor bias", fontsize=10)
    ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    fig_path = os.path.join(out_dir, filename)
    plt.subplots_adjust(wspace=0.7, hspace=0.1)
    plt.savefig(fig_path + ".png", bbox_inches="tight")


def main():
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    output2 = test_variable_receptor(return_data=True)
    plot_variable_receptor(output2, out_dir)


if __name__ == "__main__":
    main()
