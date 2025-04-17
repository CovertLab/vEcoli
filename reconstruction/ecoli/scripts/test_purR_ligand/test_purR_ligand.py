import os

import numpy as np
import scipy


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "plots")


def fit_pure_kms():
    def binding_fcn(x, K_1, K_2):
        return (2 * x**2 + K_2 * x) / (2 * (x**2 + x * K_2 + K_1 * K_2 / 2))

    def exp_fcn(K, n, h):
        return lambda x: n / (1 + (K / x) ** h)

    hypox_fcn = exp_fcn(9.3, 1.0, 1.5)
    guanine_fcn = exp_fcn(1.5, 1.0, 1.5)
    hypox_x = np.linspace(5.0, 50.0, 20)
    guanine_x = np.linspace(0.6, 9.0, 20)
    hypox_data = np.array([hypox_fcn(x) for x in hypox_x])
    guanine_data = np.array([guanine_fcn(x) for x in guanine_x])

    hypox_params, *_ = scipy.optimize.curve_fit(
        binding_fcn, hypox_x, hypox_data, p0=[9.3, 9.3], method="lm"
    )
    guanine_params, *_ = scipy.optimize.curve_fit(
        binding_fcn, guanine_x, guanine_data, p0=[1.5, 1.5], method="lm"
    )

    K_H1, K_H2 = hypox_params
    K_G1, K_G2 = guanine_params

    def mixed_fcn(h, g, K_H1, K_H2, K_G1, K_G2, K_mix):
        K_ratio = K_H1 * K_H2 / (K_G1 * K_G2)
        numerator = 2 + 1 / h * (K_H2 + K_G2 / K_mix * K_ratio * g)
        denominator = (
            numerator - 1 + 1 / h**2 * (K_H1 * K_H2 / 2 + K_ratio * (K_G2 * g + g**2))
        )
        return numerator / (2 * denominator)

    def mixed_fcn_setter(g, K_H1, K_H2, K_G1, K_G2):
        def mixed_fcn_wrapper(h, K_mix):
            return mixed_fcn(h, g, K_H1, K_H2, K_G1, K_G2, K_mix)

        return mixed_fcn_wrapper

    guanine_levels = [0, 0.5, 3.75, 6.25, 10]

    def fcn_results(K_mix):
        y_results = []
        for g in guanine_levels:
            fcn = mixed_fcn_setter(g, K_H1, K_H2, K_G1, K_G2)
            ys = [fcn(h, K_mix) for h in hypox_x]
            y_results.extend(ys)
        return np.array(y_results)

    hypox_affs = [9.3, 12, 13, 12, 18]
    hypox_sites = [1.0, 1.0, 1.0, 0.78, 0.60]
    hypox_hills = [1.5, 1.3, 1.0, 1.0, 1.0]
    expected_ys = []
    for aff, site, hill in zip(hypox_affs, hypox_sites, hypox_hills):
        fcn = exp_fcn(aff, site, hill)
        expected_ys.extend([fcn(h) for h in hypox_x])
    expected_ys = np.array(expected_ys)

    def loss_fcn(K_mix):
        k = K_mix[0]
        y_results = fcn_results(k)
        return y_results - expected_ys

    optimized_result = scipy.optimize.least_squares(loss_fcn, [9])
    K_mix_optimal = optimized_result["x"]
    K_mix_optimal = K_mix_optimal[0]
    # residuals = optimized_result["fun"]

    fitted_ys = fcn_results(K_mix_optimal)
    # fig, axs = plt.subplots(5, figsize=(5, 20))
    # for i, g in enumerate(guanine_levels):
    #     idxs = range(i*20, (i+1)*20)
    #     axs[i].plot(hypox_x, [fitted_ys[idx] for idx in idxs], label="fit")
    #     axs[i].plot(hypox_x, [expected_ys[idx] for idx in idxs], label="exp")
    #     axs[i].set_title("guanine level: "+str(g))
    #     axs[i].legend()
    #
    # if not os.path.isdir(OUTPUT_DIR):
    #     os.mkdir(OUTPUT_DIR)
    # plt.savefig(os.path.join(OUTPUT_DIR, "purR_binding_fitted"))

    ## Fit both together
    # expected_ys_both = np.concatenate((expected_ys, guanine_data))
    # def fcn_both_results(ks):
    #     k_mix, k_H1, k_H2, k_G1, k_G2 = ks
    #     # Measuring hypoxanthine
    #     y_results = []
    #     for g in guanine_levels:
    #         fcn = mixed_fcn_setter(g, k_H1, k_H2, k_G1, k_G2)
    #         ys = [fcn(h, k_mix) for h in hypox_x]
    #         y_results.extend(ys)
    #
    #     # Measuring guanine
    #     guanine_ys = [binding_fcn(guan, k_G1, k_G2) for guan in guanine_x]
    #     y_results.extend(guanine_ys)
    #     return np.array(y_results)
    #
    # def loss_both_fcn(ks):
    #     return fcn_both_results(ks) - expected_ys_both
    #
    # both_opt = scipy.optimize.least_squares(loss_both_fcn, [9, 9, 9, 2, 2])
    # ks_opt = both_opt['x']
    # k_mix, k_H1, k_H2, k_G1, k_G2 = ks_opt
    # residuals = both_opt['fun']

    # fitted_ys = fcn_both_results(ks_opt)
    # fig, axs = plt.subplots(6, figsize=(5, 20))
    # for i, g in enumerate(guanine_levels):
    #     idxs = range(i*20, (i+1)*20)
    #     axs[i].plot(hypox_x, [fitted_ys[idx] for idx in idxs], label="fit")
    #     axs[i].plot(hypox_x, [expected_ys_both[idx] for idx in idxs], label="exp")
    #     axs[i].set_title("guanine level: "+str(g))
    #     axs[i].legend()
    # axs[5].plot(guanine_x, fitted_ys[-20:], label="fit")
    # axs[5].plot(guanine_x, expected_ys_both[-20:], label="exp")
    # axs[5].set_title("guanine curve")
    # axs[5].legend()
    #
    # if not os.path.isdir(OUTPUT_DIR):
    #     os.mkdir(OUTPUT_DIR)
    # plt.savefig(os.path.join(OUTPUT_DIR, "purR_binding_both_fitted"))

    def fit_exp_fcn(x, K, n, h):
        return n / (1 + (K / x) ** h)

    def fit_hill_curve(xs, data):
        params, *_ = scipy.optimize.curve_fit(fit_exp_fcn, xs, data)
        return params

    def mixed_fcn_guanine(h, g, K_H1, K_H2, K_G1, K_G2, K_mix):
        K_ratio = K_H1 * K_H2 / (K_G1 * K_G2)
        numerator = g * (
            (K_G2 * K_ratio / h**2) + (K_G2 * K_ratio / (K_mix * h))
        ) + g**2 * (2 * K_ratio / h**2)
        denominator = (
            1
            + K_H2 / h
            + K_H1 * K_H2 / (2 * h**2)
            + numerator
            - g**2 * (K_ratio / h**2)
        )
        return numerator / (2 * denominator)

    back_fit_hill_params = []
    for i, g in enumerate(guanine_levels):
        idxs = range(i * 20, (i + 1) * 20)
        params = fit_hill_curve(hypox_x, [fitted_ys[idx] for idx in idxs])
        back_fit_hill_params.append(params)
    params = fit_hill_curve(guanine_x, [binding_fcn(g, K_G1, K_G2) for g in guanine_x])
    back_fit_hill_params.append(params)

    hypox_levels = [12.5, 25.0, 100.0]
    params_result = []
    for hypox in hypox_levels:
        ys = [
            mixed_fcn_guanine(hypox, g, K_H1, K_H2, K_G1, K_G2, K_mix_optimal)
            for g in guanine_x
        ]
        params = fit_hill_curve(guanine_x, ys)
        params_result.append(params)

    def conc_of_species(h, g):
        K_ratio = K_H1 * K_H2 / (K_G1 * K_G2)
        hrh = 1 / (
            1
            + 1 / h * (K_H2 + K_G2 / K_mix_optimal * K_ratio * g)
            + 1 / h**2 * (K_H1 * K_H2 / 2 + K_ratio * (K_G2 * g + g**2))
        )
        grg = K_ratio * (g / h) ** 2 * hrh
        hrg = K_G2 / K_mix_optimal * K_ratio * (g / h) * hrh
        rg = K_G2 * K_ratio * (g / h**2) * hrh
        rh = K_H2 * hrh / h
        r = K_H1 * K_H2 * hrh / (2 * h**2)

        return hrh, grg, hrg, rg, rh, r

    # concs = conc_of_species(0.77, 4.03)


if __name__ == "__main__":
    fit_pure_kms()
