"""
Steady-state tRNA charging and ppGpp math shared by the simulation and the
ParCa charging fit.

The ppGpp rate law here is the one used by
:py:func:`ecoli.processes.polypeptide_elongation.ppgpp_metabolite_changes`
(RelA activated by ribosomes bound to uncharged tRNA, SpoT synthesis, and SpoT
degradation inhibited by uncharged tRNA). The charging rate law is the one in
:py:func:`ecoli.processes.polypeptide_elongation.dcdt_jit`. All concentrations
are in uM and all rates in uM/s.

The ParCa fit (:py:func:`fit_condition_charging`) inverts these: for a target
ppGpp concentration it finds one scale factor on all synthetases at which the
resulting steady-state charging puts ppGpp synthesis and loss in balance.

# TODO (mia): consider removing steady_state_uncharged, fit_condition_charging,
    ppgpp_net_rate
"""

from typing import Any, Optional

import numpy as np
from scipy.optimize import brentq


def ribosome_elongation_rate(
    charged_trna: np.ndarray,
    uncharged_trna: np.ndarray,
    ribosome_conc: float,
    f: np.ndarray,
    krta: float,
    krtf: float,
    max_elong_rate: float,
) -> float:
    """
    Rate of amino acid incorporation (uM/s) for a given tRNA state, with the
    same competitive binding of charged and uncharged tRNA at the A site as
    the simulation.
    """
    numerator = 1 + np.sum(
        f * (krta / charged_trna + uncharged_trna / charged_trna * krta / krtf)
    )
    return max_elong_rate * ribosome_conc / numerator


def ppgpp_rates(
    uncharged_trna: np.ndarray,
    charged_trna: np.ndarray,
    ribosome_conc: float,
    f: np.ndarray,
    v_rib: float,
    rela_conc: float,
    spot_conc: float,
    ppgpp_conc: float,
    krta: float,
    krtf: float,
    max_elong_rate: float,
    KD_RelA: np.ndarray,
    k_RelA: float,
    k_SpoT_syn: float,
    k_SpoT_deg: float,
    KI_SpoT: np.ndarray,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """
    ppGpp synthesis and degradation rates (uM/s).

    Returns:
        4-element tuple containing

        - **v_rela_syn**: RelA synthesis rate attributed to each amino acid
        - **v_spot_syn**: SpoT synthesis rate
        - **v_deg**: SpoT degradation rate
        - **v_deg_inhibited**: degradation prevented by each amino acid's
          uncharged tRNA
    """
    numerator = 1 + charged_trna / krta + uncharged_trna / krtf
    saturated_charged = charged_trna / krta / numerator
    saturated_uncharged = uncharged_trna / krtf / numerator
    if v_rib == 0:
        ribosome_conc_a_site = f * ribosome_conc
    else:
        ribosome_conc_a_site = f * v_rib / (saturated_charged * max_elong_rate)
    ribosomes_bound_to_uncharged = ribosome_conc_a_site * saturated_uncharged

    # Rare cases with no tRNA give inf or nan: assume ribosomes bind uncharged
    # tRNA in proportion to f if any tRNA of that amino acid is present
    mask = ~np.isfinite(ribosomes_bound_to_uncharged)
    ribosomes_bound_to_uncharged[mask] = (
        ribosome_conc
        * f[mask]
        * np.array(uncharged_trna[mask] + charged_trna[mask] > 0)
    )

    # Fraction of RelA active, with competition between amino acids
    competitive_inhibition = 1 + ribosomes_bound_to_uncharged / KD_RelA
    inhibition_product = np.prod(competitive_inhibition)
    with np.errstate(divide="ignore"):
        frac_rela = 1 / (
            KD_RelA
            / ribosomes_bound_to_uncharged
            * inhibition_product
            / competitive_inhibition
            + 1
        )

    v_rela_syn = k_RelA * rela_conc * frac_rela
    v_spot_syn = k_SpoT_syn * spot_conc
    max_deg = k_SpoT_deg * spot_conc * ppgpp_conc
    fractions = uncharged_trna / KI_SpoT
    v_deg = max_deg / (1 + fractions.sum())
    v_deg_inhibited = (max_deg - v_deg) * fractions / fractions.sum()

    return v_rela_syn, v_spot_syn, v_deg, v_deg_inhibited


def charging_saturation(
    uncharged_trna: np.ndarray,
    aa_conc: np.ndarray,
    KMtf: np.ndarray,
    KMaa: np.ndarray,
) -> np.ndarray:
    """
    Fraction of the maximum charging rate (kS x synthetase) reached at these
    uncharged tRNA and amino acid concentrations (random-order bi-substrate
    rate law, as in the simulation).
    """
    u = uncharged_trna / KMtf
    a = aa_conc / KMaa
    return u * a / (1 + u + a + u * a)


def ppgpp_net_rate(
    uncharged: np.ndarray,
    total_trna: np.ndarray,
    target_ppgpp: float,
    dilution_rate: float,
    ribosome_conc: float,
    f: np.ndarray,
    rela_conc: float,
    spot_conc: float,
    params: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """
    Net ppGpp production (synthesis minus degradation and dilution, uM/s) at
    the target ppGpp concentration for a given uncharged tRNA state.
    """
    charged = total_trna - uncharged
    v_rib = ribosome_elongation_rate(
        charged,
        uncharged,
        ribosome_conc,
        f,
        params["krta"],
        params["krtf"],
        params["max_elong_rate"],
    )
    v_rela, v_spot, v_deg, _ = ppgpp_rates(
        uncharged,
        charged,
        ribosome_conc,
        f,
        v_rib,
        rela_conc,
        spot_conc,
        target_ppgpp,
        params["krta"],
        params["krtf"],
        params["max_elong_rate"],
        params["KD_RelA"],
        params["k_RelA"],
        params["k_SpoT_syn"],
        params["k_SpoT_deg"],
        params["KI_SpoT"],
    )
    net = v_rela.sum() + v_spot - v_deg - dilution_rate * target_ppgpp
    info = {
        "v_rib": v_rib,
        "v_rela_syn": v_rela.sum(),
        "v_spot_syn": v_spot,
        "v_deg": v_deg,
        "spot_inhibition": (uncharged / params["KI_SpoT"]).sum(),
    }
    return net, info


def steady_state_uncharged(
    synthetase_conc: np.ndarray,
    total_trna: np.ndarray,
    aa_conc: np.ndarray,
    ribosome_conc: float,
    f: np.ndarray,
    params: dict[str, Any],
    max_fraction: float = 0.99,
    n_iterations: int = 20,
) -> tuple[np.ndarray, float]:
    """
    Uncharged tRNA at which charging matches use by elongating ribosomes,
    for given synthetase concentrations. With saturation
    sat = a / (KMaa + a) x u / (KMtf + u), charging kS x S x sat equals use
    v_rib x f when u = KMtf x d / (kS x S x a_sat - d), with d = v_rib x f.
    Elongation depends on the tRNA state, so the two are iterated. Amino
    acids whose synthetases cannot keep up are capped at max_fraction
    uncharged.

    Returns:
        uncharged tRNA for each amino acid (uM) and the elongation rate v_rib (uM/s)
    """
    a_sat = aa_conc / (params["KMaa"] + aa_conc)
    capacity = params["kS"] * synthetase_conc * a_sat
    uncharged = np.zeros_like(total_trna)
    v_rib = 0.0
    for _ in range(n_iterations):
        v_rib = ribosome_elongation_rate(
            total_trna - uncharged,
            uncharged,
            ribosome_conc,
            f,
            params["krta"],
            params["krtf"],
            params["max_elong_rate"],
        )
        demand = v_rib * f
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(
                capacity > demand,
                params["KMtf"] * demand / (capacity - demand),
                np.inf,
            )
        new_uncharged = np.fmin(np.nan_to_num(u, nan=0.0), max_fraction * total_trna)
        if np.allclose(new_uncharged, uncharged, rtol=1e-9, atol=1e-12):
            uncharged = new_uncharged
            break
        uncharged = new_uncharged
    return uncharged, v_rib


def fit_condition_charging(
    total_trna: np.ndarray,
    aa_conc: np.ndarray,
    ribosome_conc: float,
    f: np.ndarray,
    rela_conc: float,
    spot_conc: float,
    target_ppgpp: float,
    dilution_rate: float,
    params: dict[str, Any],
    synthetase_conc: np.ndarray,
    scalable: Optional[np.ndarray] = None,
    min_scale: float = 0.05,
    max_scale: float = 20.0,
) -> dict[str, Any]:
    """
    Finds one scale factor on all synthetase concentrations that makes ppGpp
    synthesis balance degradation plus dilution at the target ppGpp
    concentration. Keeping one factor keeps the relative synthetase levels
    that ParCa set from expression data, so charging differences between
    amino acids come from those data and the synthetase kinetics.

    Args:
        total_trna: total (charged + uncharged) tRNA for each amino acid (uM)
        aa_conc: free amino acid concentrations (uM)
        ribosome_conc: active ribosome concentration (uM)
        f: fraction of each amino acid in translation (0 for amino acids
            excluded from charging)
        rela_conc: RelA concentration (uM)
        spot_conc: SpoT concentration (uM)
        target_ppgpp: ppGpp concentration to reach (uM)
        dilution_rate: growth rate, ln(2) / doubling time (1/s)
        params: krta, krtf, max_elong_rate, KD_RelA, k_RelA, k_SpoT_syn
            (already scaled for the medium), k_SpoT_deg, KI_SpoT, kS, KMtf,
            KMaa
        synthetase_conc: current synthetase concentration for each amino
            acid (uM)
        scalable: fraction of each amino acid's synthetase that the scale
            acts on (1 by default). Synthetases made from TUs that are not
            adjusted move less: synthetase x (1 + (scale - 1) x scalable).
        min_scale: lowest scale searched
        max_scale: highest scale searched

    Returns:
        Dictionary with "status" ("ok", "target_below_floor" or
        "target_above_ceiling"), "scale", "uncharged" (uM),
        "uncharged_fraction" (per amino acid), "v_rib" and the ppGpp rate
        terms at the solution. When the target cannot be reached, the
        closest bound is returned. "target_below_floor" means even the
        largest scale leaves too much ppGpp (SpoT synthesis alone is above
        the target).

    The net rate can have more than one root at extreme scales (when most
    tRNA is uncharged, elongation collapses and RelA activation changes
    shape), so the search starts at a scale of 1 and steps outward to the
    nearest sign change.
    """
    if scalable is None:
        scalable = np.ones_like(synthetase_conc)

    def evaluate(scale):
        scaled = synthetase_conc * (1 + (scale - 1) * scalable)
        uncharged, _ = steady_state_uncharged(
            scaled, total_trna, aa_conc, ribosome_conc, f, params
        )
        net, info = ppgpp_net_rate(
            uncharged,
            total_trna,
            target_ppgpp,
            dilution_rate,
            ribosome_conc,
            f,
            rela_conc,
            spot_conc,
            params,
        )
        return net, info, uncharged

    # More synthetase means more charged tRNA and less ppGpp synthesis, so the
    # net rate falls as the scale rises. Step outward from 1 to the nearest
    # sign change.
    step = 1.05
    net_one = evaluate(1.0)[0]
    status = "ok"
    if net_one == 0:
        scale = 1.0
    else:
        direction = step if net_one > 0 else 1 / step
        low, high = 1.0, 1.0 * direction
        bracket = None
        while min_scale <= high <= max_scale:
            if np.sign(evaluate(high)[0]) != np.sign(net_one):
                bracket = (min(low, high), max(low, high))
                break
            low, high = high, high * direction
        if bracket is None:
            status = "target_below_floor" if net_one > 0 else "target_above_ceiling"
            scale = max_scale if net_one > 0 else min_scale
        else:
            scale = brentq(lambda x: evaluate(x)[0], *bracket, xtol=1e-8, rtol=1e-8)

    net, info, uncharged = evaluate(scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.where(total_trna > 0, uncharged / total_trna, 0.0)
    return {
        "status": status,
        "scale": scale,
        "uncharged": uncharged,
        "uncharged_fraction": fraction,
        "net_rate_at_solution": net,
        **info,
    }
