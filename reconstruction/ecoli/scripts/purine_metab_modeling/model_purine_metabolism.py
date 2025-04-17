import os

import math
import numpy as np
import scipy
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "plots")

# kcat's in units of s^-1
KCAT_PURF = 23
KCAT_PURD = 20  ## arbitrarily determined
KCAT_PURN = 133
KCAT_PURT = 37.6
KCAT_PURL = 2.5
KCAT_PURM = 2.3
KCAT_PURK = 52  # Paper also says 52???
KCAT_PURE = 15.5  # Paper says 15.5, but questionable methods bc uses purK and purC as linkers to detect v_init?
KCAT_PURE_REV = 15.6
KCAT_PURC = 20  ## arbitrarily determined
KCAT_PURB = 20  ## arbitrarily determined
KCAT_PURH = 20  ## arbitrarily determined
KCAT_PURH_REV = 20  ## arbitrarily determined
KCAT_ADK = 360
KCAT_GMK = 360  ## arbitraily determined to be equal to KCAT_ADK

# Miscellaneous constants
KDEG_NCAIR = 0  # 0.046 # units of s^-1, i.e. half-life is 15s # TODO: put this degradation back in
# Currently, it accounts for 5-10% of loss of purK flux

# KMs in units of mol
KM_PURF_PRPP = 60e-6
KM_PURF_GLN = 2.1e-3
# KI_COMP_PURF_IMP = 200e-6 # Made up, roughly determined to be between that of AMP with no GMP, and AMP with GMP.
KI_PURF_AMP = 1000e-6
KI_PURF_GMP = 220e-6
KI_PURF_GMP_WITH_AMP = 64e-6

KM_PURD_PRA = 70e-6
KM_PURD_ATP = 170e-6
KM_PURD_GLY = 270e-6

KM_PURT_GAR = 10.1e-6
KM_PURT_ATP = 45e-6
KM_PURT_FORMATE = 319e-6

KM_PURN_GAR = 12.1e-6
KM_PURN_N10FTHF = 84.8e-6  # Reversible maybe??

KM_PURL_GLN = 64e-6
KM_PURL_ATP = 51e-6
KM_PURL_FGAR = 30e-6

KM_PURM_FGAM = 27e-6
KM_PURM_ATP = 70e-6

KM_PURK_AIR = 66e-6
KM_PURK_ATP = 48e-6

KM_PURE_REV = 22.6e-6
KEQ_PURE = 3.56

KM_PURC_ATP = 40e-6
KM_PURC_ASP = 1300e-6
KM_PURC_CAIR = 36e-6

KM_PURB_SAICAR = 20e-6  ## arbitrarily determined

KM_PURH_REV = 3e-6  ## arbitrarily determined
KEQ_PURH = 100  ## arbitrarily determined

KM_ADK_ATP = 50e-6
KM_ADK_AMP = 40e-6
KM_ADK_ADP = 92e-6
KEQ_ADK = 1.5  # Standard Gibbs free energy is ~-0.256 kcal/mol

KM_GMK_GMP = 40e-6  ## arbitrarily determined to be equal to KM_ADK_AMP
KM_GMK_ATP = 40e-6  # from paper
KM_GMK_ADP = 92e-6  ## arbitrarily determined to be equal to KM_ADK_ADP
KM_GMK_GDP = 92e-6  ## arbitrarily determined to be equal to KM_ADK_GDP
KEQ_GMK = 3.58  # Standard Gibbs free energy is ~-0.7858

NUCLS_PER_RIBOSOME = 4566

# Metabolomics/protein concentrations data, in units of mol, at minimal media
C_IMP = 2.72e-4
C_PRPP = 2.58e-4
C_GLN = 3.81e-3
C_PURF = 1.3 * 1.06e-6  # arbitrarily put to 4x EcoCyc concentration

C_GLY = 4.32e-4  # NOTE: from Lempp, different from others, but Lempp should also be minimal media
C_ATP = 9.6e-3
C_PURD = 1.18e-6

C_FORMATE = KM_PURT_FORMATE * 100
C_N10FTHF = KM_PURN_N10FTHF * 100
C_PURN = 7.71e-7
C_PURT = 5.19e-7

C_PURL = (
    8 * 7.17e-7
)  # At around 7x WCM-predicted concentration, enables just about enough Vmax to match supply.
# Proteomic data shows 4x WCM-predicted counts (NOT concentration, not sure about proteomic data cell size).

C_PURM = (
    4 * 1.88e-6
)  # At around 3x WCM-predicted monomer concentration, enables just about enough Vmax to match supply.
# Proteomic data shows 1.5x WCM-predicted counts

C_PURK = (
    4 * 6.91e-7
)  # Is a dimer; At around 3.3x WCM-predicted monomer concentration, enables just about enough Vmax to match supply
C_PURE = 1.17e-6  # Is an octamer

C_PURC = 5.29e-6
C_ASP = 4.23e-3

C_PURB = 2.01e-6
C_PURH = 1.00e-6  # TODO: check what ecocyc says

C_ADK = 7.20e-6
C_AMP = 2.80e-4
C_ADP = 5.60e-4

C_GMK = 7.93e-7
C_GMP = 2.40e-5
C_GDP = 6.80e-4

C_RIBOSOME = 2.39e-5


def minimal_oscillations_model(
    y0,
    plot_name=None,
    total_time=10000,
    timestep=0.5,
    use_rate=5.0,
    e1=1.0,
    e2=1.0,
    e3=1.0,
    kcat_e1=10.0,
    kcat_e2=10.0,
    kcat_e3=10.0,
    Km_e1=10.0,
    Km_e2=10.0,
    Km_e3=10.0,
    Ki_e1=10.0,
    Km_p=0.1,
    reversible=False,
):
    # In this minimal model, s1 is turned by e1 into s2, which is turned by e2 into s3, which is turned
    # by e3 into p. The use rate of p is use_rate. p allosterically inhibits e1.
    # We first test the case where, if the rates of e2 and e3 are never limiting, then it can stably
    # reach steady state with very few oscillations. Yes!
    # Now we test the case where, if the rates of e2 can be limiting but
    # now lower than the use_rate, it can still reach steady state
    # but with more damped oscillations.
    # We are ignoring dilution rates for now.

    # Note 1: so it appears that, if only one enzyme is rate-limiting
    # (or, maybe saying that if the rate-limiting contribution of both enzymes is not enough,
    # and that just one enzyme rate-limiting is typically not enough) then even very large changes to
    # beginning concentration will be damped.
    # Note 2: on the other hand, if two enzymes are rate-limiting to a sufficient degree,
    # then even minute differences in initial rate things will be magnified to reach regular oscillations.
    # Note 3: so a major factor here is whether p_use depends on p. If p_use
    # is itself able to introduce some sort of negative feedback on to the levels of p,
    # then clearly the oscillations will dampen out and be stable.
    # Note 4: so what happens in the actual cell? IMP is made into AMP and GMP, which are
    # reversibly converted to ADP and GDP, and then ATP and GTP, which are used by the cell.
    # AMP and GMP inhibit purF. Now what happens if AMP and GMP levels are high?
    # Note 5: so it seems that the rate of RNAP elongation does indeed depend on NTP levels,
    # but perhaps the Km is quite low relative to cellular NTP concentrations (which would make sense perhaps)?
    # TODO: are there other possible exit-side mechanisms of feedback dependence on AMP/GMP levels?
    # Note 6: also remember, in the cell you have more than 2 steps in the pathway, all of which
    # might have enzyme levels that are greater than what's needed but not too much greater?? And the logic is,
    # if splitting them into separate operons enables less proteins made to achieve the same "error rate"
    # of oscillations or some other negative consequence, then that's fine?
    # TODO: but, if there's minimal dependence on AMP/GMP levels, what's bad about oscillations in AMP/GMP
    # levels then? oscillations in ATP/GTP levels maybe?
    # Note 7: focusing on oscillations because, if just one enzyme is rate-limiting below the
    # demand rate, that's obviously bad bc the AMP/GMP levels will start to drop, but in that case
    # it's not immediately clear why having one or two enzymes being rate-limiting makes a huge difference?
    # I suppose there's more of a delay when returning back to normal?
    # But, regarding the oscillations, it seems that having one or two enzymes be rate-limiting from the
    # supply side could make a difference.
    # Although, the two are obviously connected, like something that becomes rate-limiting demand-side,
    # at some point may also become rate-limiting supply side and there's still the substrate build-up??
    # TODO: think about this part!
    # TODO: This all also depends on the magnitude of supply-side variation due to AMP/GMP level
    # changes; get some quantitative estimate of this from the parameters from literature? And also
    # of course this is related perhaps to the reduced oscillation from reduced sensitivty of the first enzyme
    # to the feedback inhibitors?

    # Note 8: TODO: so make a mathematical model with the following features. Enzyme 1, with a constant
    # substrate concentration and some degree of feedback inhibition. Enzyme 2, no regulation, MM.
    # Enzyme 3, no regulation, MM. Product usage, dependent on product level with some MM.
    # The question is, as concentrations of enzyme 2 and 3 change, when are oscillations damped,
    # and when are they undamped? How does this depend on all the other parameters in the system?
    # Also, what happens when enzyme 1 concentration changes (I suppose that just changes the steady-state
    # product level, assuming Vmax is still enough, but doesn't really affect enzymes 2 or 3 since it
    # can still achieve the required flux for demand, and so should be similar to starting from a different
    # product concentration?
    # Although it will change the V that it achieves with different inhibition, that is the sensitivity
    # to product inhibition, so will affect the dynamics in this way.)

    # Note 9: ok actually, since adk and gmk are reversible, it seems that there would probably be no oscillations.
    # Ok, so maybe move on from this idea! :)

    # Presumably, more ADP and GDP will be made, leading to more ATP and GTP through reversible reactions,
    # but the rate of transcription initiation probably doesn't depend much on this and
    # probably not the rate of elongation either? unless the rate of elongation does?
    # in that case, more ATP and GTP will be used, which lowers the reversible steady state for a lower
    # level of AMP and GMP, and also with allosteric inhibition, this leads to lower levels of AMP
    # and GMP back to normal.
    # On the other hand, if the elongation rate doesn't increase, then there will be a steady state
    # of higher AMP, ADP, ATP (probably just higher AMP since ATP doesn't want to change too much?)
    #

    def reversible_MM_eq(s, p, Km_f, kcat_f, kcat_r, Keq):
        s = max(s, 0)
        p = max(p, 0)
        # Km_f = kcat_f * Km_r / (kcat_r * Keq)
        Km_r = Km_f * (kcat_r * Keq) / kcat_f

        # maximum rate given very high s and very low p, is s*(kcat_f/Km_f) / (s/Km_f) = kcat_f
        return ((kcat_f / Km_f) * s - (kcat_r / Km_r) * p) / (
            (s / Km_f) + (p / Km_r) + 1
        )

    def MM_eq(x, Km, i_comp=None, Ki_comp=None):
        x = max(x, 0)
        if i_comp:
            i_comp = max(i_comp, 0)
            return x / (x + Km * (1 + i_comp / Ki_comp))
        return x / (x + Km)

    def kinetics_model(y, t, reversible=reversible, return_enzyme_fluxes=False):
        s1 = y[0]
        s2 = y[1]
        s3 = y[2]
        p = y[3]

        if reversible:
            v_e1 = (
                kcat_e1 * e1 * MM_eq(s1, Km_e1, p, Ki_e1)
            )  # TODO: make this reversible too
            v_e2 = e2 * reversible_MM_eq(s2, s3, Km_e2, kcat_e2, kcat_e2 / 10, 100)
            v_e3 = e3 * reversible_MM_eq(s3, p, Km_e3, kcat_e3, kcat_e3 / 10, 100)
        else:
            v_e1 = kcat_e1 * e1 * MM_eq(s1, Km_e1, p, Ki_e1)
            v_e2 = kcat_e2 * e2 * MM_eq(s2, Km_e2)
            v_e3 = kcat_e3 * e3 * MM_eq(s3, Km_e3)

        p_use = use_rate * p / (Km_p + p)

        ds1dt = 0
        ds2dt = v_e1 - v_e2
        ds3dt = v_e2 - v_e3
        dpdt = v_e3 - p_use

        return [ds1dt, ds2dt, ds3dt, dpdt]

    time = np.linspace(0, total_time, num=int(total_time / timestep) + 1)
    sol = scipy.integrate.odeint(kinetics_model, y0, time)

    if plot_name is not None:
        fig, axs = plt.subplots(len(y0), figsize=(5, 5 * len(y0)))
        for i, concs in enumerate(sol.T):
            axs[0].plot(time, concs, label=i)
            axs[0].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, plot_name))
        plt.close("all")

    ps = sol[-1000:-1, -1]
    return np.max(np.abs(ps - 20))


def oscs_parameter_search(
    e2_values, e3_values, plot_name, init_p=18, Km_p=0.1, e1=1.0, reversible=False
):
    # Note 1: If e2 or e3 is less than the level required to meet demand, then it doesn't work (clearly). Otherwise,
    # it roughly looks like (though not exactly?) that if e2 and e3 are BOTH less than some level, then there's oscillations.
    # Note 2: the magnitude of these oscillations seems to decrease gradually as e2 and e3 get larger
    # Note 3: it seems that, if a certain level of e2 or e3 produces oscillations, it will produce these
    # oscillations (with the same magnitude) regardless of the initial deviation of p from steady-state. So
    # this suggests that the steady-state becomes an unstable steady-state?
    # Note 4: Incresaing the reponsitivty of the demand rate to product level i.e. bigger Km_p), decreases the
    # range of oscillatory e2 or e3. In our simple model here, the oscillations disappear around Km_p=1.
    # Note 5: If you change the amount of e1, while adjusting Ki_e1 so that at steady-state, the same flux goes
    # through e1, then with lower amount sof e1, it seems that there's less oscillations (i.e., the upper threshold for
    # oscillations gets lower), while at higher e1, there's more oscillations (i.e., the upper threshold for oscillations
    # gets higher) and the frequency might change too? TODO: check how frequency and amplitudes change.
    # When e1 becomes quite close to the threshold for not meeting demand, the upper threshold for oscillations goes quite low too.
    # Note 6: A higher ammount of e1 (assuming same flux as before) also increases the upper threshold of Km_p at which oscillations appear,
    # i.e. it's easier to produce oscillations evenw tih some product-usage rate-dependence)
    # Note: trying with reversible MM-eq, to see how it effects the oscillations.
    # Note 7: so, having reversible reactions does dampen the oscillations, how much depending on specific parameters. If Keq is large
    # (i.e., not super reversible), or if kcat_rev is large (TODO: why?), then there's more oscillations. Right now, for Keq=1000,
    # kcat_r = kcat_f/10, it's roughly the same as irreversible. If Keq=10, or if Keq=100 and kcat_r = kcat_f/100, then almost
    # no oscillations.

    # TODO: why is it that with higher e1 (assuming the same flux), there are more oscillations?
    # TODO: vary other parameters to see if anything interesting happens? all that's left is use_rate, Km_e1, Km_e2, and Km_e3.
    # TODO: what is the overall conclusion? think biologically, and think skeptically! It is possible that this isn't relevant?
    # Ask Markus too
    # TODO: other types of inhibition (non-competitive, etc.)?
    # TODO: combine with AMP, GMP, etc. Also, what would be the physiological effect of oscillations? oscillating ATP/GTP??
    # Might 2 genes being low be bad for reasons other than oscillations?

    deviation = np.zeros((len(e2_values), len(e3_values)))
    for i, e2 in enumerate(e2_values):
        for j, e3 in enumerate(e3_values):
            deviation[i, j] = minimal_oscillations_model(
                [30, 50, 50, init_p],
                e1=e1,
                e2=e2,
                e3=e3,
                use_rate=5 * (20 + Km_p) / 20,
                Km_p=Km_p,
                Ki_e1=10 / (3 * e1 - 2),
                reversible=reversible,
            )
            # so we say, with p = 20, s1 = 30, and Ki_e1 = 10, we've that v1 = 5 = 10 * 1 * 30 / (30 + 10 * (1 + 20/Ki_e1)).
            # Now, if p = 20, s1 = 30, we've that 5 = 10 * e1 * 30 / (30 + 10 * (1 + 20 / x)) => 60 * e1 = 30 + 10 * (1+20/x)
            # => 1 + 20/x = (6 * e1 - 3) => x = 20 / (6 * e1 - 4) = 10/(3 * e1 - 2))

    fig, axs = plt.subplots(2, figsize=(5, 10))
    axs[0].imshow(deviation)
    axs[0].set_xlabel("e2 amount")
    axs[0].set_ylabel("e3 amount")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, plot_name))
    plt.close("all")


def metab_model(
    y0,
    plot_name,
    total_time=100000,
    timestep=0.5,
    doubling_time=3000,
    ribosome=C_RIBOSOME,
    gln=C_GLN,
    purF=C_PURF,
    gly=C_GLY,
    ATP=C_ATP,
    purD=C_PURD,
    purN=C_PURN,
    purT=C_PURT,
    formate=C_FORMATE,
    n10fthf=C_N10FTHF,
    purL=C_PURL,
    purM=C_PURM,
    purK=C_PURK,
    purE=C_PURE,
    purC=C_PURC,
    asp=C_ASP,
    purB=C_PURB,
    purH=C_PURH,
    adk=C_ADK,
    gmk=C_GMK,
):
    # y is a vector, right now with pRpp, PRA, IMP concentrations

    dil_rate_constant = math.log(2) / doubling_time

    def purF_MM_eq(pRpp, AMP, GMP):
        conformation_inhib = (pRpp / KM_PURF_PRPP) ** 2.5 / (
            (pRpp / KM_PURF_PRPP) ** 2.5
            + (GMP / KI_PURF_GMP) ** 4.5
            + (AMP / KI_PURF_AMP) ** 2.0 * (GMP / KI_PURF_GMP_WITH_AMP) ** 4.5
        )
        # conformation_inhib = (pRpp/KM_PURF_PRPP)**2.5 / (
        #    (pRpp/KM_PURF_PRPP)**2.5 + (GMP/KI_PURF_GMP)**4.5
        # )
        comp_inhib = pRpp / (pRpp + KM_PURF_PRPP * (1 + (AMP / KI_PURF_AMP) ** 2.0))
        return KCAT_PURF * conformation_inhib * comp_inhib

    def MM_eq(x, Km, i_comp=None, Ki_comp=None):
        x = max(x, 0)
        if i_comp:
            i_comp = max(i_comp, 0)
            return x / (x + Km * (1 + i_comp / Ki_comp))
        return x / (x + Km)

    # TODO: find the two-site model for AMP and GMP binding to purF that the paper referred to?
    def reversible_MM_eq(s, p, Km_r, kcat_f, kcat_r, Keq):
        s = max(s, 0)
        p = max(p, 0)
        Km_f = kcat_f * Km_r / (kcat_r * Keq)

        # maximum rate given very high s and very low p, is s*(kcat_f/Km_f) / (s/Km_f) = kcat_f
        return ((kcat_f / Km_f) * s - (kcat_r / Km_r) * p) / (
            (s / Km_f) + (p / Km_r) + 1
        )

    def adk_reversible_MM_eq(AMP, ATP, ADP):
        # Guessed to be of this form, based on the one substrate one product relationship.
        # TODO: check if this is approximately true?
        capacity = KCAT_ADK
        thermo = ATP * AMP - ADP**2 / KEQ_ADK
        saturation = (
            1
            / (1 + ATP * AMP / (KM_ADK_AMP * KM_ADK_ATP) + ADP**2 / KM_ADK_ADP**2)
            / (KM_ADK_AMP * KM_ADK_ATP)
        )

        return capacity * thermo * saturation

    def gmk_reversible_MM_eq(GMP, ATP, GDP, ADP):
        capacity = KCAT_GMK
        thermo = GMP * ATP - GDP * ADP / KEQ_GMK
        saturation = (
            1
            / (
                1
                + GMP * ATP / (KM_GMK_GMP * KM_GMK_ATP)
                + ADP * GDP / (KM_GMK_ADP * KM_GMK_GDP)
            )
            / (KM_GMK_GMP * KM_GMK_ATP)
        )
        return capacity * thermo * saturation

    # multiply top and bottom by Km_r/kcat_r
    # = (Keq * s - p) / (s*Keq/kcat_f + p/kcat_r + Km_r/kcat_r) = kcat_f * (Keq*s - p) / (s*Keq + p*kcat_f/kcat_r + Keq*Km_f)
    # =

    def kinetics_model(y, t, return_enzyme_fluxes=False):
        pRpp = y[0]
        PRA = y[1]
        GAR = y[2]
        FGAR = y[3]
        FGAM = y[4]
        AIR = y[5]
        NCAIR = y[6]
        CAIR = y[7]
        SAICAR = y[8]
        AICAR = y[9]
        IMP = y[10]
        AMP = y[11]
        GMP = y[12]
        ADP = y[13]
        GDP = y[14]
        ATP = y[15]

        # Substrate 1 changes
        dpRppdt = 0

        # Now, if we wanted to include allosteric regulation by AMP/GMP... (bc steps r reversible after AMP/GMP? would be
        # less responsive if u did IMP). The type and parameters of inhibition by AMP/GMP determines the responsivity of purF.
        # Substrate 11 changes

        # Let's say for now, as a test, just IMP inhibition.
        # For example, we can try competitive inhibition of IMP with pRpp?

        # Note 1: if Vmax_purF, accounting for losses due to dilution and degradation of substrate by the purH step,
        # is still less than IMP_usage, then it won't work.
        # Note 2: if the purD Vmax is not able to go very high, then IMP will go down, and even if purF has a really
        # high rate, this will just increase PRA concentration. Similarly, if let's say purL Vmax is not able to go
        # very high either, then even with purD going at its max, this will just lead to buildup of FGAR until
        # the dilution rate matches the difference between max purD and max purH.
        # Right now, the Vmax of purL is around 1.379e-05.
        # Note 3: purE gets a bit complicated since NCAIR is degradable. If it's not degradable, then just with
        # the fact that purE Vmax can achieve the required value with reasonable concentrations of NCAIR and CAIR,
        # then its fine. If it's degradable, then the flux through that step is necessarily lower than the flux
        # going in, which ends up being ~10% of the total flux due to the relative sizes of the concentration needed
        # to achieve the correct flux through (depending on Kcat and KM). This may decrease it, but if originally the
        # enabled flux is large too, then it's fine I think?
        # Note 4: Oscillations arise when the Vmax of certain steps can't meet the required Vmax that purF provides
        # based on decreased inhibition from IMP. This leads to buildup of intermediates, which then serve as a buffer
        # when the Vmax for purF decreases, creating oscillations.
        # Test 1: these oscillations go away when rate-limiting steps' Vmax's are increased. Yep, for purD and purL
        # Test 2: these oscillations go away when IMP starts higher than it needs to be?
        # Test 3: these oscillations go away when IMP's deviation is low?
        # Idea: so what matters is that, the fluxes of every reaction is able to match the maximum Vmax from purF
        # , in which case any starting level of metabolites will be stable, because there will be no buildup of
        # metabolites?
        # Note 5: Actually, it seems that if only one reaction is possibly limiting, then even if it's
        # very limiting (i.e. just above what is needed to match IMP_usage), the oscillations will fizzle out.
        # BUT, if two reactions are limiting, then they make create seemingly forever oscillations.
        # Note 6: Since purH is reversible, AICAR will tend to increase and decrease with IMP. Because
        # if IMP gets higher, the flux through purH will decrease, making AICAR higher until it meets
        # the flux through purB again. If it wasn't reversible, if IMP gets higher, AICAR won't change,
        # and its level will maintain at whatever it needs to be to match purB flux. So I suppose
        # it being reversible doesn't matter tooo much towards the overall dynamics of the system??
        # TODO: investigate why things are different when C_PURF is high vs low!
        # TODO: make minimal model with 1 or 2 substrates having rate-limiting reactions, to see
        # if the damped oscillations/full oscillations phenomenon reappears!
        # Note 7: Hypothesis part A: no oscillations with 1 substrate because, the higher levels of inhibition
        # is on a shallower part of MM curve than lower levels of inhibition, and so the built-up substrate
        # takes longer to go up than down. But, with 2 substrates, it could be enough to push it over,
        # so that stable oscillations result.
        # Hypothesis part B: Having higher basal levels of inhibition make the MM curve even shallower,
        # so that flux doesn't increase thaat much even when inhibitor levels are low? TODO: test this with
        # minimal models!
        # Hypothesis part C: Having genes NOT together in operons decreases chance that 2 rate-limiting enzymes
        # will have low counts at the same time, which would result in oscillations, whereas 1 having low counts
        # is relatively ok.
        # Hypothesis part D: inhibitor levels may change due to different demand for transcription/replication,
        # or fluctuating levels of purF (there might be important differences between the two sources?)
        # Hypothesis part E: undamped oscillations are bad because then, it oscillates kinda between the original
        # un-ideal IMP level and higher than normal, so it also frequently returns to un-ideal IMP level,
        # which may be disruptive for transcription/replication processes?? maybe? TODO: how to test this?
        # TODO: any other things to consider in this? Also should expand to AMP/GMP inhibition for more complexity!

        # TODO: figure out the source of these oscillations! The below stuff might be wrong??

        # so there's the case of "limiting steps" as well in terms of responsivity maybe? how to mathematically analyze this??
        # TODO: How does this depend on whether IMP demand is really constant, or if it depends on IMP levels as well??
        # HOWEVER, if the needed IMP is higher relative to the KM becasue Vmax is large, then oscillations fizzle out
        # because when IMP is low, the purF flux is "less higher" relatively speaking, and so there's a gradual return of
        # PRA to equilibrium, which gives time for the other metabolites to return to equilibrium as well, and so
        # IMP gradually returns as well,

        # depending on how much IMP is needed to result in the
        # desired flux, in comparison to the KM.
        #
        # That is, if the IMP needed is low relative to the KM (i.e., the Vmax
        # available is near the desired flux), then when IMP is low because purF flux has been low and as a result AICAR
        # levels are low so purH fluxes are low, the purF flux gets larger; when it hits the right concentration, the flux through purF to make PRA is correct.
        #
        # v_purF = KCAT_PURF * purF * gln * pRpp / (
        #    gln*pRpp + pRpp*KM_PURF_GLN + gln*KM_PURF_PRPP
        # )
        # v_purF = KCAT_PURF * purF * MM_eq(gln, KM_PURF_GLN) * MM_eq(pRpp, KM_PURF_PRPP, i_comp=AMP, Ki_comp=KI_COMP_PURF_AMP)
        v_purF = purF * purF_MM_eq(pRpp, AMP, GMP) * MM_eq(gln, KM_PURF_GLN)

        # Substrate 2 changes
        v_purD = (
            KCAT_PURD
            * purD
            * MM_eq(gly, KM_PURD_GLY)
            * MM_eq(ATP, KM_PURD_ATP)
            * MM_eq(PRA, KM_PURD_PRA)
        )

        dPRAdt = v_purF - v_purD - PRA * dil_rate_constant

        # Substrate 3 changes
        v_purN = (
            KCAT_PURN * purN * MM_eq(GAR, KM_PURN_GAR) * MM_eq(n10fthf, KM_PURN_N10FTHF)
        )
        v_purT = (
            KCAT_PURT
            * purT
            * MM_eq(GAR, KM_PURT_GAR)
            * MM_eq(ATP, KM_PURT_ATP)
            * MM_eq(formate, KM_PURT_FORMATE)
        )
        # TODO: look into relative amounts of purN or purT at different growth rates, looking at gene dosage?
        dGARdt = v_purD - v_purN - v_purT - GAR * dil_rate_constant

        # Substrate 4 changes
        v_purL = (
            KCAT_PURL
            * purL
            * MM_eq(FGAR, KM_PURL_FGAR)
            * MM_eq(ATP, KM_PURL_ATP)
            * MM_eq(gln, KM_PURL_GLN)
        )

        dFGARdt = v_purN + v_purT - v_purL - FGAR * dil_rate_constant

        # Substrate 5 changes
        v_purM = KCAT_PURM * purM * MM_eq(FGAM, KM_PURM_FGAM) * MM_eq(ATP, KM_PURM_ATP)

        dFGAMdt = v_purL - v_purM - FGAM * dil_rate_constant

        # Substrate 6 changes
        v_purK = KCAT_PURK * purK * MM_eq(AIR, KM_PURK_AIR) * MM_eq(ATP, KM_PURK_ATP)
        dAIRdt = v_purM - v_purK - AIR * dil_rate_constant

        # Substrate 7 changes
        v_purE = purE * reversible_MM_eq(
            NCAIR, CAIR, KM_PURE_REV, KCAT_PURE, KCAT_PURE_REV, KEQ_PURE
        )

        dNCAIRdt = v_purK - v_purE - NCAIR * (dil_rate_constant + KDEG_NCAIR)

        # TODO: right now K_eq is made to be high, whereas it's probably much lower. Think about implications for this.

        # Substrate 8 changes
        v_purC = (
            KCAT_PURC
            * purC
            * MM_eq(CAIR, KM_PURC_CAIR)
            * MM_eq(asp, KM_PURC_ASP)
            * MM_eq(ATP, KM_PURC_ATP)
        )

        dCAIRdt = v_purE - v_purC - CAIR * dil_rate_constant

        # Substrate 9 changes
        v_purB = KCAT_PURB * purB * MM_eq(SAICAR, KM_PURB_SAICAR)

        dSAICARdt = v_purC - v_purB - SAICAR * dil_rate_constant

        # Substrate 10 changes
        v_purH = purH * reversible_MM_eq(
            AICAR, IMP, KM_PURH_REV, KCAT_PURH, KCAT_PURH_REV, KEQ_PURH
        )

        dAICARdt = v_purB - v_purH - AICAR * dil_rate_constant

        # TODO: add to doc that, there's some requirements on Keq, Km_r, etc. for purH so that it has enough capacity
        # to go forward without making very large AICAR concentrations. e.g. with a favorable (i.e. high) Keq,
        # a smaller Km_r makes more flux bc it also makes Km_f smaller, allowing more flux.
        # Also, kcat_f has to be above some minimum threshold, in order to have enough Vmax capacity for the reaction to meet demand.
        dIMPdt = 0

        # Ignoring guaC for now
        IMP_usage = ribosome * NUCLS_PER_RIBOSOME / 2 * math.log(2) / doubling_time
        ADP_usage = IMP_usage / 2
        GDP_usage = IMP_usage / 2

        v_adk = adk * adk_reversible_MM_eq(AMP, ATP, ADP)
        dAMPdt = (
            v_purH / 2 - v_adk
        )  # assuming half of purH flux goes to AMP # TODO: incorporate histidine branch

        v_gmk = gmk * gmk_reversible_MM_eq(GMP, ATP, GDP, ADP)
        dGMPdt = v_purH / 2 - v_gmk

        # TODO: whether you could use adk, gmk calculated fluxes in cell, and knowing the concentrations of all the
        # metabolites, and the counts of the enzymes, to predict kinetic constants or something like that?
        dADPdt = v_adk - ADP_usage * ADP / (ADP + C_ADP / 200)
        dGDPdt = v_gmk - GDP_usage * GMP / (GMP + C_GMP / 200)
        dATPdt = 0

        if return_enzyme_fluxes:
            return [
                v_purF,
                v_purD,
                v_purN + v_purT,
                v_purL,
                v_purM,
                v_purK,
                v_purE,
                v_purC,
                v_purB,
                v_purH,
                v_adk,
                v_gmk,
                ADP_usage,
                GDP_usage,
            ]
        return [
            dpRppdt,
            dPRAdt,
            dGARdt,
            dFGARdt,
            dFGAMdt,
            dAIRdt,
            dNCAIRdt,
            dCAIRdt,
            dSAICARdt,
            dAICARdt,
            dIMPdt,
            dAMPdt,
            dGMPdt,
            dADPdt,
            dGDPdt,
            dATPdt,
        ]

    time = np.linspace(0, total_time, num=int(total_time / timestep) + 1)
    sol = scipy.integrate.odeint(kinetics_model, y0, time)

    # Find flux rates at each timestep
    IMP_usage = ribosome * NUCLS_PER_RIBOSOME / 2 * math.log(2) / doubling_time
    flux_rates = []
    for i, concs_snapshot in enumerate(sol):
        rates = kinetics_model(
            concs_snapshot, 0, return_enzyme_fluxes=True
        )  # t=0 since no t dependence in the function
        flux_rates.append(rates)
    flux_rates = np.array(flux_rates) / IMP_usage

    # TODO: so I think it comes from buildup of metabolites cuz vmax of some enzymes aren't totally responsive enough
    # to regulation by purF, so then it takes time for metabolites to go down? Test this hypothesis by using different
    # starting metabolite concentrations, different Vmax's, etc.? Any other tests to see or other possible reasons
    # or other possible things that might go wrong/functions that are useful to have?
    # And why, when purF Vmax is lower, this situation doesn't occur as much? I think that makes sense?

    fig, axs = plt.subplots(len(y0), figsize=(5, 5 * len(y0)))
    enzyme_names = [
        "purF",
        "purD",
        "purNT",
        "purL",
        "purM",
        "purK",
        "purE",
        "purC",
        "purB",
        "purH",
        "adk",
        "gmk",
        "ADP_usage",
        "GDP_usage",
    ]
    substrate_names = [
        "pRpp",
        "PRA",
        "GAR",
        "FGAR",
        "FGAM",
        "AIR",
        "NCAIR",
        "CAIR",
        "SAICAR",
        "AICAR",
        "IMP",
        "AMP",
        "GMP",
        "ADP",
        "GDP",
        "ATP",
    ]
    enzyme_setpoints = [
        KM_PURF_PRPP,
        KM_PURD_PRA,
        KM_PURN_GAR,
        KM_PURL_FGAR,
        KM_PURM_FGAM,
        KM_PURK_AIR,
        KCAT_PURE * KM_PURE_REV / (KCAT_PURE_REV * KEQ_PURE),
        KM_PURC_CAIR,
        KM_PURB_SAICAR,
        KCAT_PURH * KM_PURH_REV / (KCAT_PURH_REV * KEQ_PURH),
        C_IMP,
        C_AMP,
        C_GMP,
        C_ADP,
        C_GDP,
        C_ATP,
    ]
    for i, concs in enumerate(sol.T):
        axs[0].plot(time, np.log(concs / enzyme_setpoints[i]), label=substrate_names[i])
        axs[0].legend()
    for i, rates in enumerate(flux_rates.T):
        axs[1].plot(time, rates, label=enzyme_names[i])
        axs[1].set_ylim(-2, 2)
        axs[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, plot_name))
    plt.close("all")

    return sol


if __name__ == "__main__":
    sol = metab_model(
        [C_PRPP, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_IMP, C_AMP, C_GMP, C_ADP, C_GDP, C_ATP],
        "test_adk_with_oscs",
        purD=C_PURD,
        purL=C_PURL,
        total_time=10000,
    )

    # metab_levels = sol[-1, :]

    # sol = minimal_oscillations_model([30, 50, 50, 18], "test_reversible_oscillations", reversible=True,
    #                                 e2=0.6, e3=0.6, use_rate=5 * (20+0.1)/20, Km_p=0.1)
    # oscs_parameter_search(np.linspace(0, 2, 50), np.linspace(0, 2, 50), "oscs_param_search",
    #                       init_p=19.99, Km_p=1, e1=1)
    # oscs_parameter_search(np.linspace(0, 2, 50), np.linspace(0, 2, 50), "oscs_param_search_rev_Keq100",
    #                       init_p=18, Km_p=0.1, e1=1, reversible=True)
    # minimal_oscillations_model([30, 50, 50, 18], "test_oscillations_minimal_undamped_mid_e1",
    #                            e1=2, e2=0.8, e3=0.8, Km_p=0.1, Ki_e1=10/(3*2 - 2))
    # osc = minimal_oscillations_model([30, 50, 50, 18], "test_oscillations_minimal_undamped",
    #                                  e1=1, e2=0.6, e3=0.6,
    #                                  use_rate=5 * (20+0.1)/20)
    # Note: these set of parameters,
    # [30, 50, 50, 20], use_rate = 5 * 20+0.1/20, Km_p=0.1, are at steady state when e1=1, but NOT otherwise.
    # sol = metab_model(metab_levels, "test_2")
