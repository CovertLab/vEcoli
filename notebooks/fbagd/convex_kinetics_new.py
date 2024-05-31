import numpy as np
import cvxpy as cp
import itertools
from scipy.special import logsumexp


class ConvexKineticsNew:

    def __init__(self):
        pass

    def set_up_variables(self, S_matrix, R_matrix, flow_data, dmdt):

        # set up variables
        Sd, Sr = S_matrix, R_matrix
        n_flux_set = flow_data.shape[0]
        n_met, n_rxn = Sd.shape

        # dmdt
        abs_dmdt = np.abs(dmdt)# np.max(np.array([dmdt, np.zeros(dmdt.shape)]), axis=0)
        dmdt_mask = (abs_dmdt != 0).flatten()
        lD = np.log(abs_dmdt[dmdt_mask]).flatten()

        S = np.array(Sd)
        S_b = np.sign(S)  #
        S_s = -np.copy(S_b)  # reverse neg sign
        S_p = np.copy(S_b)
        S_s[S_b > 0] = 0  # zeros products
        S_p[S_b < 0] = 0  # zeros substrates
        S_i = np.copy(np.array(Sr) == -1)  # reaction direction does not matter
        S_a = np.copy(np.array(Sr) == 1)

        # make the code below one line
        S_s_nz, S_p_nz, S_i_nz, S_a_nz = [np.array(ar.nonzero()) for ar in [S_s, S_p, S_i, S_a]]
        S_s_mol, S_p_mol = [np.abs(S)[ar.nonzero()] for ar in [S_s, S_p]]

        # TODO Refactor all the below lines as one liners. Also are they all necessary?
        # first coordinate, e.g. metabolites w nonzero substrate/product coeff across all reactions. also works as substrate indices.
        met_s_nz = S_s_nz[0, :]
        met_p_nz = S_p_nz[0, :]
        met_i_nz = S_i_nz[0, :] if Sr is not None else None
        met_a_nz = S_a_nz[0, :] if Sr is not None else None

        # second coordinate, e.g. reactions indices for those concentrations. works to index substrates as well.
        rxn_s_nz = S_s_nz[1, :]
        rxn_p_nz = S_p_nz[1, :]
        rxn_i_nz = S_i_nz[1, :] if Sr is not None else None
        rxn_a_nz = S_a_nz[1, :] if Sr is not None else None

        # one dim is always 2
        n_Km_s = met_s_nz.shape[0]
        n_Km_p = met_p_nz.shape[0]
        n_Km_i = met_i_nz.shape[0] if Sr is not None else None
        n_Km_a = met_a_nz.shape[0] if Sr is not None else None

        c = cp.Variable([n_met, n_flux_set])
        Km_s, Km_p = cp.Variable(n_Km_s), cp.Variable(n_Km_p)
        Km_i = cp.Variable(n_Km_i) if n_Km_i else None
        Km_a = cp.Variable(n_Km_a) if n_Km_a else None

        cfwd, crev = cp.Variable(n_rxn), cp.Variable(n_rxn)

        # define y vecs
        y_s_t, y_p_t, y_i_t, y_a_t = [], [], [], []
        cd_t = []

        # define Km positions by nonzero S matrix concentrations. Activation is reverse val of inhibition.
        for i in range(n_flux_set):
            # condense the code below into one line
            y_s_t.append(cp.multiply(S_s_mol, c[met_s_nz, i] - Km_s))
            y_p_t.append(cp.multiply(S_p_mol, c[met_p_nz, i] - Km_p))
            y_i_t.append(c[met_i_nz, i] - Km_i if n_Km_i else None)
            y_a_t.append(-(c[met_a_nz, i] - Km_a) if n_Km_a else None)
            cd_t.append(c[dmdt_mask, i] - lD)

        y_s = cp.vstack(y_s_t)
        y_p = cp.vstack(y_p_t)
        y_i = cp.vstack(y_i_t)
        y_a = cp.vstack(y_a_t)
        cd_f = cp.vstack(cd_t)

        # saturation stacks
        y_f_vec = [y_s]
        y_r_vec = [y_p]
        if n_Km_i:
            y_f_vec.append(y_i)
            y_r_vec.append(y_i)
        if n_Km_a:
            y_f_vec.append(y_a)
            y_r_vec.append(y_a)

        y_f = cp.hstack(y_f_vec)
        y_r = cp.hstack(y_r_vec)


        print(f"Number of metabolites: {n_met}, number of reactions: {n_rxn}, number of flux sets: {n_flux_set}",
              f"Number of Km_s: {n_Km_s}, number of Km_p: {n_Km_p}, number of Km_i: {n_Km_i}, number of Km_a: {n_Km_a}",
              f"Number of concentrations: {c.shape}, number of y_f: {y_f.shape}, number of y_r: {y_r.shape}", sep='\n')

        return y_f, y_r, y_s, y_p, y_i, y_a, cd_f, cfwd, crev, c, Km_s, Km_p, Km_i, Km_a, S_s, S_p, S_i, S_a, \
            met_s_nz, met_p_nz, met_i_nz, met_a_nz, rxn_s_nz, rxn_p_nz, rxn_i_nz, rxn_a_nz, \
            n_rxn, n_met, n_flux_set, S_s_nz, S_p_nz, S_s_mol, S_p_mol, S_b, S

    def construct_binding_matrix(self, n_rxn, S_s, S_p, S_i, S_a, Sr, met_s_nz, met_p_nz, met_i_nz, met_a_nz, rxn_s_nz, rxn_p_nz, rxn_i_nz, rxn_a_nz):

        # number of saturation terms for sub, prod
        # make the code below cleaner
        S_s_comb = np.concatenate((S_s, S_i, S_a), axis=0)  if Sr is not None else S_s
        S_p_comb = np.concatenate((S_p, S_i, S_a), axis=0) if Sr is not None else S_p
        n_alpha = np.sum(np.power(2, np.sign(S_s_comb).sum(axis=0)) - 1)
        n_beta = np.sum(np.power(2, np.sign(S_p_comb).sum(axis=0)) - 1)

        # saturation matrix setup, first sub, then inhib, then act.
        if Sr is not None: # TODO make this cleaner
            C_alpha = np.zeros([n_alpha, len(met_s_nz) + len(met_i_nz) + len(met_a_nz)])
            C_beta = np.zeros([n_beta, len(met_p_nz) + len(met_i_nz) + len(met_a_nz)])
        else:
            C_alpha = np.zeros([n_alpha, len(met_s_nz)])
            C_beta = np.zeros([n_beta, len(met_p_nz)])

        # to separate different reactions saturation terms to their individual reaction equations.
        d_alpha, d_beta = np.zeros(n_alpha, dtype=np.int64), np.zeros(n_beta, dtype=np.int64)

        s_idx, p_idx = 0, 0

        for i in range(n_rxn):
            # pick one reaction at a time (get substrate indicies)
            if Sr is not None:
                idx_s_cur_rxn = np.concatenate((rxn_s_nz == i, rxn_i_nz == i, rxn_a_nz == i))
                idx_p_cur_rxn = np.concatenate((rxn_p_nz == i, rxn_i_nz == i, rxn_a_nz == i))
            else:
                idx_s_cur_rxn = rxn_s_nz == i
                idx_p_cur_rxn = rxn_p_nz == i

            # generates all binary permutations minus the first one since that would result in -1
            s_sat_perm = np.array(list(itertools.product([0, 1], repeat=sum(idx_s_cur_rxn))))[1:, :]
            p_sat_perm = np.array(list(itertools.product([0, 1], repeat=sum(idx_p_cur_rxn))))[1:, :]

            r_s, _ = s_sat_perm.shape
            r_p, _ = p_sat_perm.shape

            # replace zeros with saturation matrix
            C_alpha[s_idx:(s_idx+r_s), idx_s_cur_rxn] = s_sat_perm
            d_alpha[s_idx:(s_idx+r_s)] = i

            C_beta[p_idx:(p_idx+r_p), idx_p_cur_rxn] = p_sat_perm
            d_beta[p_idx:(p_idx+r_p)] = i

            s_idx += r_s # add number of rows added.
            p_idx += r_p #

        print(f"Shape of C_alpha: {C_alpha.shape}, shape of C_beta: {C_beta.shape}",
              f"Shape of d_alpha: {d_alpha.shape}, shape of d_beta: {d_beta.shape}", sep='\n')

        return C_alpha, C_beta, d_alpha, d_beta

    def construct_kinetic_objective(self, flow_data, n_flux_set, n_rxn, C_alpha, C_beta, d_alpha, d_beta,
                                    S_s_nz, S_p_nz, S, y_f, y_r, y_s, y_p, cfwd, crev):

        LSE_expr = []
        denom_expr = []

        sign = np.sign(flow_data)
        lvE = np.log(sign * flow_data)

        for j in range(n_flux_set):
            for i in range(n_rxn):
                # sum terms are separate in logsumexp. one per saturation term (row in C_alpha, C_beta)

                n_term_s, n_term_p = np.sum(d_alpha == i), np.sum(d_beta == i)

                Km_s_idx = np.nonzero(S_s_nz[1, :] == i)
                S_s_idx = S_s_nz[0, S_s_nz[1, :] == i]  # negate -1 entries

                Km_p_idx = np.nonzero(S_p_nz[1, :] == i)
                S_p_idx = S_p_nz[0, S_p_nz[1, :] == i]

                expr_sat_alpha = (C_alpha @ cp.vec(y_f[j, :]))[d_alpha == i]
                expr_sat_beta = (C_beta @ cp.vec(y_r[j, :]))[d_beta == i]
                expr_fwd = -S.T[i, S_s_idx] @ cp.vec(y_s[j, Km_s_idx]) + cfwd[i]
                expr_rev = S.T[i, S_p_idx] @ cp.vec(y_p[j, Km_p_idx]) + crev[i]

                if sign[j, i] == 1:
                    denom = expr_fwd
                    num = expr_rev
                else:
                    denom = expr_rev
                    num = expr_fwd

                LSE_expr.append(cp.hstack([
                    lvE[j, i] + expr_sat_alpha - cp.multiply(np.ones(n_term_s), denom),
                    lvE[j, i] + expr_sat_beta - cp.multiply(np.ones(n_term_p), denom),
                    lvE[j, i] + 0 - cp.multiply(np.ones(1), denom),
                    cp.multiply(np.ones(1), num) - cp.multiply(np.ones(1), denom),
                ]))

                denom_expr.append(denom)


        # print(f"LSE_expr: {LSE_expr}", f"denom_expr: {denom_expr}", sep='\n')

        return LSE_expr, denom_expr


    def create_objective_function(self, cfwd, crev, c, Km_s, Km_p, Km_i, Km_a, y_s, y_p,
                                  denom_expr, dmdt=None, l=0.01, e=0.01, f=1, g = 0.1):

        loss = 0

        l1 = cp.sum(cp.hstack([cfwd, crev, cp.vec(c)])) + cp.sum(cp.hstack([-Km_s, -Km_p]))  # regularization
        prior = cp.norm1(cp.hstack([cfwd, crev, cp.vec(c)])) + cp.norm1(cp.hstack([-Km_s, -Km_p]))  # prior
        # reg3 = cp.sum(cp.huber(cp.hstack([y_s, y_p]), 1))  # issue with matrix
        # reg4 = cp.sum(cp.max(cp.abs(cp.hstack([y_s, y_p])) - 3, 0)) # deadzone regularization


        if Km_i:
            l1 += cp.sum(cp.hstack([-Km_i]))
        if Km_a:
            l1 += cp.sum(cp.hstack([-Km_a]))

        # for i in range(len(LSE_expr)):
        #     loss += cp.norm1(cp.pos(cp.log_sum_exp(LSE_expr[i])))
        for i in range(len(denom_expr)):
            loss += f * denom_expr[i]
        loss += l * l1 + e * prior

        return loss

    def set_parameter_bounds(self, cfwd, crev, c, Km_s, Km_p, Km_i, Km_a, LSE_expr, cd_f=None,
                             lower_bound=-12, upper_bound=12):

        constr = [cp.hstack([cfwd, crev, cp.vec(c), Km_s, Km_p]) >= lower_bound,
                  cp.hstack([cfwd, crev, cp.vec(c), Km_s, Km_p]) <= upper_bound,]

        if cd_f:
            constr.extend([cd_f >= -7])

        if Km_i:
            constr.extend([Km_i >= lower_bound, Km_i <= upper_bound])
        if Km_a:
            constr.extend([Km_a >= lower_bound, Km_a <= upper_bound])

        for i in range(len(LSE_expr)):
            constr.extend([cp.pos(cp.log_sum_exp(LSE_expr[i])) <= 0])

        return constr

    def add_thermodynamic_constraints(self, constr, flow_data, K_eq, S, S_s_nz, S_p_nz, cfwd, crev, Km_s, Km_p, n_flux_set, y_s, y_p, c):

        sign = np.sign(flow_data)

        haldane = []

        for i, r in enumerate(S.T):
            Km_s_idx = np.nonzero(S_s_nz[1, :] == i)
            S_s_idx = S_s_nz[0, S_s_nz[1, :] == i]  # negate -1 entries

            Km_p_idx = np.nonzero(S_p_nz[1, :] == i)
            S_p_idx = S_p_nz[0, S_p_nz[1, :] == i]

            haldane.append(K_eq[i] == cfwd[i] - crev[i] + r[S_p_idx] @ Km_p[Km_p_idx] - (-r[S_s_idx]) @ Km_s[Km_s_idx])

        for j in range(n_flux_set):
            constr.extend([cp.multiply(S.T @ cp.vec(c[:, j]), sign[j, :]) <= cp.multiply(K_eq, sign[j, :])])

        constr.extend(haldane)

        return constr

    def set_up_problem(self, loss, constraints):

        problem = cp.Problem(cp.Minimize(loss), constraints)

        return problem

    def solve(self, problem, solver = cp.ECOS, verbose = False):

        problem.solve(solver = solver, verbose = verbose)

        return problem

    def evaluate_equality_fit(self, LSE_expr):

        equality_constr = []

        for v in LSE_expr:
            equality_constr.append(logsumexp(v.value))

        return equality_constr

    def evaluate_flux_reconstruction(self, vE, n_flux_set, n_rxn, S_b, S_s_nz, S_p_nz, d_alpha, d_beta,
                                     C_alpha, C_beta, y_f, y_r, y_s, y_p, cfwd, crev):

        reconstructed_vE = np.zeros(vE.shape)

        for j in range(n_flux_set):
            sat_expr = []
            fwd_sat = np.zeros(n_rxn)
            back_sat = np.zeros(n_rxn)
            sat = np.zeros(n_rxn)

            for i in range(n_rxn):
                # sum terms are separate in logsumexp. one per saturation term (row in C_alpha, C_beta)
                n_term_s = np.sum(d_alpha == i)
                n_term_p = np.sum(d_beta == i)
                n_term = n_term_s + n_term_p

                Km_s_idx = np.nonzero(S_s_nz[1, :] == i)
                S_s_idx = S_s_nz[0, S_s_nz[1, :] == i]  # negate -1 entries

                Km_p_idx = np.nonzero(S_p_nz[1, :] == i)
                S_p_idx = S_p_nz[0, S_p_nz[1, :] == i]

                # S_s_idx = S_s_nz[0, S_s_nz[1, :] == i]

                sat_expr.append([(C_alpha @ y_f.value[j, :].flatten())[d_alpha == i],
                                 (C_beta @ y_r.value[j, :].flatten())[d_beta == i],
                                 0,
                                 # -1*np.ones(n_lse_terms - n_term + 1)
                                 ]
                                )
                fwd_sat[i] = (np.exp(-S_b.T[i, S_s_idx] @ y_s.value[j, Km_s_idx].flatten()))  # + cfwd.value[i]
                back_sat[i] = (np.exp(S_b.T[i, S_p_idx] @ y_p.value[j, Km_p_idx].flatten()))  # + cfwd.value[i]

            for i, rxn in enumerate(sat_expr):
                s = 0

                for term in rxn:
                    s += np.sum(np.exp(term))

                sat[i] = (s)

            reconstr = np.exp(cfwd.value) * fwd_sat / sat - np.exp(crev.value) * back_sat / sat
            reconstructed_vE[j, :] = reconstr

            return reconstructed_vE

    def calculate_mechanistic_rates(self):
        pass