import numpy as np
import cvxpy as cp
import itertools
from scipy.special import logsumexp


class ConvexKineticsNew:

    def __init__(self):
        pass

    def set_up_variables(self, S_matrix, R_matrix, flow_data):

        # set up variables
        Sd = S_matrix
        Sr = R_matrix
        n_flux_set = flow_data.shape[0]

        # set up variables
        n_met = len(Sd.index)
        n_rxn = len(Sd.columns)

        S_mol = np.array(Sd)
        S = np.sign(S_mol)  #
        S_s = -np.copy(S)  # reverse neg sign
        S_p = np.copy(S)
        S_s[S > 0] = 0  # zeros products
        S_p[S < 0] = 0  # zeros substrates
        S_i = np.copy(np.array(Sr) == -1)  # reaction direction does not matter
        S_a = np.copy(np.array(Sr) == 1)

        S_s_nz = np.array(S_s.nonzero())
        S_p_nz = np.array(S_p.nonzero())
        S_i_nz = np.array(S_i.nonzero())
        S_a_nz = np.array(S_a.nonzero())
        S_s_mol = np.abs(S_mol)[S_s.nonzero()]
        S_p_mol = np.abs(S_mol)[S_p.nonzero()]

        # TODO Refactor all the below lines as one liners
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
        n_Km_s = np.max(met_s_nz.shape)
        n_Km_p = np.max(met_p_nz.shape)
        n_Km_i = np.max(met_i_nz.shape) if Sr is not None else None
        n_Km_a = np.max(met_a_nz.shape) if Sr is not None else None

        c = cp.Variable([n_met, n_flux_set])
        Km_s = cp.Variable(n_Km_s)
        Km_p = cp.Variable(n_Km_p)
        Km_i = cp.Variable(n_Km_i) if n_Km_i else None
        Km_a = cp.Variable(n_Km_a) if n_Km_a else None

        cfwd = cp.Variable(n_rxn)
        crev = cp.Variable(n_rxn)

        # define y vecs
        y_s_t, y_p_t, y_i_t, y_a_t = [], [], [], []

        # define Km positions by nonzero S matrix concentrations. Activation is reverse val of inhibition.
        # TODO Add molecularity here.
        for i in range(n_flux_set):
            y_s_t.append(cp.multiply(S_s_mol, c[met_s_nz, i] - Km_s))
            y_p_t.append(cp.multiply(S_p_mol, c[met_p_nz, i] - Km_p))
            y_i_t.append(c[met_i_nz, i] - Km_i if n_Km_i else None)
            y_a_t.append(-(c[met_a_nz, i] - Km_a) if n_Km_a else None)

        y_s = cp.vstack(y_s_t)
        y_p = cp.vstack(y_p_t)
        y_i = cp.vstack(y_i_t)
        y_a = cp.vstack(y_a_t)

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

        return y_f, y_r, y_s, y_p, y_i, y_a, cfwd, crev, c, Km_s, Km_p, Km_i, Km_a, S_s, S_p, S_i, S_a, \
            met_s_nz, met_p_nz, met_i_nz, met_a_nz, rxn_s_nz, rxn_p_nz, rxn_i_nz, rxn_a_nz, \
            n_rxn, n_met, n_flux_set, S_s_nz, S_p_nz, S

    # def add_regulation(self, regulation_matrix):
    #
    #     pass
    #
    # def add_flow_data(self, flow_data):
    #
    #     pass

    def construct_binding_matrix(self, n_rxn, S_s, S_p, S_i, S_a, Sr, met_s_nz, met_p_nz, met_i_nz, met_a_nz, rxn_s_nz, rxn_p_nz, rxn_i_nz, rxn_a_nz):

        # number of saturation terms for sub, prod
        S_s_comb = np.concatenate((S_s, S_i, S_a), axis=0) if Sr is not None else S_s
        S_p_comb = np.concatenate((S_p, S_i, S_a), axis=0) if Sr is not None else S_p
        n_alpha = np.sum(np.power(2, np.sign(S_s_comb).sum(axis=0)) - 1)
        n_beta = np.sum(np.power(2, np.sign(S_p_comb).sum(axis=0)) - 1)

        # saturation matrix setup, first sub, then inhib, then act.
        C_alpha = np.zeros([n_alpha, len(met_s_nz) + len(met_i_nz) + len(met_a_nz)]) if Sr is not None else np.zeros(
            [n_alpha, len(met_s_nz)])
        C_beta = np.zeros([n_beta, len(met_p_nz) + len(met_i_nz) + len(met_a_nz)]) if Sr is not None else np.zeros(
            [n_beta, len(met_p_nz)])

        # to separate different reactions saturation terms to their individual reaction equations.
        d_alpha = np.zeros(n_alpha, dtype=np.int64)
        d_beta = np.zeros(n_beta, dtype=np.int64)

        idx = 0

        for i in range(n_rxn):
            # pick one reaction at a time (get substrate indicies)
            # idx_cur_rxn = rxn_s_nz == i
            # TODO This does not properly multiply by molecularity. Alternatively, generate C_alpha and
            # TODO beta without molecularity (first ==1) and then multiply by molecularity in the end.
            idx_cur_rxn = np.concatenate((rxn_s_nz == i, rxn_i_nz == i, rxn_a_nz == i)) if Sr is not None else rxn_s_nz == i

            # generates all binary permutations minus the first one since that would result in -1
            sat_perm = np.array(list(itertools.product([0, 1], repeat=sum(idx_cur_rxn))))
            sat_perm = sat_perm[1:, :]

            r, _ = sat_perm.shape

            # replace zeros with saturation matrix
            C_alpha[idx:(idx + r), idx_cur_rxn] = sat_perm
            d_alpha[idx:(idx + r)] = i

            idx += r  # add row #

        idx = 0

        for i in range(n_rxn):
            idx_cur_rxn = np.concatenate((rxn_p_nz == i, rxn_i_nz == i, rxn_a_nz == i)) if Sr is not None else rxn_p_nz == i

            sat_perm = np.array(list(itertools.product([0, 1], repeat=sum(idx_cur_rxn))))
            sat_perm = sat_perm[1:, :]

            r, _ = sat_perm.shape

            C_beta[idx:(idx + r), idx_cur_rxn] = sat_perm
            d_beta[idx:(idx + r)] = i

            idx += r  # add row #

        print(f"Shape of C_alpha: {C_alpha.shape}, shape of C_beta: {C_beta.shape}",
              f"Shape of d_alpha: {d_alpha.shape}, shape of d_beta: {d_beta.shape}", sep='\n')

        return C_alpha, C_beta, d_alpha, d_beta

    def construct_kinetic_objective(self, flow_data, n_flux_set, n_rxn, C_alpha, C_beta, d_alpha, d_beta, S_s_nz, S_p_nz, S, y_f, y_r, y_s, y_p, cfwd, crev):

        LSE_expr = []
        denom_expr = []

        sign = np.sign(flow_data)
        lvE = np.log(sign * flow_data)

        for j in range(n_flux_set):
            for i in range(n_rxn):
                # sum terms are separate in logsumexp. one per saturation term (row in C_alpha, C_beta)

                n_term_s = np.sum(d_alpha == i)
                n_term_p = np.sum(d_beta == i)
                n_term = n_term_s + n_term_p

                Km_s_idx = np.nonzero(S_s_nz[1, :] == i)
                S_s_idx = S_s_nz[0, S_s_nz[1, :] == i]  # negate -1 entries

                Km_p_idx = np.nonzero(S_p_nz[1, :] == i)
                S_p_idx = S_p_nz[0, S_p_nz[1, :] == i]

                if sign[j, i] == 1:
                    LSE_expr.append(cp.hstack([
                        lvE[j, i] + (C_alpha @ cp.vec(y_f[j, :]))[d_alpha == i]
                        - cp.multiply(np.ones(n_term_s), - S.T[i, S_s_idx] @ cp.vec(y_s[j, Km_s_idx])) - cfwd[i],
                        lvE[j, i] + (C_beta @ cp.vec(y_r[j, :]))[d_beta == i]
                        - cp.multiply(np.ones(n_term_p), - S.T[i, S_s_idx] @ cp.vec(y_s[j, Km_s_idx])) - cfwd[i],

                        lvE[j, i] + 0 - cp.multiply(np.ones(1), -S.T[i, S_s_idx] @ cp.vec(y_s[j, Km_s_idx])) - cfwd[i],

                        cp.multiply(np.ones(1), S.T[i, S_p_idx] @ cp.vec(y_p[j, Km_p_idx])) + crev[i]
                        - cp.multiply(np.ones(1), -S.T[i, S_s_idx] @ cp.vec(y_s[j, Km_s_idx])) - cfwd[i],

                    ]
                    )
                    )  # remove +1 here, could also have cfwd outside objec.

                    denom_expr.append(cp.multiply(np.ones(1), -S.T[i, S_s_idx] @ cp.vec(y_s[j, Km_s_idx])) + cfwd[i], )

                # keep saturation term the same, switch around fwd and rev terms. flip all signs with S matrix since it's signed.
                if sign[j, i] == -1:
                    LSE_expr.append(cp.hstack([lvE[j, i] + (C_alpha @ cp.vec(y_f[j, :]))[d_alpha == i]
                                               - cp.multiply(np.ones(n_term_s),
                                                             S.T[i, S_p_idx] @ cp.vec(y_p[j, Km_p_idx])) - crev[i],

                                               lvE[j, i] + (C_beta @ cp.vec(y_r[j, :]))[d_beta == i]
                                               - cp.multiply(np.ones(n_term_p),
                                                             S.T[i, S_p_idx] @ cp.vec(y_p[j, Km_p_idx])) - crev[i],

                                               lvE[j, i] + 0 - cp.multiply(np.ones(1),
                                                                           S.T[i, S_p_idx] @ cp.vec(y_p[j, Km_p_idx])) -
                                               crev[i],

                                               cp.multiply(np.ones(1), - S.T[i, S_s_idx] @ cp.vec(y_s[j, Km_s_idx])) +
                                               cfwd[i]
                                               - cp.multiply(np.ones(1), S.T[i, S_p_idx] @ cp.vec(y_p[j, Km_p_idx])) -
                                               crev[i],

                                               ]
                                              )
                                    )

                    denom_expr.append(cp.multiply(np.ones(1), S.T[i, S_p_idx] @ cp.vec(y_p[j, Km_p_idx])) + crev[i])

        # print(f"LSE_expr: {LSE_expr}", f"denom_expr: {denom_expr}", sep='\n')

        return LSE_expr, denom_expr


    def create_objective_function(self, cfwd, crev, c, Km_s, Km_p, Km_i, Km_a, y_s, y_p, LSE_expr, denom_expr):

        l = 0.001
        e = 0.001
        f = 0.000001
        reg = cp.sum(cp.hstack([cfwd, crev, cp.vec(c)])) + cp.sum(cp.hstack([-Km_s, -Km_p]))  # regularization
        reg2 = cp.norm1(cp.hstack([cfwd, crev, cp.vec(c)])) + cp.norm1(cp.hstack([-Km_s, -Km_p]))  # regularization
        reg3 = cp.sum(cp.huber(cp.hstack([y_s, y_p]), 1))  # issue with matrix

        if Km_i:
            reg += cp.sum(cp.hstack([-Km_i]))
        if Km_a:
            reg += cp.sum(cp.hstack([-Km_a]))
        # reg3 = cp.norm1(cp.hstack([y_s, y_p])) # take a look at this

        loss = 0
        for i in range(len(LSE_expr)):
            loss += cp.norm1(cp.pos(cp.log_sum_exp(LSE_expr[i])))
        for i in range(len(denom_expr)):
            loss += 0.01 * denom_expr[i]
        loss += l * reg
        loss += e * reg2
        # loss += f * reg3

        return loss

    def set_parameter_bounds(self, cfwd, crev, c, Km_s, Km_p, Km_i, Km_a, lower_bound=-12, upper_bound=12):

        constr = [cp.hstack([cfwd, crev, cp.vec(c), Km_s, Km_p]) >= lower_bound,
                  cp.hstack([cfwd, crev, cp.vec(c), Km_s, Km_p]) <= upper_bound,
                  ]

        if Km_i:
            constr.extend([Km_i >= lower_bound, Km_i <= upper_bound])
        if Km_a:
            constr.extend([Km_a >= lower_bound, Km_a <= upper_bound])

        return constr

    def add_mechanistic_constraints(self, constr, flow_data, K_eq, S, S_s_nz, S_p_nz, cfwd, crev, Km_s, Km_p, n_flux_set, y_s, y_p, c):

        sign = np.sign(flow_data)

        haldane = []
        fwd_flux = []

        for i, r in enumerate(S.T):
            Km_s_idx = np.nonzero(S_s_nz[1, :] == i)
            S_s_idx = S_s_nz[0, S_s_nz[1, :] == i]  # negate -1 entries

            Km_p_idx = np.nonzero(S_p_nz[1, :] == i)
            S_p_idx = S_p_nz[0, S_p_nz[1, :] == i]

            haldane.append(K_eq[i] == cfwd[i] - crev[i] + r[S_p_idx] @ Km_p[Km_p_idx] - (-r[S_s_idx]) @ Km_s[Km_s_idx])

        for j in range(n_flux_set):
            for i, r in enumerate(S.T):
                Km_s_idx = np.nonzero(S_s_nz[1, :] == i)
                S_s_idx = S_s_nz[0, S_s_nz[1, :] == i]  # negate -1 entries

                Km_p_idx = np.nonzero(S_p_nz[1, :] == i)
                S_p_idx = S_p_nz[0, S_p_nz[1, :] == i]

                if sign[j, i] == 1:
                    fwd_flux.append(cfwd[i] + (-r[S_s_idx]) @ cp.vec(y_s[j, Km_s_idx]) - (crev[i] + r[S_p_idx] @ cp.vec(
                        y_p[j, Km_p_idx])) >= 0)  # add minus since s matrix has minus

                if sign[j, i] == -1:
                    fwd_flux.append(cfwd[i] + (-r[S_s_idx]) @ cp.vec(y_s[j, Km_s_idx]) - (crev[i] + r[S_p_idx] @ cp.vec(
                        y_p[j, Km_p_idx])) <= 0)  # add minus since s matrix has minus

            constr.extend([cp.multiply(S.T @ cp.vec(c[:, j]), sign[j, :]) <= cp.multiply(K_eq, sign[j, :])])

        constr.extend(haldane)
        constr.extend(fwd_flux)

        return constr

    def set_up_problem(self, loss, constraints):

        problem = cp.Problem(cp.Minimize(loss), constraints)

        return problem

    def solve(self, problem, solver = cp.ECOS, verbose = False):

        problem.solve(solver = solver, verbose = verbose)

        return problem

    def evaluate_fit(self, LSE_expr):

        for v in LSE_expr:
            print(logsumexp(v.value))

    def calculate_mechanistic_rates(self):
        pass