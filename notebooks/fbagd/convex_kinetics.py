import numpy as np
import cvxpy as cp
import itertools
from scipy.special import logsumexp

class ConvexKinetics:

    def __init__(self, S_matrix):

        self.regulation = False # TODO make this in a better way
        self.loss = 0
        self.constraints = []

        self.n_met = len(S_matrix.index)
        self.n_rxn = len(S_matrix.columns)

        S_mol = np.array(S_matrix)   # has molecularity
        self.S = np.sign(S_mol)      # has only +1 and -1 stoichiometries

        self.S_s, self.S_p = -np.copy(self.S), np.copy(self.S) # reverse neg sign
        self.S_s[self.S > 0] = 0 # zeros products
        self.S_p[self.S < 0] = 0 # zeros substrates

        self.S_s_nz = np.array(self.S_s.nonzero())    # substrate indices (metabolite, reaction)
        self.S_p_nz = np.array(self.S_p.nonzero())    # product indices (metabolite, reaction)
        self.S_s_mol = np.abs(S_mol)[self.S_s.nonzero()]  # substrate molecularity at indices
        self.S_p_mol = np.abs(S_mol)[self.S_p.nonzero()]  # product molecularity at indices

        # TODO Refactor all the below lines as one liners
        # first coordinate, e.g. metabolites w nonzero substrate/product coeff across all reactions. also works as substrate indices.
        self.met_s_nz, self.met_p_nz = self.S_s_nz[0, :], self.S_p_nz[0, :]     #

        # second coordinate, e.g. reactions indices for those concentrations. works to index substrates as well.
        self.rxn_s_nz, self.rxn_p_nz = self.S_s_nz[1, :], self.S_p_nz[1, :]

        # one dim is always 2
        n_Km_s, n_Km_p = len(self.met_s_nz), len(self.met_p_nz) # number of substrate and product Km

        self.Km_s, self.Km_p = cp.Variable(n_Km_s), cp.Variable(n_Km_p) # substrate and product Km
        self.cfwd, self.crev = cp.Variable(self.n_rxn), cp.Variable(self.n_rxn) # forward and reverse reaction rate constants

        print(f"Number of metabolites: {self.n_met}", f"Number of reactions: {self.n_rxn}",
              f"Number of substrate Km: {n_Km_s}", f"Number of product Km: {n_Km_p}",
              f"Number of forward rate constants: {self.n_rxn}", f"Number of reverse rate constants: {self.n_rxn}", sep="\n")

    def add_regulation(self, regulation_matrix):

        # TODO Assert that regulation matrix has same order of metabolites as S matrix

        Sr = regulation_matrix

        self.S_i = np.copy(np.array(Sr) == -1) # reaction direction does not matter
        self.S_a = np.copy(np.array(Sr) == 1)

        S_i_nz = np.array(self.S_i.nonzero())
        S_a_nz = np.array(self.S_a.nonzero())

        self.met_i_nz, self.met_a_nz = S_i_nz[0, :], S_a_nz[0, :]
        self.rxn_i_nz, self.rxn_a_nz = S_i_nz[1, :], S_a_nz[1, :]

        n_Km_i, n_Km_a = len(self.met_i_nz), len(self.met_a_nz)
        self.Km_i = cp.Variable(n_Km_i) if n_Km_i else None
        self.Km_a = cp.Variable(n_Km_a) if n_Km_a else None

        self.regulation = True

        print(f"Number of inhibition Km: {n_Km_i}", f"Number of activation Km: {n_Km_a}", sep="\n")

    def add_flow_data(self, flow_data):

        # TODO assert flux columns are identical to S_matrix columns
        self.n_flux_set = len(flow_data.index)
        self.flow_data = np.array(flow_data)

        self.c = cp.Variable([self.n_met, self.n_flux_set])    # concentrations

        y_s_t, y_p_t, y_i_t, y_a_t = [], [], [], []

        # define Km positions by nonzero S matrix concentrations. Activation is reverse val of inhibition.
        for i in range(self.n_flux_set):
            y_s_t.append(cp.multiply(self.S_s_mol, self.c[self.met_s_nz, i] - self.Km_s))
            y_p_t.append(cp.multiply(self.S_p_mol, self.c[self.met_p_nz, i] - self.Km_p))
            y_i_t.append(self.c[self.met_i_nz, i] - self.Km_i if self.Km_i else None)
            y_a_t.append(-(self.c[self.met_a_nz, i] - self.Km_a) if self.Km_a else None)

        self.y_s, self.y_p, self.y_i, self.y_a = cp.vstack(y_s_t), cp.vstack(y_p_t), cp.vstack(y_i_t), cp.vstack(y_a_t)

        y_f_vec, y_r_vec = [self.y_s], [self.y_p]
        if self.Km_i:
            y_f_vec.append(self.y_i)
            y_r_vec.append(self.y_i)
        if self.Km_a:
            y_f_vec.append(self.y_a)
            y_r_vec.append(self.y_a)

        self.y_f, self.y_r = cp.hstack(y_f_vec), cp.hstack(y_r_vec)

        print(f"Number of flux sets: {self.n_flux_set}, Number of concentration variables: {self.n_met * self.n_flux_set}")

    def construct_binding_matrix(self):

        # number of saturation terms for sub, prod
        # make the code below cleaner
        S_s_comb = np.concatenate((self.S_s, self.S_i, self.S_a), axis=0)  if self.regulation else self.S_s
        S_p_comb = np.concatenate((self.S_p, self.S_i, self.S_a), axis=0) if self.regulation else self.S_p
        n_alpha = np.sum(np.power(2, np.sign(S_s_comb).sum(axis=0)) - 1)
        n_beta = np.sum(np.power(2, np.sign(S_p_comb).sum(axis=0)) - 1)

        # saturation matrix setup, first sub, then inhib, then act.
        C_alpha = np.zeros([n_alpha, len(self.met_s_nz) + len(self.met_i_nz) + len(self.met_a_nz)])
        C_beta = np.zeros([n_beta, len(self.met_p_nz) + len(self.met_i_nz) + len(self.met_a_nz)])

        # to separate different reactions saturation terms to their individual reaction equations.
        d_alpha, d_beta = np.zeros(n_alpha, dtype=np.int8), np.zeros(n_beta, dtype=np.int8)

        s_idx, p_idx = 0, 0

        for i in range(self.n_rxn):
            # pick one reaction at a time (get substrate indicies)
            idx_s_cur_rxn = np.concatenate((self.rxn_s_nz == i, self.rxn_i_nz == i, self.rxn_a_nz == i))
            idx_p_cur_rxn = np.concatenate((self.rxn_p_nz == i, self.rxn_i_nz == i, self.rxn_a_nz == i))

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

        # TODO Use return instead of setting as attribute
        self.C_alpha, self.C_beta, self.d_alpha, self.d_beta = C_alpha, C_beta, d_alpha, d_beta

    def construct_kinetic_objective(self):

        # TODO
        LSE_expr, denom_expr = [], []

        sign = np.sign(self.flow_data)
        lvE = np.log(sign * self.flow_data)

        for j in range(self.n_flux_set):
            for i in range(self.n_rxn):
                # sum terms are separate in logsumexp. one per saturation term (row in C_alpha, C_beta)

                Km_s_idx = np.nonzero(self.rxn_s_nz == i) # TODO have to set as attribute
                S_s_idx = self.S_s_nz[0, self.rxn_s_nz == i] # negate -1 entries

                Km_p_idx = np.nonzero(self.rxn_p_nz == i)
                S_p_idx = self.S_p_nz[0, self.rxn_p_nz == i]

                if sign[j, i] == 1:
                    expr_num = [
                                lvE[j, i] + (self.C_alpha @ cp.vec(self.y_f[j, :]))[self.d_alpha == i],
                                lvE[j, i] + (self.C_beta @ cp.vec(self.y_r[j, :]))[self.d_beta == i],
                                lvE[j, i] + 0,
                                self.S.T[i, S_p_idx] @ cp.vec(self.y_p[j, Km_p_idx])  + self.crev[i] # TODO did removing multiply break this?
                                ] # TODO first three terms are the same. can be combined outside if statement

                    expr_denom = - (- self.S.T[i, S_s_idx] @ cp.vec(self.y_s[j, Km_s_idx])) - self.cfwd[i]
                    expr = cp.hstack(expr_num) + expr_denom # TODO this might be wrong
                    LSE_expr.append(expr) # TODO vectorize this

                    denom_expr.append(-expr_denom) # TODO vectorize this


                # keep saturation term the same, switch around fwd and rev terms. flip all signs with S matrix since it's signed.
                if sign[j, i] == -1:
                    expr_num = [
                                lvE[j, i] + (self.C_alpha @ cp.vec(self.y_f[j, :]))[self.d_alpha == i],
                                lvE[j, i] + (self.C_beta @ cp.vec(self.y_r[j, :]))[self.d_beta == i],
                                lvE[j, i] + 0,
                                - self.S.T[i, S_s_idx] @ cp.vec(self.y_s[j, Km_s_idx]) + self.cfwd[i]
                                ]

                    expr_denom = - (self.S.T[i, S_p_idx] @ cp.vec(self.y_p[j, Km_p_idx])) - self.crev[i]
                    expr = cp.hstack(expr_num) + expr_denom # TODO this might be wrong

                    LSE_expr.append(expr)

                    denom_expr.append(-expr_denom)

            self.LSE_expr, self.denom_expr = LSE_expr, denom_expr

    def create_objective_function(self, prior_weight = 0.001, l1_weight = 0.001, denom_weight = 0.01):

        p = prior_weight
        l1 = l1_weight
        l1_term =  cp.sum(cp.hstack([self.cfwd, self.crev, cp.vec(self.c)])) + cp.sum(cp.hstack([-self.Km_s, -self.Km_p])) # regularization (l1 because geometric)
        p_term = cp.norm1(cp.hstack([self.cfwd, self.crev, cp.vec(self.c)])) + cp.norm1(cp.hstack([-self.Km_s, -self.Km_p])) # regularization # TODO these are conflicting also prior

        if self.Km_i is not None:
            l1_term += cp.sum(cp.hstack([-self.Km_i]))
        if self.Km_a is not None:
            l1_term += cp.sum(cp.hstack([-self.Km_a])) # TODO this might break if no Km

        for i in range(len(self.LSE_expr)):
            self.loss += cp.norm1(cp.pos(cp.log_sum_exp(self.LSE_expr[i])))
        for i in range(len(self.denom_expr)):
            self.loss += denom_weight * self.denom_expr[i]

        self.loss += l1 * l1_term + p * p_term

    def set_parameter_bounds(self, lower_bound = -12, upper_bound = 12):

        self.constraints.append(cp.hstack([self.cfwd, self.crev, cp.vec(self.c), self.Km_s, self.Km_p]) >= lower_bound)
        self.constraints.append(cp.hstack([self.cfwd, self.crev, cp.vec(self.c), self.Km_s, self.Km_p]) <= upper_bound) # TODO might be append lol

        if self.Km_i:
            self.constraints.extend([self.Km_i >= -lower_bound, self.Km_i <= upper_bound])
        if self.Km_a:
            self.constraints.extend([self.Km_a >= -lower_bound, self.Km_a <= upper_bound])

    def add_mechanistic_constraints(self, K_eq):

        # TODO Assert that order of K_eq is the same as order of self.S (dict)
        # TODO Only add thermo constraints for reactions with data
        sign = np.sign(self.flow_data)
        haldane = []
        fwd_flux = []

        for i, r in enumerate(self.S.T):    # TODO do this for kinetic objective creation
            Km_s_idx = np.nonzero(self.rxn_s_nz == i) # TODO have to set as attribute
            S_s_idx = self.S_s_nz[0, self.rxn_s_nz == i] # negate -1 entries

            Km_p_idx = np.nonzero(self.rxn_p_nz == i)
            S_p_idx = self.S_p_nz[0, self.rxn_p_nz == i]

            haldane.append(K_eq[i] == self.cfwd[i] - self.crev[i] + r[S_p_idx] @ self.Km_p[Km_p_idx] - (-r[S_s_idx]) @ self.Km_s[Km_s_idx])

        for j in range(self.n_flux_set):
            for i, r in enumerate(self.S.T): # TODO use the above for loop
                Km_s_idx = np.nonzero(self.rxn_s_nz == i) # TODO have to set as attribute
                S_s_idx = self.S_s_nz[0, self.rxn_s_nz == i] # negate -1 entries

                Km_p_idx = np.nonzero(self.rxn_p_nz == i)
                S_p_idx = self.S_p_nz[0, self.rxn_p_nz == i]

                if sign[j, i] == 1:
                    fwd_flux.append(self.cfwd[i] + (-r[S_s_idx]) @ cp.vec(self.y_s[j, Km_s_idx])
                                    - (self.crev[i] + r[S_p_idx] @ cp.vec(self.y_p[j, Km_p_idx]))  >= 0)  # add minus since s matrix has minus

                if sign[j, i] == -1:
                    fwd_flux.append(self.cfwd[i] + (-r[S_s_idx]) @ cp.vec(self.y_s[j, Km_s_idx])
                                    - (self.crev[i] + r[S_p_idx] @ cp.vec(self.y_p[j, Km_p_idx]))  <= 0)  # add minus since s matrix has minus

            self.constraints.extend([cp.multiply(self.S.T @ cp.vec(self.c[:, j]), sign[j, :])  <= cp.multiply(K_eq, sign[j, :])])

        self.constraints.extend(haldane)
        self.constraints.extend(fwd_flux)

    def set_up_problem(self):

        self.problem = cp.Problem(cp.Minimize(self.loss), self.constraints)

    def solve(self, solver = cp.ECOS, verbose = False, **kwargs):

        self.problem.solve(solver = solver, verbose = verbose, **kwargs)

    def evaluate_fit(self):

        for v in self.LSE_expr:
            print(logsumexp(v.value))

    def calculate_mechanistic_rates(self):

        reconstructed_flow = np.zeros(self.flow_data.shape)

        for j in range(self.n_flux_set):
            sat_expr = []
            fwd_sat = np.zeros(self.n_rxn)
            back_sat = np.zeros(self.n_rxn)
            sat = np.zeros(self.n_rxn)

            for i in range(self.n_rxn):
                # sum terms are separate in logsumexp. one per saturation term (row in C_alpha, C_beta)
                n_term_s = np.sum(self.d_alpha == i)
                n_term_p = np.sum(self.d_beta == i)
                n_term = n_term_s + n_term_p

                Km_s_idx = np.nonzero(self.rxn_s_nz == i)
                S_s_idx = self.S_s_nz[0, self.rxn_s_nz == i] # negate -1 entries

                Km_p_idx = np.nonzero(self.rxn_p_nz == i)
                S_p_idx = self.S_p_nz[0, self.rxn_p_nz == i]

                #S_s_idx = S_s_nz[0, S_s_nz[1, :] == i]

                sat_expr.append(           [ (self.C_alpha @ self.y_f.value[j, :].flatten())[self.d_alpha == i] ,
                                             (self.C_beta @ self.y_r.value[j, :].flatten())[self.d_beta == i],
                                             0,
                                           ]
                               )
                fwd_sat[i] = (np.exp(-self.S.T[i, S_s_idx] @ self.y_s.value[j, Km_s_idx].flatten())) # + cfwd.value[i]
                back_sat[i] = (np.exp(self.S.T[i, S_p_idx] @ self.y_p.value[j, Km_p_idx].flatten())) # + cfwd.value[i]



            for i, rxn in enumerate(sat_expr):
                s = 0

                for term in rxn:
                    s += np.sum(np.exp(term))

                sat[i] = (s)

            reconstr = np.exp(self.cfwd.value) * fwd_sat/sat - np.exp(self.crev.value) * back_sat/sat
            print(reconstr)
            reconstructed_flow[j, :] = reconstr