import argparse
import csv
import os
import re
import sys
import time
from typing import Dict, List, Optional, Set, Tuple, cast
import math

import matplotlib.pyplot as plt
import numpy as np
import scipy
from rnaseq_utils import *


def stochastic_sim(n_gens, plot_name, ligand_fxn):

    ## Parameters
    doubling_time = 40

    mRNA_deg_rate = np.log(2) / 10 # 10 min degradation

    # Average of 20 mRNA molecules per cell, 40min doubling time, so makes
    # one every 2 minutes.
    mRNA_total_synth = 1/2

    # Average of 4000 protein molecules per cell, 40min doubling time, so makes
    # 100 every minute.
    protein_total_synth = 100

    # 4 promoters: gene D is unregulated, gene A regulates gene B and gene C,
    # and also autoregulates. gene A is a repressor with a corepressor (?).
    # Out of 20 mRNA molecules, I want 1 molecule of A, 9 molecules of B,
    # 3 molecules of C, 7 molecules of D.
    #ks_unbound = np.array([3, 9, 3, 7])
    ks_unbound = np.array([6, 9, 3, 7])
    ks_bound = np.array([3, 1, 3, 7])
    #ks_bound = np.array([0.5, 5, 1, 7])

    # Teffs: 4000 protein molecules per cell, I want around 100 molecules of A,
    # 2000 molecules of B, 400 of C, 1500 of D when unrepressed.
    teffs = np.array([100/3, 2000/9, 400/3, 1500/7])

    ligand_threshold = 100 # Arbitrary
    TF_binding_sites = np.array([1, 1, 0, 0]) # Might be interesting to look
    # at changes to this?

    # TF binding/unbinding: let's say it binds once per minute per promoter and TF.
    # Unbind rate for active, let's say once per 20 min.
    active_TF_bind_rate = 1
    active_TF_unbind_rate = 1/20
    inactive_TF_unbind_rate = 1
    inactive_TF_bind_rate = 0

    # At each time point, we have a RNA synth rate and a protein synth rate.
    # This is divied up based on the k's, and also deg rates.

    def total_synth_from_time(time):
        cell_time = time % 40
        mRNA_synth = mRNA_total_synth*np.exp(cell_time/doubling_time) / \
                     (math.e-1)
        protein_synth = protein_total_synth*np.exp(cell_time/doubling_time) /\
                        (math.e-1)

        return mRNA_synth, protein_synth

    def get_ks_from_promoters(promoter_state):
        ks = []
        for i, bound in enumerate(promoter_state):
            if bound == 1:
                ks.append(ks_bound[i])
            else:
                ks.append(ks_unbound[i])

        return np.array(ks)

    def calculate_rates(mRNA_levels, protein_levels, promoter_state, ligand,
                        time):
        mRNA_synth, protein_synth = total_synth_from_time(time)
        # Calculate mRNA synthesis/degradation rates
        ks = get_ks_from_promoters(promoter_state)
        mRNA_rates = mRNA_synth * ks / np.sum(ks)
        mRNA_deg = mRNA_deg_rate * mRNA_levels
        # Calculate protein synthesis rates
        mRNA_teffs = np.multiply(mRNA_levels, teffs)
        if np.sum(mRNA_teffs) > 0:
            protein_rates = protein_synth * mRNA_teffs / np.sum(mRNA_teffs)
        else:
            protein_rates = np.zeros(len(mRNA_levels))

        # Case if ligand binds to TFs:
        if ligand > ligand_threshold:
            n_free_active_TFs = protein_levels[0] - np.sum(promoter_state)
            TF_unbind_rates = promoter_state * active_TF_unbind_rate
            TF_bind_rates = []
            for i, bound in enumerate(promoter_state):
                if bound == 1:
                    TF_bind_rates.append(0)
                else:
                    TF_bind_rates.append(n_free_active_TFs * TF_binding_sites[i] *
                                         active_TF_bind_rate)
            TF_bind_rates = np.array(TF_bind_rates)
        # Case if ligand doesn't bind to TFs
        else:
            n_free_active_TFs = 0
            TF_unbind_rates = promoter_state * inactive_TF_unbind_rate
            TF_bind_rates = promoter_state * inactive_TF_bind_rate # 0

        all_rates = []
        all_rates.extend(mRNA_rates)
        all_rates.extend(protein_rates)
        all_rates.extend(mRNA_deg)
        all_rates.extend(TF_bind_rates)
        all_rates.extend(TF_unbind_rates)
        return all_rates, n_free_active_TFs

    def run_step(mRNA_levels, protein_levels, promoter_state, ligand, time):
        # TODO: finish this gillespie, run the sims with
        # ligand off then on then off, and see what happens.
        # And test at different growth rates too (with corresponding changes
        # in mRNA or protein total synth) to see what happens with smth like
        # trpR maybe?
        rates, _ = calculate_rates(mRNA_levels, protein_levels, promoter_state,
                                ligand, time)

        time_rand = np.random.uniform()
        timestep = 1/np.sum(rates) * np.log(1/time_rand)

        rxn_rand = np.random.uniform()
        cutoff_values = np.array([np.sum(rates[:i+1]) for i in range(len(rates))])
        cutoff_values = cutoff_values / np.sum(rates)
        if rxn_rand < cutoff_values[0]:
            mRNA_levels[0] += 1
        elif rxn_rand < cutoff_values[1]:
            mRNA_levels[1] += 1
        elif rxn_rand < cutoff_values[2]:
            mRNA_levels[2] += 1
        elif rxn_rand < cutoff_values[3]:
            mRNA_levels[3] += 1
        elif rxn_rand < cutoff_values[4]:
            protein_levels[0] += 1
        elif rxn_rand < cutoff_values[5]:
            protein_levels[1] += 1
        elif rxn_rand < cutoff_values[6]:
            protein_levels[2] += 1
        elif rxn_rand < cutoff_values[7]:
            protein_levels[3] += 1
        elif rxn_rand < cutoff_values[8]:
            mRNA_levels[0] -= 1
        elif rxn_rand < cutoff_values[9]:
            mRNA_levels[1] -= 1
        elif rxn_rand < cutoff_values[10]:
            mRNA_levels[2] -= 1
        elif rxn_rand < cutoff_values[11]:
            mRNA_levels[3] -= 1
        elif rxn_rand < cutoff_values[12]:
            assert promoter_state[0] == 0
            promoter_state[0] = 1
        elif rxn_rand < cutoff_values[13]:
            assert promoter_state[1] == 0
            promoter_state[1] = 1
        elif rxn_rand < cutoff_values[14]:
            assert promoter_state[2] == 0
            promoter_state[2] = 1
        elif rxn_rand < cutoff_values[15]:
            assert promoter_state[3] == 0
            promoter_state[3] = 1
        elif rxn_rand < cutoff_values[16]:
            assert promoter_state[0] == 1
            promoter_state[0] = 0
        elif rxn_rand < cutoff_values[17]:
            assert promoter_state[1] == 1
            promoter_state[1] = 0
        elif rxn_rand < cutoff_values[18]:
            assert promoter_state[2] == 1
            promoter_state[2] = 0
        elif rxn_rand < cutoff_values[19]:
            assert promoter_state[3] == 1
            promoter_state[3] = 0

        return timestep

    def run_one_gen(initial_mRNA, initial_protein, initial_promoter,
                       ligand_fxn, initial_time):
        timeseries = [initial_time]
        mRNA_timeseries = [copy.deepcopy(initial_mRNA)]
        protein_timeseries = [copy.deepcopy(initial_protein)]
        promoter_timeseries = [copy.deepcopy(initial_promoter)]
        ligand_timeseries = [ligand_fxn(initial_time)]

        end_time = initial_time + doubling_time
        while initial_time < end_time:
            ligand = ligand_fxn(initial_time)
            timestep = run_step(initial_mRNA, initial_protein, initial_promoter,
                                ligand, initial_time)
            initial_time += timestep
            mRNA_timeseries.append(copy.deepcopy(initial_mRNA))
            protein_timeseries.append(copy.deepcopy(initial_protein))
            promoter_timeseries.append(copy.deepcopy(initial_promoter))
            ligand_timeseries.append(ligand)
            timeseries.append(initial_time)

        div_mRNA = np.array([np.random.binomial(mRNA, 0.5) for mRNA in initial_mRNA])
        div_protein = np.array([np.random.binomial(protein, 0.5) for protein in initial_protein])
        return div_mRNA, div_protein, copy.deepcopy(initial_promoter), initial_time,\
            mRNA_timeseries, protein_timeseries, promoter_timeseries, \
            ligand_timeseries, timeseries

    def run_multigen(initial_mRNA, initial_protein, initial_promoter, ligand_fxn,
                     n_gens, plot_name):
        initial_time = 0
        mRNA_timeseries = []
        protein_timeseries = []
        promoter_timeseries = []
        ligand_timeseries = []
        timeseries = []
        for i in range(n_gens):
            # Run one generation
            div_mRNA, div_protein, div_promoter, div_time, mRNA_time, protein_time,\
            promoter_time, ligand_time, time = run_one_gen(initial_mRNA,
            initial_protein, initial_promoter, ligand_fxn, initial_time)
            # Record simulation results
            mRNA_timeseries.extend(mRNA_time)
            protein_timeseries.extend(protein_time)
            promoter_timeseries.extend(promoter_time)
            ligand_timeseries.extend(ligand_time)
            timeseries.extend(time)
            # Prepare for next generation
            initial_mRNA = div_mRNA
            initial_protein = div_protein
            initial_promoter = div_promoter
            initial_time = div_time

        mRNA_timeseries = np.array(mRNA_timeseries).T
        protein_timeseries = np.array(protein_timeseries).T
        promoter_timeseries = np.array(promoter_timeseries).T
        ligand_timeseries = np.array(ligand_timeseries)
        timeseries = np.array(timeseries)

        fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(20, 20))
        axs[0, 0].plot(timeseries, mRNA_timeseries[0], label='A')
        axs[0, 1].plot(timeseries, mRNA_timeseries[1], label='B')
        axs[0, 2].plot(timeseries, mRNA_timeseries[2], label='C')
        axs[0, 3].plot(timeseries, mRNA_timeseries[3], label='D')
        axs[0, 0].set_title("mRNA levels")
        axs[0, 0].set_xlabel("Time (min)")
        axs[0, 0].set_ylabel("mRNA counts")
        axs[0, 0].legend()

        axs[1, 0].plot(timeseries, protein_timeseries[0], label='A')
        axs[1, 1].plot(timeseries, protein_timeseries[1], label='B')
        axs[1, 2].plot(timeseries, protein_timeseries[2], label='C')
        axs[1, 3].plot(timeseries, protein_timeseries[3], label='D')
        axs[1, 0].set_title("Protein levels")
        axs[1, 0].set_xlabel("Time (min)")
        axs[1, 0].set_ylabel("Protein counts")
        axs[1, 0].legend()

        axs[2, 0].plot(timeseries, promoter_timeseries[0], label='A')
        axs[2, 0].plot(timeseries, promoter_timeseries[1], label='B')
        axs[2, 0].plot(timeseries, promoter_timeseries[2], label='C')
        axs[2, 0].plot(timeseries, promoter_timeseries[3], label='D')
        axs[2, 0].set_title("Promoter occupancy")
        axs[2, 0].set_xlabel("Time (min)")
        axs[2, 0].set_ylabel("Promoter occupied")
        axs[2, 0].set_ylim(0, 1.5)
        axs[2, 0].legend()

        axs[3, 0].plot(timeseries, ligand_timeseries)
        axs[3, 0].set_title("Ligand levels")
        axs[3, 0].set_xlabel("Time (min)")
        axs[3, 0].set_ylabel("Ligand levels (a.u.)")

        plt.tight_layout()
        save_dir = os.path.join(OUTPUT_DIR, "TF_modeling")
        plt.savefig(os.path.join(save_dir, plot_name))
        plt.close('all')


    initial_mRNA = np.array([0, 0, 0, 0])
    initial_protein = np.array([0, 0, 0, 0])
    initial_promoter = np.array([0, 0, 0, 0])
    run_multigen(initial_mRNA, initial_protein, initial_promoter, ligand_fxn,
                 n_gens, plot_name)



def run_model(total_time, plot_name):
    timestep = 0.1

    # Right now: think about the concentration, no cell division, etc.
    ## Parameters
    dilution_rate = 1 # 1/s
    deg_mRNA = 2 # 1/s
    RNAP_capacity = 10 # molecules/s
    rib_capacity = 100

    ks_unbound = np.array([1, 2, 3, 4])
    ks_bound = np.array([0.1, 2, 3, 4])
    teffs = np.array([1, 1, 1, 1])

    n_total_TFs = 10
    ligand_threshold = 100
    TF_bind_probs = np.array([0.5, 0.5, 0, 0])


    ## Functions for calculating derivatives
    def synth_rates_mRNA(ks, capacity):
        return capacity * ks / np.sum(ks)

    def loss_rates_mRNA(levels):
        return levels * (deg_mRNA + dilution_rate)

    def synth_rates_protein(levels, teffs, capacity):
        relat_rates = np.multiply(teffs, levels)
        if np.sum(relat_rates) > 0:
            return capacity * relat_rates / np.sum(relat_rates)
        return np.array([0., 0., 0., 0.])

    def loss_rates_protein(levels):
        return levels * dilution_rate

    def mRNA_rate(levels, ks, capacity):
        return synth_rates_mRNA(ks, capacity) - loss_rates_mRNA(levels)

    def protein_rate(levels_mRNA, levels_protein, teffs, capacity):
        return synth_rates_protein(levels_mRNA, teffs,
                                capacity) - loss_rates_protein(levels_protein)


    # At a given moment, we have n active TFs. Assume ligand binding
    # is instantaneous (?)
    # There are 4 promoters, 1 site of which could be bound.

    # Rn, assume TF binding/unbinding is instantaneous.
    # So when there's no ligand, there's no active TFs, and so none are bound.
    # When there's ligand, the TFs become active. So then they bind to DNA.
    # And they stay there. When ligand is gone again, the TFs will all lose their
    # ligand, and they'll unbind.


    def calculate_ks(promoters):
        ks = []
        for i, bound in enumerate(promoters):
            if bound == 1:
                ks.append(ks_bound[i])
            else:
                ks.append(ks_unbound[i])

        return np.array(ks)

    def promoter_state(promoter_previous, active_TFs):
        '''
        Given the number of active TFs, binds TFs to promoters.
        Uses the previous promoter state since TF binding is permanent.
        NOTE: we consider that TFs can either be all active or all inactive,
        and don't consider the dynamics of when a TF could alternate between
        active and inactive.
        '''
        num_bound = np.sum(promoter_previous)

        if num_bound == active_TFs:
            return promoter_previous
        elif num_bound < active_TFs:
            if active_TFs >= np.count_nonzero(TF_bind_probs):
                # Occupy all promoter sites if they can be
                return [int(x > 0) for x in TF_bind_probs]
            else:
                # Get number of extra TFs to bind
                TFs_to_bind = active_TFs - num_bound

                # Get which binding sites are open
                binding_sites = []
                for i in range(len(TF_bind_probs)):
                    if promoter_previous[i] == 0 and TF_bind_probs[i] > 0:
                        binding_sites.append(i)

                # Randomly select the binding sites to bind
                binding_probs = np.array([TF_bind_probs[i]
                                             for i in binding_sites])
                binding_probs = binding_probs / np.sum(binding_probs)
                bind_where = np.random.choice(binding_sites, TFs_to_bind,
                                              replace=False, p=binding_probs)

                # Update new promoter array
                new_promoter = copy.deepcopy(promoter_previous)
                for i in bind_where:
                    new_promoter[i] = 1

                return new_promoter
        else:
            # Get number of TFs to unbind
            TFs_to_unbind = num_bound - active_TFs

            # Get which binding sites are occupied
            binding_sites = np.nonzero(promoter_previous)[0]

            # Randomly select the binding sites to unbind, weighted opposite
            # to probabilities of binding
            unbinding_probs = np.array([1/TF_bind_probs[i]
                                         for i in binding_sites])
            unbinding_probs = unbinding_probs / np.sum(unbinding_probs)
            unbind_where = np.random.choice(binding_sites, TFs_to_unbind,
                                          replace=False, p=unbinding_probs)

            # Update new promoter array
            new_promoter = copy.deepcopy(promoter_previous)
            for i in unbind_where:
                new_promoter[i] = 0

            return new_promoter

    def activate_TFs(ligand, n_TFs):
        if ligand > ligand_threshold:
            return n_TFs
        else:
            return 0

    def run_simulation(mRNA_levels, protein_levels, t, RNAP_capacity,
                       rib_capacity, teffs, n_TFs,
                       ligand_timeseries, promoter_previous):
        active_TFs = activate_TFs(ligand_timeseries(t), n_TFs(t))
        promoter_new = promoter_state(promoter_previous, active_TFs)
        ks = calculate_ks(promoter_new)
        mRNA_rates = mRNA_rate(mRNA_levels, ks, RNAP_capacity)
        protein_rates = protein_rate(mRNA_levels, protein_levels,
                                     teffs, rib_capacity)


        return mRNA_rates, protein_rates, active_TFs, promoter_new

    ## Set initial conditions and run simulation
    init_mRNA_levels = np.array([0., 0., 0., 0.])
    init_protein_levels = np.array([0., 0., 0., 0.])
    time = np.arange(0, total_time, timestep)
    init_promoter = np.array([0, 0, 0, 0])

    # So ligand comes, the n_active_TFs should rise, then one step later they
    # should bind to promoters, then the rates should change and mRNAs should
    # change.

    def ligand_timeseries(t):
        if t > 20 and t < 30:
            return 0
        return 200

    def n_TFs(t):
        return 10
        # if t < total_time/4:
        #     return 1
        # if t < total_time/2:
        #     return 10
        # if t < total_time * 3/4:
        #     return 1
        # return 10

    active_TFs_record = []
    promoters_record = []
    mRNA_levels_record = []
    protein_levels_record = []

    mRNA_levels = init_mRNA_levels
    protein_levels = init_protein_levels
    promoters = init_promoter
    for t in time:
        mRNA_rates, protein_rates, active_TFs, promoter_new = run_simulation(
            mRNA_levels, protein_levels, t, RNAP_capacity, rib_capacity, teffs,
            n_TFs, ligand_timeseries, promoters)
        # Update promoter and mRNA levels
        mRNA_levels += mRNA_rates * timestep
        protein_levels += protein_rates * timestep
        promoters = promoter_new

        # Record relevant quantities
        active_TFs_record.append(active_TFs)
        promoters_record.append(copy.deepcopy(promoter_new))
        mRNA_levels_record.append(copy.deepcopy(mRNA_levels))
        protein_levels_record.append(copy.deepcopy(protein_levels))


    # results = scipy.integrate.odeint(run_simulation, init_levels,
    #                        time, args=(total_synthesis, n_total_TFs,
    #     ligand_timeseries, init_promoter))

    active_TFs_record = np.array(active_TFs_record)
    promoters_record = np.array(promoters_record)
    mRNA_levels_record = np.array(mRNA_levels_record)
    protein_levels_record = np.array(protein_levels_record)

    # Make plot
    fig, axs = plt.subplots(4, figsize=(5, 20))
    for i in range(4):
        axs[0].plot(time, mRNA_levels_record[:, i], label=str(i))
    axs[0].set_title("mRNA concentrations over time")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("mRNA concentrations")
    axs[0].legend()

    for i in range(4):
        axs[1].plot(time, protein_levels_record[:, i], label=str(i))
    axs[1].set_title("Protein concentrations over time")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Protein concentrations")
    axs[1].legend()

    axs[2].plot(time, active_TFs_record)
    axs[2].set_title("Number of active TFs over time")
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Number of active TFs")

    for i in range(4):
        axs[3].plot(time, promoters_record[:, i], label=str(i))
    axs[3].set_title("Promoter occupancy over time")
    axs[3].set_xlabel("Time")
    axs[3].set_ylabel("Promoter occupancy")
    axs[3].legend()

    plt.tight_layout()
    save_dir = os.path.join(OUTPUT_DIR, "TF_modeling")
    plt.savefig(os.path.join(save_dir, plot_name))
    plt.close('all')
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    def ligand_fxn(time):
        #return 0
        # if time < 240:
        #     return 0
        return 200
    stochastic_sim(60, "AutoregRegHighTF", ligand_fxn)
