import numpy as np
from Fundamental.NonHermitian_Hamiltonian import NonHermitian_Hamiltonian
from Fundamental.Number_Points import points
from Self_Consistent_Hartree_Fock.Self_Consistent_Hartree_Fock import self_consistent_check
import scipy
from Fundamental.Biorthogonal import biortogonal_normalize
from Fundamental.Square_Lattice import square_lattice_nonherm_PBC

#########################
#Hyperbolic Lattice Code#
#########################

def onsite_hubbard_tight_binding_mat(p, q, num_levels, alpha):
    h0 = NonHermitian_Hamiltonian(p, q, num_levels, alpha, 1)
    zero_block = np.zeros((np.size(h0, 0), np.size(h0, 1)))
    full_ham = np.block([[h0, zero_block], [zero_block, h0]])
    return(full_ham)

def add_onsite_hubbard_repulsion_hartree_decomposition(p, q, num_levels, initial_ham, U, local_order_params):
    sublatA_upspin_diag_vals = local_order_params[int(points(p,q,num_levels)[1]):int(3*points(p,q,num_levels)[1]/2)]
    sublatA_downspin_diag_vals = local_order_params[:int((points(p,q,num_levels)[1]) / 2)]
    sublatB_upspin_diag_vals = local_order_params[int((3*points(p, q, num_levels)[1]) / 2):]
    sublatB_downspin_diag_vals = local_order_params[int(points(p,q,num_levels)[1]/2):int(points(p, q, num_levels)[1])]
    onsite_hubbard_diag_vals = np.concatenate((sublatA_upspin_diag_vals, sublatB_upspin_diag_vals,
                                               sublatA_downspin_diag_vals, sublatB_downspin_diag_vals
                                               ))
    onsite_hubbard_mat = U * np.diag(onsite_hubbard_diag_vals)

    hartree_fock_ham = initial_ham + onsite_hubbard_mat
    return (hartree_fock_ham)

def SCHF_onsite_hubbard_disorderalreadyadded_saverawoutput(p, q, num_levels, initial_ham, U_list, initial_guess, tolerance):

    system_deltaafm = np.array([])
    raw_deltaafm = np.zeros((1,np.size(initial_ham,0)+1))

    initial_guess_repeat_motif = np.repeat(initial_guess, int((2*points(p,q,num_levels)[1])/4))
    initial_guess_array = np.concatenate((initial_guess_repeat_motif, -initial_guess_repeat_motif,
                                          -initial_guess_repeat_motif, initial_guess_repeat_motif))
    for U in U_list:
        print('Currently calculating U = {}'.format(U))
        old_local_order_params = initial_guess_array
        ham = add_onsite_hubbard_repulsion_hartree_decomposition(p,q,num_levels,initial_ham,U,initial_guess_array)
        converge_status = False
        while converge_status == False:
            eigvals, ls_mat, rs_mat = scipy.linalg.eig(ham, left=True, right=True)
            del ham ###CODE RAM FIX
            ls_mat = np.conj(ls_mat)
            ls_mat, rs_mat = biortogonal_normalize(ls_mat, rs_mat)
            negative_energy_indices = np.where(eigvals<0)[0]
            ls_mat_filled = ls_mat[:, negative_energy_indices]
            rs_mat_filled = rs_mat[:, negative_energy_indices]
            site_probs = np.array([])
            for row in range(np.size(ls_mat_filled,0)):
                site_probs = np.append(site_probs, np.sum(ls_mat_filled[row, :]*rs_mat_filled[row, :]))
            del ls_mat ###CODE RAM FIX
            del rs_mat ###CODE RAM FIX
            del ls_mat_filled ###CODE RAM FIX
            del rs_mat_filled ###CODE RAM FIX
            del eigvals ###CODE RAM FIX
            new_local_order_params = site_probs - 0.5
            del site_probs ###CODE RAM FIX
            converge_status = self_consistent_check(new_local_order_params, old_local_order_params, tolerance)
            print('Max convergence difference: {}'.format(np.max(np.abs(new_local_order_params-old_local_order_params))))
            old_local_order_params = new_local_order_params
            ham = add_onsite_hubbard_repulsion_hartree_decomposition(p,q,num_levels,initial_ham,U,new_local_order_params)

        sublatticeA_upspin_avg = np.abs(np.average(new_local_order_params[:int(len(new_local_order_params)/4)]))
        sublatticeA_downspin_avg = np.abs(np.average(new_local_order_params[int(len(new_local_order_params)/2):int(3*len(new_local_order_params)/4)]))
        sublatticeB_upspin_avg = np.abs(np.average(new_local_order_params[int(len(new_local_order_params)/4):int(len(new_local_order_params)/2)]))
        sublatticeB_downspin_avg = np.abs(np.average(new_local_order_params[int(3*len(new_local_order_params)/4):]))
        sys_order_val = 0.5 * (sublatticeA_upspin_avg + sublatticeA_downspin_avg + sublatticeB_upspin_avg + sublatticeB_downspin_avg)
        system_deltaafm = np.append(system_deltaafm, sys_order_val)

        raw_deltaafm = np.vstack((raw_deltaafm, np.concatenate((np.array([U]), new_local_order_params))))

        del new_local_order_params ###CODE RAM FIX

    raw_deltaafm = raw_deltaafm[1:,:]
    return(raw_deltaafm, system_deltaafm)

################
#Square Lattice#
################

from Fundamental.Square_Lattice import square_lattice_nonherm
from Fundamental.Square_Lattice import number_square_points

def onsite_hubbard_tight_binding_mat_square(nl, alpha):
    h0 = square_lattice_nonherm(nl, alpha)
    zero_block = np.zeros((np.size(h0, 0), np.size(h0, 1)))
    full_ham = np.block([[h0, zero_block], [zero_block, h0]])
    return(full_ham)

def add_onsite_hubbard_repulsion_hartree_decomposition_square(nl, initial_ham, U, local_order_params):
    sublatA_upspin_diag_vals = local_order_params[int(number_square_points(nl)):int(3*number_square_points(nl)/2)]
    sublatA_downspin_diag_vals = local_order_params[:int(number_square_points(nl) / 2)]
    sublatB_upspin_diag_vals = local_order_params[int((3*number_square_points(nl)) / 2):]
    sublatB_downspin_diag_vals = local_order_params[int(number_square_points(nl)/2):int(number_square_points(nl))]
    onsite_hubbard_diag_vals = np.concatenate((sublatA_upspin_diag_vals, sublatB_upspin_diag_vals,
                                               sublatA_downspin_diag_vals, sublatB_downspin_diag_vals
                                               ))
    onsite_hubbard_mat = U * np.diag(onsite_hubbard_diag_vals)

    hartree_fock_ham = initial_ham + onsite_hubbard_mat
    return (hartree_fock_ham)

def SCHF_onsite_hubbard_disorderalreadyadded_saverawoutput_square(nl, initial_ham, U_list, initial_guess, tolerance):

    system_deltaafm = np.array([])
    raw_deltaafm = np.zeros((1,np.size(initial_ham,0)+1))

    initial_guess_repeat_motif = np.repeat(initial_guess, int((2*number_square_points(nl))/4))
    initial_guess_array = np.concatenate((initial_guess_repeat_motif, -initial_guess_repeat_motif,
                                          -initial_guess_repeat_motif, initial_guess_repeat_motif))
    for U in U_list:
        print('Currently calculating U = {}'.format(U))
        old_local_order_params = initial_guess_array
        ham = add_onsite_hubbard_repulsion_hartree_decomposition_square(nl,initial_ham,U,initial_guess_array)
        converge_status = False
        while converge_status == False:
            eigvals, ls_mat, rs_mat = scipy.linalg.eig(ham, left=True, right=True)
            del ham ###RAM Code Fix
            ls_mat = np.conj(ls_mat)
            ls_mat, rs_mat = biortogonal_normalize(ls_mat, rs_mat)
            negative_energy_indices = np.where(eigvals<0)[0]
            ls_mat_filled = ls_mat[:, negative_energy_indices]
            rs_mat_filled = rs_mat[:, negative_energy_indices]
            site_probs = np.array([])
            for row in range(np.size(ls_mat_filled,0)):
                site_probs = np.append(site_probs, np.sum(ls_mat_filled[row, :]*rs_mat_filled[row, :]))
            del ls_mat ###RAM Code Fix
            del rs_mat ###RAM Code Fix
            del ls_mat_filled ###RAM Code Fix
            del rs_mat_filled ###RAM Code Fix
            del eigvals ###RAM Code Fix
            new_local_order_params = site_probs - 0.5
            del site_probs ###RAM Code Fix
            converge_status = self_consistent_check(new_local_order_params, old_local_order_params, tolerance)
            print('Max convergence difference: {}'.format(np.max(np.abs(new_local_order_params-old_local_order_params))))
            old_local_order_params = new_local_order_params
            ham = add_onsite_hubbard_repulsion_hartree_decomposition_square(nl,initial_ham,U,new_local_order_params)

        sublatticeA_upspin_avg = np.abs(np.average(new_local_order_params[:int(len(new_local_order_params)/4)]))
        sublatticeA_downspin_avg = np.abs(np.average(new_local_order_params[int(len(new_local_order_params)/2):int(3*len(new_local_order_params)/4)]))
        sublatticeB_upspin_avg = np.abs(np.average(new_local_order_params[int(len(new_local_order_params)/4):int(len(new_local_order_params)/2)]))
        sublatticeB_downspin_avg = np.abs(np.average(new_local_order_params[int(3*len(new_local_order_params)/4):]))
        sys_order_val = 0.5 * (sublatticeA_upspin_avg + sublatticeA_downspin_avg + sublatticeB_upspin_avg + sublatticeB_downspin_avg)
        system_deltaafm = np.append(system_deltaafm, sys_order_val)

        raw_deltaafm = np.vstack((raw_deltaafm, np.concatenate((np.array([U]), new_local_order_params))))

    raw_deltaafm = raw_deltaafm[1:,:]
    return(raw_deltaafm, system_deltaafm)

###########
#Honeycomb#
###########

from Fundamental.NonHermitian_Hamiltonian import NonHermitian_Honeycomb_periodic_boundary
from Fundamental.Honeycomb_Lattice import honeycomb_points

def onsite_hubbard_tight_binding_mat_honeycomb(nl, alpha):
    h0 = NonHermitian_Honeycomb_periodic_boundary(nl, 1, alpha)
    zero_block = np.zeros((np.size(h0, 0), np.size(h0, 1)))
    full_ham = np.block([[h0, zero_block], [zero_block, h0]])
    return(full_ham)

def add_onsite_hubbard_repulsion_hartree_decomposition_honeycomb(nl, initial_ham, U, local_order_params):
    sublatA_upspin_diag_vals = local_order_params[int(honeycomb_points(nl)[1]):int(3*honeycomb_points(nl)[1]/2)]
    sublatA_downspin_diag_vals = local_order_params[:int((honeycomb_points(nl)[1]) / 2)]
    sublatB_upspin_diag_vals = local_order_params[int((3*honeycomb_points(nl)[1]) / 2):]
    sublatB_downspin_diag_vals = local_order_params[int(honeycomb_points(nl)[1]/2):int(honeycomb_points(nl)[1])]
    onsite_hubbard_diag_vals = np.concatenate((sublatA_upspin_diag_vals, sublatB_upspin_diag_vals,
                                               sublatA_downspin_diag_vals, sublatB_downspin_diag_vals
                                               ))
    onsite_hubbard_mat = U * np.diag(onsite_hubbard_diag_vals)

    hartree_fock_ham = initial_ham + onsite_hubbard_mat
    return (hartree_fock_ham)

def SCHF_onsite_hubbard_disorderalreadyadded_saverawoutput_honeycomb(nl, initial_ham, U_list, initial_guess, tolerance):

    system_deltaafm = np.array([])
    raw_deltaafm = np.zeros((1,np.size(initial_ham,0)+1))

    initial_guess_repeat_motif = np.repeat(initial_guess, int((2*honeycomb_points(nl)[1])/4))
    initial_guess_array = np.concatenate((initial_guess_repeat_motif, -initial_guess_repeat_motif,
                                          -initial_guess_repeat_motif, initial_guess_repeat_motif))
    for U in U_list:
        print('Currently calculating U = {}'.format(U))
        old_local_order_params = initial_guess_array
        ham = add_onsite_hubbard_repulsion_hartree_decomposition_honeycomb(nl,initial_ham,U,initial_guess_array)
        converge_status = False
        while converge_status == False:
            eigvals, ls_mat, rs_mat = scipy.linalg.eig(ham, left=True, right=True)
            del ham ###CODE RAM FIX
            ls_mat = np.conj(ls_mat)
            ls_mat, rs_mat = biortogonal_normalize(ls_mat, rs_mat)
            negative_energy_indices = np.where(eigvals<0)[0]
            ls_mat_filled = ls_mat[:, negative_energy_indices]
            rs_mat_filled = rs_mat[:, negative_energy_indices]
            site_probs = np.array([])
            for row in range(np.size(ls_mat_filled,0)):
                site_probs = np.append(site_probs, np.sum(ls_mat_filled[row, :]*rs_mat_filled[row, :]))
            del ls_mat ###CODE RAM FIX
            del rs_mat ###CODE RAM FIX
            del ls_mat_filled ###CODE RAM FIX
            del rs_mat_filled ###CODE RAM FIX
            del eigvals ###CODE RAM FIX
            new_local_order_params = site_probs - 0.5
            del site_probs
            converge_status = self_consistent_check(new_local_order_params, old_local_order_params, tolerance)
            print('Max convergence difference: {}'.format(np.max(np.abs(new_local_order_params-old_local_order_params))))
            old_local_order_params = new_local_order_params
            ham = add_onsite_hubbard_repulsion_hartree_decomposition_honeycomb(nl,initial_ham,U,new_local_order_params)

        sublatticeA_upspin_avg = np.abs(np.average(new_local_order_params[:int(len(new_local_order_params)/4)]))
        sublatticeA_downspin_avg = np.abs(np.average(new_local_order_params[int(len(new_local_order_params)/2):int(3*len(new_local_order_params)/4)]))
        sublatticeB_upspin_avg = np.abs(np.average(new_local_order_params[int(len(new_local_order_params)/4):int(len(new_local_order_params)/2)]))
        sublatticeB_downspin_avg = np.abs(np.average(new_local_order_params[int(3*len(new_local_order_params)/4):]))
        sys_order_val = 0.5 * (sublatticeA_upspin_avg + sublatticeA_downspin_avg + sublatticeB_upspin_avg + sublatticeB_downspin_avg)
        system_deltaafm = np.append(system_deltaafm, sys_order_val)

        raw_deltaafm = np.vstack((raw_deltaafm, np.concatenate((np.array([U]), new_local_order_params))))

    raw_deltaafm = raw_deltaafm[1:,:]
    return(raw_deltaafm, system_deltaafm)

################
#Bernal Bilayer#
################

from Fundamental.NonHermitian_Hamiltonian import NonHermitian_Bilayer_Honeycomb_PBC

def onsite_hubbard_tight_binding_mat_bernalbilayer(nl, alpha):
    h0 = NonHermitian_Bilayer_Honeycomb_PBC(nl, 1, alpha)
    zero_block = np.zeros((np.size(h0, 0), np.size(h0, 1)))
    full_ham = np.block([[h0, zero_block], [zero_block, h0]])
    return(full_ham)

def add_onsite_hubbard_repulsion_hartree_decomposition_bernalbilayer(nl, initial_ham, U, local_order_params):
    number_sites_bilayer = 2*honeycomb_points(nl)[1]

    sublatA1_upspin_diag_vals = local_order_params[int(number_sites_bilayer):int(5*number_sites_bilayer/4)]
    sublatA2_upspin_diag_vals = local_order_params[int(3*number_sites_bilayer/2):int(7*number_sites_bilayer/4)]
    sublatB1_upspin_diag_vals = local_order_params[int(5*number_sites_bilayer/4):int(3*number_sites_bilayer/2)]
    sublatB2_upspin_diag_vals = local_order_params[int(7*number_sites_bilayer/4):]

    sublatA1_downspin_diag_vals = local_order_params[:int(number_sites_bilayer/4)]
    sublatA2_downspin_diag_vals = local_order_params[int(number_sites_bilayer/2):int(3*number_sites_bilayer/4)]
    sublatB1_downspin_diag_vals = local_order_params[int(number_sites_bilayer/4):int(number_sites_bilayer/2)]
    sublatB2_downspin_diag_vals = local_order_params[int(3*number_sites_bilayer/4):int(number_sites_bilayer)]

    onsite_hubbard_diag_vals = np.concatenate((sublatA1_upspin_diag_vals, sublatB1_upspin_diag_vals,
                                               sublatA2_upspin_diag_vals, sublatB2_upspin_diag_vals,
                                               sublatA1_downspin_diag_vals, sublatB1_downspin_diag_vals,
                                               sublatA2_downspin_diag_vals, sublatB2_downspin_diag_vals
                                               ))
    onsite_hubbard_mat = U * np.diag(onsite_hubbard_diag_vals)

    hartree_fock_ham = initial_ham + onsite_hubbard_mat
    return (hartree_fock_ham)

def SCHF_onsite_hubbard_disorderalreadyadded_saverawoutput_bernalbilayer(nl, initial_ham, U_list, initial_guess, tolerance):

    system_deltaafm = np.array([])
    raw_deltaafm = np.zeros((1,np.size(initial_ham,0)+1))

    number_sites_bilayer = 2*honeycomb_points(nl)[1]
    initial_guess_repeat_motif = np.repeat(initial_guess, int(number_sites_bilayer/4))
    initial_guess_array = np.concatenate((initial_guess_repeat_motif, -initial_guess_repeat_motif,
                                          -initial_guess_repeat_motif, initial_guess_repeat_motif,
                                          -initial_guess_repeat_motif, initial_guess_repeat_motif,
                                          initial_guess_repeat_motif, -initial_guess_repeat_motif,
                                          ))
    for U in U_list:
        print('Currently calculating U = {}'.format(U))
        old_local_order_params = initial_guess_array
        ham = add_onsite_hubbard_repulsion_hartree_decomposition_bernalbilayer(nl,initial_ham,U,initial_guess_array)
        converge_status = False
        while converge_status == False:
            eigvals, ls_mat, rs_mat = scipy.linalg.eig(ham, left=True, right=True)
            del ham ###RAM Code Fix
            ls_mat = np.conj(ls_mat)
            ls_mat, rs_mat = biortogonal_normalize(ls_mat, rs_mat)
            negative_energy_indices = np.where(eigvals<0)[0]
            ls_mat_filled = ls_mat[:, negative_energy_indices]
            rs_mat_filled = rs_mat[:, negative_energy_indices]
            site_probs = np.array([])
            for row in range(np.size(ls_mat_filled,0)):
                site_probs = np.append(site_probs, np.sum(ls_mat_filled[row, :]*rs_mat_filled[row, :]))
            del ls_mat ###RAM Code Fix
            del rs_mat ###RAM Code Fix
            del ls_mat_filled ###RAM Code Fix
            del rs_mat_filled ###RAM Code Fix
            del eigvals ###RAM Code Fix
            new_local_order_params = site_probs - 0.5
            del site_probs
            converge_status = self_consistent_check(new_local_order_params, old_local_order_params, tolerance)
            print('Max convergence difference: {}'.format(np.max(np.abs(new_local_order_params-old_local_order_params))))
            old_local_order_params = new_local_order_params
            ham = add_onsite_hubbard_repulsion_hartree_decomposition_bernalbilayer(nl,initial_ham,U,new_local_order_params)

        sublatticeA1_upspin_avg = np.abs(np.average(new_local_order_params[:int(number_sites_bilayer/4)]))
        sublatticeB1_upspin_avg = np.abs(np.average(new_local_order_params[int(number_sites_bilayer/4):int(number_sites_bilayer/2)]))
        sublatticeA2_upspin_avg = np.abs(np.average(new_local_order_params[int(number_sites_bilayer/2):int(3*number_sites_bilayer/4)]))
        sublatticeB2_upspin_avg = np.abs(np.average(new_local_order_params[int(3*number_sites_bilayer/4):int(number_sites_bilayer)]))

        sublatticeA1_downspin_avg = np.abs(np.average(new_local_order_params[int(number_sites_bilayer):int(5*number_sites_bilayer/4)]))
        sublatticeB1_downspin_avg = np.abs(np.average(new_local_order_params[int(5*number_sites_bilayer/4):int(3*number_sites_bilayer/2)]))
        sublatticeA2_downspin_avg = np.abs(np.average(new_local_order_params[int(3*number_sites_bilayer/2):int(7*number_sites_bilayer/4)]))
        sublatticeB2_downspin_avg = np.abs(np.average(new_local_order_params[int(7*number_sites_bilayer/4):int(2*number_sites_bilayer)]))

        sys_order_val = 0.25 * (sublatticeA1_upspin_avg + sublatticeA1_downspin_avg +
                                sublatticeB1_upspin_avg + sublatticeB1_downspin_avg +
                                sublatticeA2_upspin_avg + sublatticeA2_downspin_avg +
                                sublatticeB2_upspin_avg + sublatticeB2_downspin_avg
                                )
        system_deltaafm = np.append(system_deltaafm, sys_order_val)

        raw_deltaafm = np.vstack((raw_deltaafm, np.concatenate((np.array([U]), new_local_order_params))))

    raw_deltaafm = raw_deltaafm[1:,:]
    return(raw_deltaafm, system_deltaafm)

#################################
#Hyperbolic Peierls Substitution#
#################################

from Fundamental.NonHermitian_Hamiltonian import NonHermitian_PeierlsSub_Hamiltonian

def onsite_hubbard_tight_binding_mat_peierls_substitution(p, q, num_levels, alpha, alphamag):
    h0 = NonHermitian_PeierlsSub_Hamiltonian(p, q, num_levels, alpha, 1, alphamag)
    zero_block = np.zeros((np.size(h0, 0), np.size(h0, 1)))
    full_ham = np.block([[h0, zero_block], [zero_block, h0]])
    return (full_ham)

#"add_onsite_hubbard_repulsion_hartree_decomposition" should still work for peierls sub case as long as initial_ham
#is the generated from "onsite_hubbard_tight_binding_mat_peierls_substitution"

#"SCHF_onsite_hubbard_disorderalreadyadded_saverawoutput" should also still work for peierls sub case as long as
#initial_ham is generated from "onsite_hubbard_tight_binding_mat_peierls_substitution"

##########################################
#Euclidean Honeycomb Peierls Substitution#
##########################################

from Fundamental.NonHermitian_Hamiltonian import NonHermitian_PeierlsSub_Honeycomb

def onsite_hubbard_tight_binding_mat_peierls_substitution_honeycomb(nl, alpha, alphamag):
    h0 = NonHermitian_PeierlsSub_Honeycomb(nl, alpha, alphamag)
    zero_block = np.zeros((np.size(h0, 0), np.size(h0, 1)))
    full_ham = np.block([[h0, zero_block], [zero_block, h0]])
    return (full_ham)

#"add_onsite_hubbard_repulsion_hartree_decomposition_honeycomb" should still work for peierls substitution case so long as
#initial_ham is generated from "onsite_hubbard_tight_binding_mat_peierls_substitution_honeycomb"

#"SCHF_onsite_hubbard_disorderalreadyadded_saverawoutput_honeycomb" should also still work for peierls substitution case
# so long as initial_ham is generated from "onsite_hubbard_tight_binding_mat_peierls_substitution_honeycomb"

############################
#Euclidean Honeycomb OpenBC#
############################

from Fundamental.NonHermitian_Hamiltonian import NonHermitian_Honeycomb

def onsite_hubbard_tight_binding_mat_honeycomb_openbc(nl, alpha):
    h0 = NonHermitian_Honeycomb(nl, 1, alpha)
    zeroblock = np.zeros((np.size(h0, 0), np.size(h0, 1)))
    full_ham = np.block([[h0, zeroblock], [zeroblock, h0]])
    return(full_ham)

#The original honeycomb "add_onsite_hubbard_repulsion_hartree_decomposition_honeycomb" and
#"SCHF_onsite_hubbard_disorderalreadyadded_saverawoutput_honeycomb" should still work when used with
#"onsite_hubbard_tight_binding_mat_honeycomb_openbc"

####################
#Square Lattice PBC#
####################

from Fundamental.Square_Lattice import square_lattice_nonherm
from Fundamental.Square_Lattice import number_square_points

def onsite_hubbard_tight_binding_mat_square_PBC(nl, alpha):
    h0 = square_lattice_nonherm_PBC(nl, alpha)
    zero_block = np.zeros((np.size(h0, 0), np.size(h0, 1)))
    full_ham = np.block([[h0, zero_block], [zero_block, h0]])
    return(full_ham)

#add_onsite_hubbard_repulsion_hartree_decomposition_square should still work for PBC case
#SCHF_onsite_hubbard_disorderalreadyadded_saverawoutput_square should also still work for PBC case
