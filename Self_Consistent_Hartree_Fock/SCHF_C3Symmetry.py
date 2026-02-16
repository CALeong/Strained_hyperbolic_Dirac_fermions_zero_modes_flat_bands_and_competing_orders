import numpy as np
import scipy
import time
from Fundamental.Hamiltonian import H0
from Fundamental.NonHermitian_Hamiltonian import site_assignment
from Self_Consistent_Hartree_Fock.Self_Consistent_Hartree_Fock import self_consistent_check
from Self_Consistent_Hartree_Fock.Self_Consistent_Hartree_Fock import site_assignment_NonHermitian
from Self_Consistent_Hartree_Fock.Self_Consistent_Hartree_Fock import hartreefock_hamiltonian_addition_NonHermitian
from Fundamental.Biorthogonal import biortogonal_normalize
from Fundamental.Number_Points import points
from Fundamental.Hubbard import add_onsite_hubbard_repulsion_hartree_decomposition
from Axial_Magnetic_Field.symmetry import c3symmetry_site_label_transform_sequenceofsites_hyperbolicq3

#Copy of selfconsist_hartreefock_NonHermitian_DisorderAlreadyAdded_SaveRawOutputs_MoreEfficient but with average over C3
def selfconsist_hartreefock_NonHermitian_DisorderAlreadyAdded_SaveRawOutputs_MoreEfficient_C3Average(p, q, num_levels, initial_ham, initial_guess, tolerance, hfcoeff_list):
    a_sites, b_sites = site_assignment_NonHermitian(p, q, num_levels)  # Label sites for two basis lattice
    deltas_raw_list = np.zeros((1, np.size(initial_ham, 1) + 1))
    deltas_list = np.array([])  # Holds final delta value for each hfcoeff

    asites_og, bsites_og = site_assignment(p, q, num_levels, H0(p, q, num_levels))
    sublatbasis = np.concatenate((asites_og, bsites_og))
    reversebasis = np.array([])
    for i in np.arange(np.size(initial_ham, 0)):
        reversebasis = np.append(reversebasis, np.where(i == sublatbasis)[0])
    sublatbasis = np.array([int(slb) for slb in sublatbasis])
    reversebasis = np.array([int(rb) for rb in reversebasis])

    c3basismat = c3symmetry_site_label_transform_sequenceofsites_hyperbolicq3(p, num_levels)

    for hfcoeff in hfcoeff_list:  # Go over all desired V values
        print(hfcoeff)
        converge_status = False
        deltas = np.array([])
        for d in range(np.size(initial_ham, 0)):
            if np.any(a_sites == d):
                deltas = np.append(deltas, initial_guess)
            else:
                deltas = np.append(deltas, -initial_guess)
        # deltas = np.repeat(initial_guess, np.size(initial_ham, 0)) #Initial guess
        ham = hartreefock_hamiltonian_addition_NonHermitian(p, q, num_levels, initial_ham, deltas, hfcoeff)  # HF Hamiltonian with initial guess
        converge_progress_count = 0
        while converge_status == False:
            start_time = time.time()
            # print('Diagonalization Start')
            eners, left_eigvec_matrix, right_eigvec_matrix = scipy.linalg.eig(ham, left=True, right=True)  # diagonalize ham to get energies and eigvecs
            del ham
            # print('Diagonalization Done')
            left_eigvec_matrix = np.conj(left_eigvec_matrix)
            left_eigvec_matrix, right_eigvec_matrix = biortogonal_normalize(left_eigvec_matrix, right_eigvec_matrix)
            # print('Normalization Done')
            negative_energy_indices = np.where(eners < 0)  # Find indices where E<0
            left_negative_energy_eigvec_matrix = left_eigvec_matrix[:, negative_energy_indices[0]]
            right_negative_energy_eigvec_matrix = right_eigvec_matrix[:, negative_energy_indices[0]]
            del eners
            del left_eigvec_matrix
            del right_eigvec_matrix
            del negative_energy_indices
            # print('Getting filled states done')

            site_probs = np.array([])
            left_negative_energy_eigvec_matrix_ogbasis = np.copy(left_negative_energy_eigvec_matrix)[reversebasis, :]
            right_negative_energy_eigvec_matrix_ogbasis = np.copy(right_negative_energy_eigvec_matrix)[reversebasis, :]

            for row in range(np.size(left_negative_energy_eigvec_matrix, 0)):
                site_probs = np.append(site_probs, np.sum(left_negative_energy_eigvec_matrix[row, :] * right_negative_energy_eigvec_matrix[row, :]))
            for rc in range(np.size(c3basismat, 0)):
                left_negative_energy_eigvec_matrix_rot = np.copy(left_negative_energy_eigvec_matrix_ogbasis)[c3basismat[rc, :], :]
                right_negative_energy_eigvec_matrix_rot = np.copy(right_negative_energy_eigvec_matrix_ogbasis)[c3basismat[rc, :], :]
                site_probs_rot_to_add = np.sum(left_negative_energy_eigvec_matrix_rot * right_negative_energy_eigvec_matrix_rot, axis=1)
                site_probs = site_probs + site_probs_rot_to_add[sublatbasis]
            site_probs = site_probs / (np.size(c3basismat, 0) + 1)
            # print('Calculating site probabilities done')

            prev_delta = deltas
            deltas = site_probs - 0.5
            del left_negative_energy_eigvec_matrix
            del right_negative_energy_eigvec_matrix
            del left_negative_energy_eigvec_matrix_rot
            del right_negative_energy_eigvec_matrix_rot
            del site_probs
            converge_status = self_consistent_check(deltas, prev_delta, tolerance)
            # print('Getting convergence status done')
            ham = hartreefock_hamiltonian_addition_NonHermitian(p, q, num_levels, initial_ham, deltas, hfcoeff)
            # print('Getting new hamiltonian done')
            converge_progress_count += 1
            print(converge_progress_count, np.max(np.abs(deltas - prev_delta)), 'Time taken: {}'.format(time.time() - start_time))
        print('Convergence done')
        a_deltas = np.array([])
        b_deltas = np.array([])
        for d in range(len(deltas)):
            if np.any(a_sites == d):
                a_deltas = np.append(a_deltas, deltas[d])
            else:
                b_deltas = np.append(b_deltas, deltas[d])
        a_delta_avg = np.abs(np.average(a_deltas))
        b_delta_avg = np.abs(np.average(b_deltas))
        deltas_list = np.append(deltas_list, 0.5 * (a_delta_avg + b_delta_avg))
        deltas_raw_list = np.vstack((deltas_raw_list, np.concatenate((np.array([hfcoeff]), deltas))))
    deltas_raw_list = deltas_raw_list[1:, :]
    return (deltas_raw_list, deltas_list)

#Copy of Hubbard SCHF code for hyperbolic but with adjustments for symmetry average
def SCHF_onsite_hubbard_disorderalreadyadded_saverawoutput_C3Average(p, q, num_levels, initial_ham, U_list, initial_guess, tolerance):
    system_deltaafm = np.array([])
    raw_deltaafm = np.zeros((1, np.size(initial_ham, 0) + 1))

    initial_guess_repeat_motif = np.repeat(initial_guess, int((2 * points(p, q, num_levels)[1]) / 4))
    initial_guess_array = np.concatenate((initial_guess_repeat_motif, -initial_guess_repeat_motif,
                                          -initial_guess_repeat_motif, initial_guess_repeat_motif))

    asites_og, bsites_og = site_assignment(p, q, num_levels, H0(p, q, num_levels))
    sublatbasis = np.concatenate((asites_og, bsites_og))
    sublatbasis = np.array([int(slb) for slb in sublatbasis])
    sublatbasis = np.hstack((sublatbasis, sublatbasis+int(np.size(initial_ham, 0)/2)))

    reversebasis = np.array([])
    for i in np.arange(np.size(initial_ham, 0)):
        reversebasis = np.append(reversebasis, np.where(i == sublatbasis)[0])
    reversebasis = np.array([int(rb) for rb in reversebasis])

    c3basismat_spinless = c3symmetry_site_label_transform_sequenceofsites_hyperbolicq3(p, num_levels)
    c3basismat = np.hstack((c3basismat_spinless, c3basismat_spinless + int(np.size(initial_ham, 0)/2)))
    del c3basismat_spinless

    for U in U_list:
        print('Currently calculating U = {}'.format(U))
        old_local_order_params = initial_guess_array
        ham = add_onsite_hubbard_repulsion_hartree_decomposition(p, q, num_levels, initial_ham, U, initial_guess_array)
        converge_status = False
        while converge_status == False:
            eigvals, ls_mat, rs_mat = scipy.linalg.eig(ham, left=True, right=True)
            del ham  ###CODE RAM FIX
            ls_mat = np.conj(ls_mat)
            ls_mat, rs_mat = biortogonal_normalize(ls_mat, rs_mat)
            negative_energy_indices = np.where(eigvals < 0)[0]
            ls_mat_filled = ls_mat[:, negative_energy_indices]
            rs_mat_filled = rs_mat[:, negative_energy_indices]

            site_probs = np.array([])
            ls_mat_filled_ogbasis = np.copy(ls_mat_filled)[reversebasis, :]
            rs_mat_filled_ogbasis = np.copy(rs_mat_filled)[reversebasis, :]
            for row in range(np.size(ls_mat_filled, 0)):
                site_probs = np.append(site_probs, np.sum(ls_mat_filled[row, :] * rs_mat_filled[row, :]))
            for rc in range(np.size(c3basismat, 0)):
                ls_mat_filled_rot = np.copy(ls_mat_filled_ogbasis)[c3basismat[rc, :], :]
                rs_mat_filled_rot = np.copy(rs_mat_filled_ogbasis)[c3basismat[rc, :], :]
                site_probs_rot_to_add = np.sum(ls_mat_filled_rot * rs_mat_filled_rot, axis=1)
                site_probs = site_probs + site_probs_rot_to_add[sublatbasis]
            site_probs = site_probs / (np.size(c3basismat, 0) + 1)

            del ls_mat  ###CODE RAM FIX
            del rs_mat  ###CODE RAM FIX
            del ls_mat_filled  ###CODE RAM FIX
            del rs_mat_filled  ###CODE RAM FIX
            del ls_mat_filled_ogbasis
            del rs_mat_filled_ogbasis
            del ls_mat_filled_rot
            del rs_mat_filled_rot
            del site_probs_rot_to_add
            del eigvals  ###CODE RAM FIX

            new_local_order_params = site_probs - 0.5
            del site_probs  ###CODE RAM FIX
            converge_status = self_consistent_check(new_local_order_params, old_local_order_params, tolerance)
            print('Max convergence difference: {}'.format(np.max(np.abs(new_local_order_params - old_local_order_params))))
            old_local_order_params = new_local_order_params
            ham = add_onsite_hubbard_repulsion_hartree_decomposition(p, q, num_levels, initial_ham, U, new_local_order_params)

        sublatticeA_upspin_avg = np.abs(np.average(new_local_order_params[:int(len(new_local_order_params) / 4)]))
        sublatticeA_downspin_avg = np.abs(np.average(new_local_order_params[int(len(new_local_order_params) / 2):int(3 * len(new_local_order_params) / 4)]))
        sublatticeB_upspin_avg = np.abs(np.average(new_local_order_params[int(len(new_local_order_params) / 4):int(len(new_local_order_params) / 2)]))
        sublatticeB_downspin_avg = np.abs(np.average(new_local_order_params[int(3 * len(new_local_order_params) / 4):]))
        sys_order_val = 0.5 * (sublatticeA_upspin_avg + sublatticeA_downspin_avg + sublatticeB_upspin_avg + sublatticeB_downspin_avg)
        system_deltaafm = np.append(system_deltaafm, sys_order_val)

        raw_deltaafm = np.vstack((raw_deltaafm, np.concatenate((np.array([U]), new_local_order_params))))

        del new_local_order_params  ###CODE RAM FIX

    raw_deltaafm = raw_deltaafm[1:, :]
    return (raw_deltaafm, system_deltaafm)
