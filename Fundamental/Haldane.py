import numpy as np
from Fundamental.General_Hamiltonian import get_if_point_is_connected_with_upper_layer_q3_general
from Fundamental.General_Hamiltonian import get_points_that_connect_with_prev_layer_q3_general
from Fundamental.Number_Points import points
from Fundamental.NonHermitian_Hamiltonian import site_assignment
import time
import scipy
from Fundamental.Biorthogonal import biortogonal_normalize
from Fundamental.Hamiltonian import H0
from Fundamental.Local import get_plaquet_boundary_sites_hyperbolicq3
from Fundamental.Local import get_nnn_hoppings_around_plaquet_sublatticebasis_hyperbolicq3
from Fundamental.Local import calculate_haldane_current_around_plaquet_sublatticebasis

def haldane_current_hyperbolicq3(p, num_levels, tight_bind_ham, t2):
    haldane_mat = np.copy(tight_bind_ham).astype(np.complex_)

    points_per_level = points(p, 3, num_levels)[0]

    # points_conn_next_level = sites on current gen that connect to above gen sites
    # points_that_conn_from_above = sites on next gen that connect down to current gen

    # Hard code first gen since get_if_point_is_connected_with_upper_layer_q3_general does not work for first gen
    sites_on_level = np.arange(0, p, dtype=int)
    sites_on_next_level = np.arange(p, np.sum(points_per_level[:2]), dtype=int)
    if_points_conn_next_level = get_if_point_is_connected_with_upper_layer_q3_general(p, 0)
    if_points_conn_next_level[0] = True  # Hard code fix
    points_conn_next_level = sites_on_level[if_points_conn_next_level]
    points_that_conn_from_above = get_points_that_connect_with_prev_layer_q3_general(p, 1, p)
    for i in range(p):  # intragen nnn hopping
        haldane_mat[i, np.take(sites_on_level, i + 2, mode='wrap')] = 1j * t2
        haldane_mat[np.take(sites_on_level, i + 2, mode='wrap'), i] = np.conj(1j * t2)
    for i in range(len(points_conn_next_level)):  # intergen nnn hopping
        next_level_site_connected_index = np.where(sites_on_next_level == points_that_conn_from_above[i])[0]
        nnn_sites_plusone = np.take(sites_on_next_level, next_level_site_connected_index + 1, mode='wrap')
        nnn_sites_minusone = np.take(sites_on_next_level, next_level_site_connected_index - 1, mode='wrap')
        haldane_mat[points_conn_next_level[i], nnn_sites_plusone] = 1j * t2
        haldane_mat[nnn_sites_plusone, points_conn_next_level[i]] = np.conj(1j * t2)
        haldane_mat[points_conn_next_level[i], nnn_sites_minusone] = np.conj(1j * t2)
        haldane_mat[nnn_sites_minusone, points_conn_next_level[i]] = 1j * t2
        # Finally need to take into account nnn hopping for adjacent points_that_conn_from_above
        nnn_sites_plusplusone = np.take(points_that_conn_from_above, i + 1, mode='wrap')
        nnn_sites_minusminusone = np.take(points_that_conn_from_above, i - 1, mode='wrap')
        haldane_mat[points_conn_next_level[i], nnn_sites_plusplusone] = np.conj(1j * t2)
        haldane_mat[nnn_sites_plusplusone, points_conn_next_level[i]] = 1j * t2
        haldane_mat[points_conn_next_level[i], nnn_sites_minusminusone] = 1j * t2
        haldane_mat[nnn_sites_minusminusone, points_conn_next_level[i]] = np.conj(1j * t2)

    for nl in range(1, num_levels - 1):  # Handles all other generations except the last gen
        sites_on_level = np.arange(np.sum(points_per_level[:nl]), np.sum(points_per_level[:nl + 1]), dtype=int)
        sites_on_next_level = np.arange(np.sum(points_per_level[:nl + 1]), np.sum(points_per_level[:nl + 2]), dtype=int)
        if_points_conn_next_level = get_if_point_is_connected_with_upper_layer_q3_general(p, nl)
        points_conn_next_level = sites_on_level[if_points_conn_next_level]
        points_that_conn_from_above = get_points_that_connect_with_prev_layer_q3_general(p, nl + 1, sites_on_next_level[0])
        for i in range(len(sites_on_level)):  # intragen nnn hopping
            if np.take(sites_on_level, i+1, mode='wrap') in points_conn_next_level:
                haldane_mat[sites_on_level[i], np.take(sites_on_level, i + 2, mode='wrap')] = 1j * t2
                haldane_mat[np.take(sites_on_level, i + 2, mode='wrap'), sites_on_level[i]] = np.conj(1j * t2)
            else:
                haldane_mat[sites_on_level[i], np.take(sites_on_level, i + 2, mode='wrap')] = np.conj(1j * t2)
                haldane_mat[np.take(sites_on_level, i + 2, mode='wrap'), sites_on_level[i]] = 1j * t2

        # intergen nnn hopping
        for i in range(len(points_conn_next_level)):
            next_level_site_connected_index = np.where(sites_on_next_level == points_that_conn_from_above[i])[0]
            nnn_sites_plusone = np.take(sites_on_next_level, next_level_site_connected_index + 1, mode='wrap')
            nnn_sites_minusone = np.take(sites_on_next_level, next_level_site_connected_index - 1, mode='wrap')
            haldane_mat[points_conn_next_level[i], nnn_sites_plusone] = 1j * t2
            haldane_mat[nnn_sites_plusone, points_conn_next_level[i]] = np.conj(1j * t2)
            haldane_mat[points_conn_next_level[i], nnn_sites_minusone] = np.conj(1j * t2)
            haldane_mat[nnn_sites_minusone, points_conn_next_level[i]] = 1j * t2
            # Need to take into account nnn hopping for adjacent points_that_conn_from_above
            # If and elifs to avoid problems for points_conn_next_level on four-siders
            if (points_conn_next_level[i] - np.take(points_conn_next_level, i - 1, mode='wrap') == 1) and (np.take(points_conn_next_level, i + 1, mode='wrap') - points_conn_next_level[i] == 1):
                nnn_sites_plusplusone = np.take(points_that_conn_from_above, i + 1, mode='wrap')
                nnn_sites_minusminusone = np.take(points_that_conn_from_above, i - 1, mode='wrap')
                haldane_mat[points_conn_next_level[i], nnn_sites_plusplusone] = np.conj(1j * t2)
                haldane_mat[nnn_sites_plusplusone, points_conn_next_level[i]] = 1j * t2
                haldane_mat[points_conn_next_level[i], nnn_sites_minusminusone] = 1j * t2
                haldane_mat[nnn_sites_minusminusone, points_conn_next_level[i]] = np.conj(1j * t2)
            elif points_conn_next_level[i] - np.take(points_conn_next_level, i - 1, mode='wrap') == 1:
                nnn_sites_minusminusone = np.take(points_that_conn_from_above, i - 1, mode='wrap')
                haldane_mat[points_conn_next_level[i], nnn_sites_minusminusone] = 1j * t2
                haldane_mat[nnn_sites_minusminusone, points_conn_next_level[i]] = np.conj(1j * t2)
                # haldane_mat[points_conn_next_level[i], np.take(sites_on_level, np.where(sites_on_level == points_conn_next_level[i])[0] + 2, mode='wrap')] = np.conj(1j * t2)
                # haldane_mat[np.take(sites_on_level, np.where(sites_on_level == points_conn_next_level[i])[0] + 2, mode='wrap'), points_conn_next_level[i]] = 1j * t2
            elif np.take(points_conn_next_level, i + 1, mode='wrap') - points_conn_next_level[i] == 1:
                nnn_sites_plusplusone = np.take(points_that_conn_from_above, i + 1, mode='wrap')
                haldane_mat[points_conn_next_level[i], nnn_sites_plusplusone] = np.conj(1j * t2)
                haldane_mat[nnn_sites_plusplusone, points_conn_next_level[i]] = 1j * t2
                # haldane_mat[points_conn_next_level[i], np.take(sites_on_level, np.where(sites_on_level == points_conn_next_level[i])[0] - 2, mode='wrap')] = 1j * t2
                # haldane_mat[
                #     np.take(sites_on_level, np.where(sites_on_level == points_conn_next_level[i])[0] - 2, mode='wrap'),
                #     points_conn_next_level[i]] = np.conj(1j * t2)
        # Finally need to account for sites on gen that do not connect via NN hopping to next gen
        points_not_conn_next_level = np.setdiff1d(sites_on_level, points_conn_next_level)
        for pncnl in points_not_conn_next_level:
            pncnl_index = np.where(sites_on_level == pncnl)[0]
            siteoneabove = np.take(sites_on_level, pncnl_index + 1, mode='wrap')
            siteonebelow = np.take(sites_on_level, pncnl_index - 1, mode='wrap')
            nnnnextgenoneabovesite = points_that_conn_from_above[np.where(points_conn_next_level == siteoneabove)[0]]
            nnnnextgenonebelowsite = points_that_conn_from_above[np.where(points_conn_next_level == siteonebelow)[0]]
            haldane_mat[pncnl, nnnnextgenoneabovesite] = np.conj(1j * t2)
            haldane_mat[nnnnextgenoneabovesite, pncnl] = 1j * t2
            haldane_mat[pncnl, nnnnextgenonebelowsite] = 1j * t2
            haldane_mat[nnnnextgenonebelowsite, pncnl] = np.conj(1j * t2)

    # Finally handle last gen
    sites_on_level = np.arange(np.sum(points_per_level[:num_levels-1]), np.sum(points_per_level), dtype=int)
    if_points_conn_next_level = get_if_point_is_connected_with_upper_layer_q3_general(p, num_levels - 1)
    points_conn_next_level = sites_on_level[if_points_conn_next_level]
    for i in range(len(sites_on_level)):  # intragen nnn hopping (all intergen hopping already taken care of)
        if np.take(sites_on_level, i+1, mode='wrap') in points_conn_next_level:
            haldane_mat[sites_on_level[i], np.take(sites_on_level, i + 2, mode='wrap')] = 1j * t2
            haldane_mat[np.take(sites_on_level, i + 2, mode='wrap'), sites_on_level[i]] = np.conj(1j * t2)
        else:
            haldane_mat[sites_on_level[i], np.take(sites_on_level, i + 2, mode='wrap')] = np.conj(1j * t2)
            haldane_mat[np.take(sites_on_level, i + 2, mode='wrap'), sites_on_level[i]] = 1j * t2

    #Last bit of code to reverse sign of phase for NNN hopping on B sublattice
    asites, bsites = site_assignment(p, 3, num_levels, tight_bind_ham)
    bsites = [int(b) for b in bsites]
    haldane_mat[bsites, :] = np.conj(haldane_mat[bsites, :])

    return (haldane_mat)

def haldane_current_termonly_hyperbolicq3(p, num_levels, tight_bind_ham, t2):
    wholeham = haldane_current_hyperbolicq3(p, num_levels, tight_bind_ham, t2)
    return(wholeham - tight_bind_ham)

def haldane_current_hyperbolicq3_sublatticeblockform(p, num_levels, tight_bind_ham, t2):
    haldane_mat = haldane_current_hyperbolicq3(p, num_levels, tight_bind_ham, t2)
    asites, bsites = site_assignment(p, 3, num_levels, tight_bind_ham)
    asites = [int(a) for a in asites]
    bsites = [int(b) for b in bsites]
    newbasis = np.concatenate((asites, bsites))
    haldanemat_newbasis = haldane_mat[:, newbasis][newbasis, :]
    return(haldanemat_newbasis)

def haldane_current_termonly_hyperbolicq3_sublatticeblockform(p, num_levels, tight_bind_ham, t2):
    wholeham = haldane_current_hyperbolicq3(p, num_levels, tight_bind_ham, t2)
    asites, bsites = site_assignment(p, 3, num_levels, tight_bind_ham)
    asites = [int(a) for a in asites]
    bsites = [int(b) for b in bsites]
    newbasis = np.concatenate((asites, bsites))
    wholeham_newbasis = wholeham[:, newbasis][newbasis, :]
    tight_bind_ham_newbasis = tight_bind_ham[:, newbasis][newbasis, :]
    return (wholeham_newbasis - tight_bind_ham_newbasis)

# def update_haldane_term(haldane_term_mat, t2mat, nnn_hopping_indices):
#     original_mat = np.copy(haldane_term_mat)
#     original_mat[nnn_hopping_indices] = t2mat[nnn_hopping_indices]
#     return(original_mat)

def get_nnn_hopping_indices_hyperbolicq3(p, num_levels):
    tight_bind_ham = H0(p, 3, num_levels)
    haldane_mat = haldane_current_termonly_hyperbolicq3_sublatticeblockform(p, num_levels, tight_bind_ham, 1)
    return(np.where(haldane_mat != 0))

def get_nnn_hopping_indices_hyperbolicq3_ogbasis(p, num_levels):
    tight_bind_ham = H0(p, 3, num_levels)
    haldane_mat = haldane_current_termonly_hyperbolicq3(p, num_levels, tight_bind_ham, 1)
    return(np.where(haldane_mat != 0))

def get_nnn_hopping_indices_signspecific_hyperbolicq3(p, num_levels):
    tight_bind_ham = H0(p, 3, num_levels)
    haldane_mat = haldane_current_termonly_hyperbolicq3_sublatticeblockform(p, num_levels, tight_bind_ham, 1)
    return(np.where(np.imag(haldane_mat) > 0), np.where(np.imag(haldane_mat) < 0))

def self_consistent_check_2d_imag(new_res_mat, old_res_mat, tolerance):
    difference = np.abs(np.imag(new_res_mat - old_res_mat))
    if np.any(difference > tolerance):
        return(False)
    else:
        return(True)

def selfconsist_hartreefock_NonHermitian_DisorderAlreadyAdded_SaveRawOutputs_MoreEfficient(p, num_levels, initial_tight_bind_ham, initial_guess, tolerance, hfcoeff_list):
    nnn_hopping_indices = get_nnn_hopping_indices_hyperbolicq3(p, num_levels)
    nnn_hopping_indices_pos, nnn_hopping_indices_neg = get_nnn_hopping_indices_signspecific_hyperbolicq3(p, num_levels)
    total_num_sites = int(points(p, 3, num_levels)[1])
    final_results_dict = {}

    plaq_boundary_sites = get_plaquet_boundary_sites_hyperbolicq3(p, num_levels)
    plaq_isites, plaq_jsites = get_nnn_hoppings_around_plaquet_sublatticebasis_hyperbolicq3(p, num_levels, plaq_boundary_sites)

    for hfcoeff in hfcoeff_list:  # Go over all desired V2 values
        print(hfcoeff)
        final_results_dict['V2={}'.format(hfcoeff)] = np.zeros((3, len(nnn_hopping_indices[0])), dtype=np.complex_)
        converge_status = False
        old_t2mat = np.zeros((total_num_sites, total_num_sites), dtype=np.complex_)
        old_t2mat[nnn_hopping_indices_pos] = 1j * initial_guess
        old_t2mat[nnn_hopping_indices_neg] = -1j * initial_guess
        ham = initial_tight_bind_ham - (hfcoeff * old_t2mat)
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
            new_t2mat = np.zeros((total_num_sites, total_num_sites), dtype=np.complex_)
            for nhi in range(len(nnn_hopping_indices[0])):
                new_t2mat[(nnn_hopping_indices[1])[nhi], (nnn_hopping_indices[0])[nhi]] = np.sum(left_negative_energy_eigvec_matrix[(nnn_hopping_indices[0])[nhi], :] * right_negative_energy_eigvec_matrix[(nnn_hopping_indices[1])[nhi], :])
            # print('Calculating site probabilities done')
            del left_negative_energy_eigvec_matrix
            del right_negative_energy_eigvec_matrix
            print('Max Real Component: {}'.format(np.max(np.abs(np.real(new_t2mat)))))
            print('Total flux through system: {}'.format(np.sum(np.imag(calculate_haldane_current_around_plaquet_sublatticebasis(new_t2mat, plaq_isites, plaq_jsites)))))
            converge_status = self_consistent_check_2d_imag(new_t2mat, old_t2mat, tolerance)
            # print('Getting convergence status done')
            ham = initial_tight_bind_ham - (hfcoeff * new_t2mat)
            # print('Getting new hamiltonian done')
            converge_progress_count += 1
            print(converge_progress_count, np.max(np.abs(np.imag(new_t2mat - old_t2mat))), 'Time taken: {}'.format(time.time() - start_time))
            old_t2mat = np.copy(new_t2mat)
        print('Convergence done')
        (final_results_dict['V2={}'.format(hfcoeff)])[0,:] = nnn_hopping_indices[0]
        (final_results_dict['V2={}'.format(hfcoeff)])[1,:] = nnn_hopping_indices[1]
        t2values_flattened_ordered = np.array([])
        for ncii in range(len(nnn_hopping_indices[0])):
            t2values_flattened_ordered = np.append(t2values_flattened_ordered, new_t2mat[(nnn_hopping_indices[0])[ncii], (nnn_hopping_indices[1])[ncii]])
        (final_results_dict['V2={}'.format(hfcoeff)])[2, :] = t2values_flattened_ordered
    return(final_results_dict)

########################################################################################################################

from Fundamental.Honeycomb_Lattice import honeycomb_points
from Fundamental.Honeycomb_Lattice import honeycomb_lattice
from Fundamental.NonHermitian_Hamiltonian import site_assignment_honeycomb
from Axial_Magnetic_Field.honeycomb import connected_to_next_gen_points, points_on_level
from Axial_Magnetic_Field.honeycomb import connected_to_prev_gen_points
from Fundamental.Local import get_plaquet_boundary_sites_honeycomb, get_nnn_hoppings_around_plaquet_sublatticebasis_honeycomb
from Fundamental.Local import calculate_haldane_current_around_plaquet_sublatticebasis

def haldane_current_honeycomb(num_levels, tight_bind_ham, t2):
    haldane_mat = np.copy(tight_bind_ham).astype(np.complex_)

    points_per_level = honeycomb_points(num_levels)[0]

    # points_conn_next_level = sites on current gen that connect to above gen sites
    # points_that_conn_from_above = sites on next gen that connect down to current gen

    # Hard code first gen since get_if_point_is_connected_with_upper_layer_q3_general does not work for first gen
    sites_on_level = np.arange(0, 6, dtype=int)
    sites_on_next_level = np.arange(6, np.sum(points_per_level[:2]), dtype=int)
    points_conn_next_level = connected_to_next_gen_points(1)
    points_that_conn_from_above = connected_to_prev_gen_points(2)
    for i in range(6):  # intragen nnn hopping
        haldane_mat[i, np.take(sites_on_level, i + 2, mode='wrap')] = 1j * t2
        haldane_mat[np.take(sites_on_level, i + 2, mode='wrap'), i] = np.conj(1j * t2)
    for i in range(len(points_conn_next_level)):  # intergen nnn hopping
        next_level_site_connected_index = np.where(sites_on_next_level == points_that_conn_from_above[i])[0]
        nnn_sites_plusone = np.take(sites_on_next_level, next_level_site_connected_index + 1, mode='wrap')
        nnn_sites_minusone = np.take(sites_on_next_level, next_level_site_connected_index - 1, mode='wrap')
        haldane_mat[points_conn_next_level[i], nnn_sites_plusone] = 1j * t2
        haldane_mat[nnn_sites_plusone, points_conn_next_level[i]] = np.conj(1j * t2)
        haldane_mat[points_conn_next_level[i], nnn_sites_minusone] = np.conj(1j * t2)
        haldane_mat[nnn_sites_minusone, points_conn_next_level[i]] = 1j * t2
        # Finally need to take into account nnn hopping for adjacent points_that_conn_from_above
        nnn_sites_plusplusone = np.take(points_that_conn_from_above, i + 1, mode='wrap')
        nnn_sites_minusminusone = np.take(points_that_conn_from_above, i - 1, mode='wrap')
        haldane_mat[points_conn_next_level[i], nnn_sites_plusplusone] = np.conj(1j * t2)
        haldane_mat[nnn_sites_plusplusone, points_conn_next_level[i]] = 1j * t2
        haldane_mat[points_conn_next_level[i], nnn_sites_minusminusone] = 1j * t2
        haldane_mat[nnn_sites_minusminusone, points_conn_next_level[i]] = np.conj(1j * t2)

    for nl in range(1, num_levels - 1):  # Handles all other generations except the last gen
        sites_on_level = np.arange(np.sum(points_per_level[:nl]), np.sum(points_per_level[:nl + 1]), dtype=int)
        sites_on_next_level = np.arange(np.sum(points_per_level[:nl + 1]), np.sum(points_per_level[:nl + 2]), dtype=int)
        points_conn_next_level = connected_to_next_gen_points(nl + 1)
        points_that_conn_from_above = connected_to_prev_gen_points(nl + 2)
        for i in range(len(sites_on_level)):  # intragen nnn hopping
            if np.take(sites_on_level, i + 1, mode='wrap') in points_conn_next_level: #Need this due to honeycomb geometry
                haldane_mat[sites_on_level[i], np.take(sites_on_level, i + 2, mode='wrap')] = 1j * t2
                haldane_mat[np.take(sites_on_level, i + 2, mode='wrap'), sites_on_level[i]] = np.conj(1j * t2)
            else:
                haldane_mat[sites_on_level[i], np.take(sites_on_level, i + 2, mode='wrap')] = np.conj(1j * t2)
                haldane_mat[np.take(sites_on_level, i + 2, mode='wrap'), sites_on_level[i]] = 1j * t2

        # intergen nnn hopping
        for i in range(len(points_conn_next_level)):
            next_level_site_connected_index = np.where(sites_on_next_level == points_that_conn_from_above[i])[0]
            nnn_sites_plusone = np.take(sites_on_next_level, next_level_site_connected_index + 1, mode='wrap')
            nnn_sites_minusone = np.take(sites_on_next_level, next_level_site_connected_index - 1, mode='wrap')
            haldane_mat[points_conn_next_level[i], nnn_sites_plusone] = 1j * t2
            haldane_mat[nnn_sites_plusone, points_conn_next_level[i]] = np.conj(1j * t2)
            haldane_mat[points_conn_next_level[i], nnn_sites_minusone] = np.conj(1j * t2)
            haldane_mat[nnn_sites_minusone, points_conn_next_level[i]] = 1j * t2
            # Modified code to account nnn hopping for adjacent points_that_conn_from_above for honeycomb geometry
            if np.take(points_conn_next_level, i+1, mode='wrap') - points_conn_next_level[i] == 1:
                nnn_sites_plusplusone = np.take(points_that_conn_from_above, i + 1, mode='wrap')
                haldane_mat[points_conn_next_level[i], nnn_sites_plusplusone] = np.conj(1j * t2)
                haldane_mat[nnn_sites_plusplusone, points_conn_next_level[i]] = 1j * t2
            if points_conn_next_level[i] - points_conn_next_level[i-1] == 1:
                nnn_sites_minusminusone = np.take(points_that_conn_from_above, i - 1, mode='wrap')
                haldane_mat[points_conn_next_level[i], nnn_sites_minusminusone] = 1j * t2
                haldane_mat[nnn_sites_minusminusone, points_conn_next_level[i]] = np.conj(1j * t2)
        # Finally need to account for sites on gen that do not connect via NN hopping to next gen
        points_not_conn_next_level = np.setdiff1d(sites_on_level, points_conn_next_level)
        for pncnl in points_not_conn_next_level:
            pncnl_index = np.where(sites_on_level == pncnl)[0]
            siteoneabove = np.take(sites_on_level, pncnl_index + 1, mode='wrap')
            siteonebelow = np.take(sites_on_level, pncnl_index - 1, mode='wrap')
            nnnnextgenoneabovesite = points_that_conn_from_above[np.where(points_conn_next_level == siteoneabove)[0]]
            nnnnextgenonebelowsite = points_that_conn_from_above[np.where(points_conn_next_level == siteonebelow)[0]]
            haldane_mat[pncnl, nnnnextgenoneabovesite] = np.conj(1j * t2)
            haldane_mat[nnnnextgenoneabovesite, pncnl] = 1j * t2
            haldane_mat[pncnl, nnnnextgenonebelowsite] = 1j * t2
            haldane_mat[nnnnextgenonebelowsite, pncnl] = np.conj(1j * t2)

    # Finally handle last gen
    points_conn_next_level = connected_to_next_gen_points(num_levels)
    sites_on_level = np.arange(np.sum(points_per_level[:num_levels - 1]), np.sum(points_per_level), dtype=int)
    for i in range(len(sites_on_level)):  # intragen nnn hopping (all intergen hopping already taken care of)
        if (sites_on_level[i] in points_conn_next_level) and (np.take(sites_on_level, i+1, mode='wrap') not in points_conn_next_level):
            haldane_mat[sites_on_level[i], np.take(sites_on_level, i + 2, mode='wrap')] = np.conj(1j * t2)
            haldane_mat[np.take(sites_on_level, i + 2, mode='wrap'), sites_on_level[i]] = 1j * t2
        else:
            haldane_mat[sites_on_level[i], np.take(sites_on_level, i + 2, mode='wrap')] = 1j * t2
            haldane_mat[np.take(sites_on_level, i + 2, mode='wrap'), sites_on_level[i]] = np.conj(1j * t2)

    #Last bit of code to reverse sign of phase for NNN hopping on B sublattice
    asites, bsites = site_assignment_honeycomb(num_levels, honeycomb_lattice(num_levels))
    bsites = [int(b) for b in bsites]
    haldane_mat[bsites, :] = np.conj(haldane_mat[bsites, :])

    return (haldane_mat)

def haldane_current_termonly_honeycomb(num_levels, tight_bind_ham, t2):
    wholeham = haldane_current_honeycomb(num_levels, tight_bind_ham, t2)
    return(wholeham - tight_bind_ham)

def haldane_current_termonly_honeycomb_sublatticeblockform(num_levels, tight_bind_ham, t2):
    asites, bsites = site_assignment_honeycomb(num_levels, tight_bind_ham)
    asites = [int(a) for a in asites]
    bsites = [int(b) for b in bsites]
    newbasis = np.concatenate((asites, bsites))
    haldane_mat_ogbasis = haldane_current_termonly_honeycomb(num_levels, tight_bind_ham, t2)
    haldane_mat_newbasis = haldane_mat_ogbasis[:,newbasis][newbasis,:]
    return(haldane_mat_newbasis)

def get_nnn_hopping_indices_honeycomb(num_levels):
    tight_bind_ham = honeycomb_lattice(num_levels)
    haldane_mat = haldane_current_termonly_honeycomb_sublatticeblockform(num_levels, tight_bind_ham, 1)
    return(np.where(haldane_mat != 0))

def get_nnn_hopping_indices_originalbasis_honeycomb(num_levels):
    tight_bind_ham = honeycomb_lattice(num_levels)
    haldane_mat = haldane_current_termonly_honeycomb(num_levels, tight_bind_ham, 1)
    return(np.where(haldane_mat != 0))

def get_nnn_hopping_indices_signspecific_honeycomb(num_levels):
    tight_bind_ham = honeycomb_lattice(num_levels)
    haldane_mat = haldane_current_termonly_honeycomb_sublatticeblockform(num_levels, tight_bind_ham, 1)
    return(np.where(np.imag(haldane_mat) > 0), np.where(np.imag(haldane_mat) < 0))

def selfconsist_hartreefock_NonHermitian_DisorderAlreadyAdded_SaveRawOutputs_MoreEfficient_honeycomb(num_levels, initial_tight_bind_ham, initial_guess, tolerance, hfcoeff_list):
    nnn_hopping_indices = get_nnn_hopping_indices_honeycomb(num_levels)
    nnn_hopping_indices_pos, nnn_hopping_indices_neg = get_nnn_hopping_indices_signspecific_honeycomb(num_levels)
    total_num_sites = int(honeycomb_points(num_levels)[1])
    final_results_dict = {}
    plaq_bound_sites = get_plaquet_boundary_sites_honeycomb(num_levels)
    plaq_nnn_hopping_isites, plaq_nnn_hopping_jsites = get_nnn_hoppings_around_plaquet_sublatticebasis_honeycomb(num_levels, plaq_bound_sites)

    for hfcoeff in hfcoeff_list:  # Go over all desired V2 values
        print(hfcoeff)
        final_results_dict['V2={}'.format(hfcoeff)] = np.zeros((3, len(nnn_hopping_indices[0])), dtype=np.complex_)
        converge_status = False
        old_t2mat = np.zeros((total_num_sites, total_num_sites), dtype=np.complex_)

        old_t2mat[nnn_hopping_indices_pos] = 1j * initial_guess
        old_t2mat[nnn_hopping_indices_neg] = -1j * initial_guess
        ham = initial_tight_bind_ham - (hfcoeff * old_t2mat)
        converge_progress_count = 0

        print('sum of phase over all plaquet', np.sum(np.imag(calculate_haldane_current_around_plaquet_sublatticebasis(old_t2mat, plaq_nnn_hopping_isites, plaq_nnn_hopping_jsites))))
        print('Max real component: {}'.format(np.max(np.abs(np.real(old_t2mat)))))
        while converge_status == False:
            start_time = time.time()
            # print('Diagonalization Start')
            eners, left_eigvec_matrix, right_eigvec_matrix = scipy.linalg.eig(ham, left=True, right=True)  # diagonalize ham to get energies and eigvecs
            # eners, right_eigvec_matrix = scipy.linalg.eigh(ham)
            del ham
            # print('Diagonalization Done')
            left_eigvec_matrix = np.conj(left_eigvec_matrix)
            # left_eigvec_matrix = np.conj(right_eigvec_matrix)
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
            new_t2mat = np.zeros((total_num_sites, total_num_sites), dtype=np.complex_)
            for nhi in range(len(nnn_hopping_indices[0])):
                new_t2mat[(nnn_hopping_indices[1])[nhi], (nnn_hopping_indices[0])[nhi]] = np.sum(left_negative_energy_eigvec_matrix[(nnn_hopping_indices[0])[nhi], :] * right_negative_energy_eigvec_matrix[(nnn_hopping_indices[1])[nhi], :])
            # print('t2mat is Hermitian: {}'.format(np.all(new_t2mat == np.conj(np.transpose(new_t2mat)))))
            del left_negative_energy_eigvec_matrix
            del right_negative_energy_eigvec_matrix

            print('sum of phase over all plaquets: {}'.format(np.sum(np.imag(calculate_haldane_current_around_plaquet_sublatticebasis(new_t2mat, plaq_nnn_hopping_isites, plaq_nnn_hopping_jsites)))))
            print('Max real component: {}'.format(np.max(np.abs(np.real(new_t2mat)))))
            converge_status = self_consistent_check_2d_imag(new_t2mat, old_t2mat, tolerance)
            converge_progress_count += 1
            print(converge_progress_count, np.max(np.abs(np.imag(new_t2mat - old_t2mat))), 'Time taken: {}'.format(time.time() - start_time))

            ham = initial_tight_bind_ham - (hfcoeff * new_t2mat)

            old_t2mat = np.copy(new_t2mat)


        print('Convergence done')
        (final_results_dict['V2={}'.format(hfcoeff)])[0,:] = nnn_hopping_indices[0]
        (final_results_dict['V2={}'.format(hfcoeff)])[1,:] = nnn_hopping_indices[1]
        t2values_flattened_ordered = np.array([])
        for ncii in range(len(nnn_hopping_indices[0])):
            t2values_flattened_ordered = np.append(t2values_flattened_ordered, new_t2mat[(nnn_hopping_indices[0])[ncii], (nnn_hopping_indices[1])[ncii]])
        (final_results_dict['V2={}'.format(hfcoeff)])[2, :] = t2values_flattened_ordered

    return(final_results_dict)

#######################################################################################################################

from Axial_Magnetic_Field.symmetry import return_c3symmetry_site_label_transform_basis_honeycomb
from Fundamental.NonHermitian_Hamiltonian import site_assignment_honeycomb
from Fundamental.Honeycomb_Lattice import honeycomb_lattice

def selfconsist_hartreefock_Hermitian_SublatticeBasis_DisorderAlreadyAdded_SaveRawOutputs_MoreEfficient_honeycomb_C3Average(num_levels, initial_tight_bind_ham, initial_guess, tolerance, hfcoeff_list):
    total_magnetic_flux_values = np.array([])
    max_real_components = np.array([])
    nnn_hopping_indices = get_nnn_hopping_indices_honeycomb(num_levels)
    nnn_hopping_indices_pos, nnn_hopping_indices_neg = get_nnn_hopping_indices_signspecific_honeycomb(num_levels)
    total_num_sites = int(honeycomb_points(num_levels)[1])
    final_results_dict = {}
    plaq_bound_sites = get_plaquet_boundary_sites_honeycomb(num_levels)
    plaq_nnn_hopping_isites, plaq_nnn_hopping_jsites = get_nnn_hoppings_around_plaquet_sublatticebasis_honeycomb(num_levels, plaq_bound_sites)

    rotated_basis_one, rotated_basis_two = return_c3symmetry_site_label_transform_basis_honeycomb(num_levels)
    c3basismat = np.vstack((rotated_basis_one, rotated_basis_two))
    del rotated_basis_one
    del rotated_basis_two

    asites, bsites = site_assignment_honeycomb(num_levels, honeycomb_lattice(num_levels))
    sublatticebasis = np.concatenate((asites, bsites))
    sublatticebasis = np.array([int(sbl) for sbl in sublatticebasis])
    reversebasis = np.array([], dtype=int)
    for i in np.arange(np.size(initial_tight_bind_ham, 0)):
        reversebasis = np.append(reversebasis, np.where(i == sublatticebasis)[0])

    nnn_hopping_indices_ogbasis = get_nnn_hopping_indices_originalbasis_honeycomb(num_levels)

    for hfcoeff in hfcoeff_list:  # Go over all desired V2 values
        print(hfcoeff)
        final_results_dict['V2={}'.format(hfcoeff)] = np.zeros((3, len(nnn_hopping_indices[0])), dtype=np.complex_)
        converge_status = False
        old_t2mat = np.zeros((total_num_sites, total_num_sites), dtype=np.complex_)

        old_t2mat[nnn_hopping_indices_pos] = 1j * initial_guess
        old_t2mat[nnn_hopping_indices_neg] = -1j * initial_guess

        ham = initial_tight_bind_ham - (hfcoeff * old_t2mat)

        converge_progress_count = 0

        print('sum of phase over all plaquet', np.sum(np.imag(calculate_haldane_current_around_plaquet_sublatticebasis(old_t2mat, plaq_nnn_hopping_isites, plaq_nnn_hopping_jsites))))
        print('Max real component: {}'.format(np.max(np.abs(np.real(old_t2mat)))))
        while converge_status == False:
            start_time = time.time()
            # print('Diagonalization Start')
            eners, right_eigvec_matrix = scipy.linalg.eigh(ham)
            del ham
            # print('Diagonalization Done')
            left_eigvec_matrix = np.conj(right_eigvec_matrix)
            negative_energy_indices = np.where(eners < 0)  # Find indices where E<0
            left_negative_energy_eigvec_matrix = left_eigvec_matrix[:, negative_energy_indices[0]]
            right_negative_energy_eigvec_matrix = right_eigvec_matrix[:, negative_energy_indices[0]]
            del eners
            del left_eigvec_matrix
            del right_eigvec_matrix
            del negative_energy_indices
            # print('Getting filled states done')
            new_t2mat = np.zeros((total_num_sites, total_num_sites), dtype=np.complex_)
            for nhi in range(len(nnn_hopping_indices[0])):
                new_t2mat[(nnn_hopping_indices[0])[nhi], (nnn_hopping_indices[1])[nhi]] = np.sum(left_negative_energy_eigvec_matrix[(nnn_hopping_indices[1])[nhi], :] * right_negative_energy_eigvec_matrix[(nnn_hopping_indices[0])[nhi], :])
            left_negative_energy_eigvec_matrix_ogbasis = left_negative_energy_eigvec_matrix[reversebasis, :]
            right_negative_energy_eigvec_matrix_ogbasis = right_negative_energy_eigvec_matrix[reversebasis, :]
            for rc in range(np.size(c3basismat, 0)):
                rotated_term_to_add = np.zeros((total_num_sites, total_num_sites), dtype=np.complex_)
                left_negative_energy_eigvec_matrix_ogbasis_rotated = np.copy(left_negative_energy_eigvec_matrix_ogbasis)[c3basismat[rc, :], :]
                right_negative_energy_eigvec_matrix_ogbasis_rotated = np.copy(right_negative_energy_eigvec_matrix_ogbasis)[c3basismat[rc, :], :]
                for nhi in range(len(nnn_hopping_indices_ogbasis[0])):
                    rotated_term_to_add[(nnn_hopping_indices_ogbasis[1])[nhi], (nnn_hopping_indices_ogbasis[0])[nhi]] = np.sum(left_negative_energy_eigvec_matrix_ogbasis_rotated[(nnn_hopping_indices_ogbasis[0])[nhi], :] * right_negative_energy_eigvec_matrix_ogbasis_rotated[(nnn_hopping_indices_ogbasis[1])[nhi], :])
                new_t2mat = new_t2mat + rotated_term_to_add[:, sublatticebasis][sublatticebasis, :]
            new_t2mat = new_t2mat / (np.size(c3basismat, 0) + 1)

            del left_negative_energy_eigvec_matrix
            del right_negative_energy_eigvec_matrix
            del left_negative_energy_eigvec_matrix_ogbasis
            del right_negative_energy_eigvec_matrix_ogbasis
            del left_negative_energy_eigvec_matrix_ogbasis_rotated
            del right_negative_energy_eigvec_matrix_ogbasis_rotated
            del rotated_term_to_add

            print('sum of phase over all plaquets: {}'.format(np.sum(np.imag(calculate_haldane_current_around_plaquet_sublatticebasis(new_t2mat, plaq_nnn_hopping_isites, plaq_nnn_hopping_jsites)))))
            print('Max real component: {}'.format(np.max(np.abs(np.real(new_t2mat)))))
            total_magnetic_flux_values = np.append(total_magnetic_flux_values,
                                                   np.sum(np.imag(calculate_haldane_current_around_plaquet_sublatticebasis(new_t2mat, plaq_nnn_hopping_isites, plaq_nnn_hopping_jsites))))
            max_real_components = np.append(max_real_components, np.max(np.abs(np.real(new_t2mat))))
            converge_status = self_consistent_check_2d_imag(new_t2mat, old_t2mat, tolerance)
            converge_progress_count += 1
            print(converge_progress_count, np.max(np.abs(np.imag(new_t2mat - old_t2mat))), 'Time taken: {}'.format(time.time() - start_time))

            ham = initial_tight_bind_ham - (hfcoeff * new_t2mat)

            old_t2mat = np.copy(new_t2mat)

        print('Convergence done')
        (final_results_dict['V2={}'.format(hfcoeff)])[0,:] = nnn_hopping_indices[0]
        (final_results_dict['V2={}'.format(hfcoeff)])[1,:] = nnn_hopping_indices[1]
        t2values_flattened_ordered = np.array([])
        for ncii in range(len(nnn_hopping_indices[0])):
            t2values_flattened_ordered = np.append(t2values_flattened_ordered, new_t2mat[(nnn_hopping_indices[0])[ncii], (nnn_hopping_indices[1])[ncii]])
        (final_results_dict['V2={}'.format(hfcoeff)])[2, :] = t2values_flattened_ordered

    return(final_results_dict, total_magnetic_flux_values, max_real_components)

def selfconsist_hartreefock_Hermitian_SublatticeBasis_DisorderAlreadyAdded_SaveRawOutputs_MoreEfficient_honeycomb_C3Average_DiscardReal(num_levels, initial_tight_bind_ham, initial_guess, tolerance, hfcoeff_list):
    total_magnetic_flux_values = np.array([])
    max_real_components = np.array([])
    nnn_hopping_indices = get_nnn_hopping_indices_honeycomb(num_levels)
    nnn_hopping_indices_pos, nnn_hopping_indices_neg = get_nnn_hopping_indices_signspecific_honeycomb(num_levels)
    total_num_sites = int(honeycomb_points(num_levels)[1])
    final_results_dict = {}
    plaq_bound_sites = get_plaquet_boundary_sites_honeycomb(num_levels)
    plaq_nnn_hopping_isites, plaq_nnn_hopping_jsites = get_nnn_hoppings_around_plaquet_sublatticebasis_honeycomb(num_levels, plaq_bound_sites)

    rotated_basis_one, rotated_basis_two = return_c3symmetry_site_label_transform_basis_honeycomb(num_levels)
    c3basismat = np.vstack((rotated_basis_one, rotated_basis_two))
    del rotated_basis_one
    del rotated_basis_two

    asites, bsites = site_assignment_honeycomb(num_levels, honeycomb_lattice(num_levels))
    sublatticebasis = np.concatenate((asites, bsites))
    sublatticebasis = np.array([int(sbl) for sbl in sublatticebasis])
    reversebasis = np.array([], dtype=int)
    for i in np.arange(np.size(initial_tight_bind_ham, 0)):
        reversebasis = np.append(reversebasis, np.where(i == sublatticebasis)[0])

    nnn_hopping_indices_ogbasis = get_nnn_hopping_indices_originalbasis_honeycomb(num_levels)

    for hfcoeff in hfcoeff_list:  # Go over all desired V2 values
        print(hfcoeff)
        final_results_dict['V2={}'.format(hfcoeff)] = np.zeros((3, len(nnn_hopping_indices[0])), dtype=np.complex_)
        converge_status = False
        old_t2mat = np.zeros((total_num_sites, total_num_sites), dtype=np.complex_)

        old_t2mat[nnn_hopping_indices_pos] = 1j * initial_guess
        old_t2mat[nnn_hopping_indices_neg] = -1j * initial_guess

        ham = initial_tight_bind_ham - (hfcoeff * old_t2mat)

        converge_progress_count = 0

        print('sum of phase over all plaquet', np.sum(np.imag(calculate_haldane_current_around_plaquet_sublatticebasis(old_t2mat, plaq_nnn_hopping_isites, plaq_nnn_hopping_jsites))))
        print('Max real component: {}'.format(np.max(np.abs(np.real(old_t2mat)))))
        while converge_status == False:
            start_time = time.time()
            # print('Diagonalization Start')
            eners, right_eigvec_matrix = scipy.linalg.eigh(ham)
            del ham
            # print('Diagonalization Done')
            left_eigvec_matrix = np.conj(right_eigvec_matrix)
            negative_energy_indices = np.where(eners < 0)  # Find indices where E<0
            left_negative_energy_eigvec_matrix = left_eigvec_matrix[:, negative_energy_indices[0]]
            right_negative_energy_eigvec_matrix = right_eigvec_matrix[:, negative_energy_indices[0]]
            del eners
            del left_eigvec_matrix
            del right_eigvec_matrix
            del negative_energy_indices
            # print('Getting filled states done')
            new_t2mat = np.zeros((total_num_sites, total_num_sites), dtype=np.complex_)
            for nhi in range(len(nnn_hopping_indices[0])):
                new_t2mat[(nnn_hopping_indices[0])[nhi], (nnn_hopping_indices[1])[nhi]] = np.sum(left_negative_energy_eigvec_matrix[(nnn_hopping_indices[1])[nhi], :] * right_negative_energy_eigvec_matrix[(nnn_hopping_indices[0])[nhi], :])
            left_negative_energy_eigvec_matrix_ogbasis = left_negative_energy_eigvec_matrix[reversebasis, :]
            right_negative_energy_eigvec_matrix_ogbasis = right_negative_energy_eigvec_matrix[reversebasis, :]
            for rc in range(np.size(c3basismat, 0)):
                rotated_term_to_add = np.zeros((total_num_sites, total_num_sites), dtype=np.complex_)
                left_negative_energy_eigvec_matrix_ogbasis_rotated = np.copy(left_negative_energy_eigvec_matrix_ogbasis)[c3basismat[rc, :], :]
                right_negative_energy_eigvec_matrix_ogbasis_rotated = np.copy(right_negative_energy_eigvec_matrix_ogbasis)[c3basismat[rc, :], :]
                for nhi in range(len(nnn_hopping_indices_ogbasis[0])):
                    rotated_term_to_add[(nnn_hopping_indices_ogbasis[1])[nhi], (nnn_hopping_indices_ogbasis[0])[nhi]] = np.sum(left_negative_energy_eigvec_matrix_ogbasis_rotated[(nnn_hopping_indices_ogbasis[0])[nhi], :] * right_negative_energy_eigvec_matrix_ogbasis_rotated[(nnn_hopping_indices_ogbasis[1])[nhi], :])
                new_t2mat = new_t2mat + rotated_term_to_add[:, sublatticebasis][sublatticebasis, :]
            new_t2mat = new_t2mat / (np.size(c3basismat, 0) + 1)

            del left_negative_energy_eigvec_matrix
            del right_negative_energy_eigvec_matrix
            del left_negative_energy_eigvec_matrix_ogbasis
            del right_negative_energy_eigvec_matrix_ogbasis
            del left_negative_energy_eigvec_matrix_ogbasis_rotated
            del right_negative_energy_eigvec_matrix_ogbasis_rotated
            del rotated_term_to_add

            print('sum of phase over all plaquets: {}'.format(np.sum(np.imag(calculate_haldane_current_around_plaquet_sublatticebasis(new_t2mat, plaq_nnn_hopping_isites, plaq_nnn_hopping_jsites)))))
            print('Max real component: {}'.format(np.max(np.abs(np.real(new_t2mat)))))
            total_magnetic_flux_values = np.append(total_magnetic_flux_values,
                                                   np.sum(np.imag(calculate_haldane_current_around_plaquet_sublatticebasis(new_t2mat, plaq_nnn_hopping_isites, plaq_nnn_hopping_jsites))))
            max_real_components = np.append(max_real_components, np.max(np.abs(np.real(new_t2mat))))
            converge_status = self_consistent_check_2d_imag(new_t2mat, old_t2mat, tolerance)
            converge_progress_count += 1
            print(converge_progress_count, np.max(np.abs(np.imag(new_t2mat - old_t2mat))), 'Time taken: {}'.format(time.time() - start_time))

            ham = initial_tight_bind_ham - (hfcoeff * np.imag(new_t2mat))*1j ###Only difference for "DiscardReal"

            old_t2mat = np.copy(new_t2mat)

        print('Convergence done')
        (final_results_dict['V2={}'.format(hfcoeff)])[0,:] = nnn_hopping_indices[0]
        (final_results_dict['V2={}'.format(hfcoeff)])[1,:] = nnn_hopping_indices[1]
        t2values_flattened_ordered = np.array([])
        for ncii in range(len(nnn_hopping_indices[0])):
            t2values_flattened_ordered = np.append(t2values_flattened_ordered, new_t2mat[(nnn_hopping_indices[0])[ncii], (nnn_hopping_indices[1])[ncii]])
        (final_results_dict['V2={}'.format(hfcoeff)])[2, :] = t2values_flattened_ordered

    return(final_results_dict, total_magnetic_flux_values, max_real_components)

#Copy of above code but with a few changes to make it work with NH
def selfconsist_hartreefock_NonHermitian_SublatticeBasis_DisorderAlreadyAdded_SaveRawOutputs_MoreEfficient_honeycomb_C3Average_DiscardReal(num_levels, initial_tight_bind_ham, initial_guess, tolerance, hfcoeff_list):
    total_magnetic_flux_values = np.array([])
    max_real_components = np.array([])

    #Two extra trackers (that I vstack later) to track the imaginary eigenvalues since adding NH may cause issues
    max_imag_eigvals = np.array([])
    min_imag_eigvals = np.array([])
    #Also another tracker to track smallest magnitude real component of eigenvalues
    smallest_real_eigvals = np.array([])

    nnn_hopping_indices = get_nnn_hopping_indices_honeycomb(num_levels)
    nnn_hopping_indices_pos, nnn_hopping_indices_neg = get_nnn_hopping_indices_signspecific_honeycomb(num_levels)
    total_num_sites = int(honeycomb_points(num_levels)[1])
    final_results_dict = {}
    plaq_bound_sites = get_plaquet_boundary_sites_honeycomb(num_levels)
    plaq_nnn_hopping_isites, plaq_nnn_hopping_jsites = get_nnn_hoppings_around_plaquet_sublatticebasis_honeycomb(num_levels, plaq_bound_sites)

    rotated_basis_one, rotated_basis_two = return_c3symmetry_site_label_transform_basis_honeycomb(num_levels)
    c3basismat = np.vstack((rotated_basis_one, rotated_basis_two))
    del rotated_basis_one
    del rotated_basis_two

    asites, bsites = site_assignment_honeycomb(num_levels, honeycomb_lattice(num_levels))
    sublatticebasis = np.concatenate((asites, bsites))
    sublatticebasis = np.array([int(sbl) for sbl in sublatticebasis])
    reversebasis = np.array([], dtype=int)
    for i in np.arange(np.size(initial_tight_bind_ham, 0)):
        reversebasis = np.append(reversebasis, np.where(i == sublatticebasis)[0])

    nnn_hopping_indices_ogbasis = get_nnn_hopping_indices_originalbasis_honeycomb(num_levels)

    for hfcoeff in hfcoeff_list:  # Go over all desired V2 values
        print(hfcoeff)
        final_results_dict['V2={}'.format(hfcoeff)] = np.zeros((3, len(nnn_hopping_indices[0])), dtype=np.complex_)
        converge_status = False
        old_t2mat = np.zeros((total_num_sites, total_num_sites), dtype=np.complex_)

        old_t2mat[nnn_hopping_indices_pos] = 1j * initial_guess
        old_t2mat[nnn_hopping_indices_neg] = -1j * initial_guess

        ham = initial_tight_bind_ham - (hfcoeff * old_t2mat)

        converge_progress_count = 0

        print('sum of phase over all plaquet', np.sum(np.imag(calculate_haldane_current_around_plaquet_sublatticebasis(old_t2mat, plaq_nnn_hopping_isites, plaq_nnn_hopping_jsites))))
        print('Max real component: {}'.format(np.max(np.abs(np.real(old_t2mat)))))
        while converge_status == False:
            start_time = time.time()
            # print('Diagonalization Start')

            #Changed so using eig instead of eigh #1/2 changes for NH
            eners, left_eigvec_matrix, right_eigvec_matrix = scipy.linalg.eig(ham, left=True, right=True)

            del ham
            # print('Diagonalization Done')

            #These lines are changed so left eigenvectors are defined properly and are biorthogonally normalized
            #2/2 changes for NH
            left_eigvec_matrix = np.conj(left_eigvec_matrix)
            left_eigvec_matrix, right_eigvec_matrix = biortogonal_normalize(left_eigvec_matrix, right_eigvec_matrix)

            #Appending the extra imag part trackers that I now have for NH SCHF calculation loop
            max_imag_eigvals = np.append(max_imag_eigvals, np.max(np.imag(eners)))
            min_imag_eigvals = np.append(min_imag_eigvals, np.min(np.imag(eners)))
            #Also append smallest real component of eigval tracker
            smallest_real_eigvals = np.append(smallest_real_eigvals, np.min(np.abs(np.real(eners))))

            negative_energy_indices = np.where(np.real(eners) < 0)  # Find indices where E<0; added np.real to ensure behaves how I expect
            left_negative_energy_eigvec_matrix = left_eigvec_matrix[:, negative_energy_indices[0]]
            right_negative_energy_eigvec_matrix = right_eigvec_matrix[:, negative_energy_indices[0]]
            del eners
            del left_eigvec_matrix
            del right_eigvec_matrix
            del negative_energy_indices
            # print('Getting filled states done')
            new_t2mat = np.zeros((total_num_sites, total_num_sites), dtype=np.complex_)
            for nhi in range(len(nnn_hopping_indices[0])):
                new_t2mat[(nnn_hopping_indices[0])[nhi], (nnn_hopping_indices[1])[nhi]] = np.sum(left_negative_energy_eigvec_matrix[(nnn_hopping_indices[1])[nhi], :] * right_negative_energy_eigvec_matrix[(nnn_hopping_indices[0])[nhi], :])
            left_negative_energy_eigvec_matrix_ogbasis = left_negative_energy_eigvec_matrix[reversebasis, :]
            right_negative_energy_eigvec_matrix_ogbasis = right_negative_energy_eigvec_matrix[reversebasis, :]
            for rc in range(np.size(c3basismat, 0)):
                rotated_term_to_add = np.zeros((total_num_sites, total_num_sites), dtype=np.complex_)
                left_negative_energy_eigvec_matrix_ogbasis_rotated = np.copy(left_negative_energy_eigvec_matrix_ogbasis)[c3basismat[rc, :], :]
                right_negative_energy_eigvec_matrix_ogbasis_rotated = np.copy(right_negative_energy_eigvec_matrix_ogbasis)[c3basismat[rc, :], :]
                for nhi in range(len(nnn_hopping_indices_ogbasis[0])):
                    rotated_term_to_add[(nnn_hopping_indices_ogbasis[1])[nhi], (nnn_hopping_indices_ogbasis[0])[nhi]] = np.sum(left_negative_energy_eigvec_matrix_ogbasis_rotated[(nnn_hopping_indices_ogbasis[0])[nhi], :] * right_negative_energy_eigvec_matrix_ogbasis_rotated[(nnn_hopping_indices_ogbasis[1])[nhi], :])
                new_t2mat = new_t2mat + rotated_term_to_add[:, sublatticebasis][sublatticebasis, :]
            new_t2mat = new_t2mat / (np.size(c3basismat, 0) + 1)

            del left_negative_energy_eigvec_matrix
            del right_negative_energy_eigvec_matrix
            del left_negative_energy_eigvec_matrix_ogbasis
            del right_negative_energy_eigvec_matrix_ogbasis
            del left_negative_energy_eigvec_matrix_ogbasis_rotated
            del right_negative_energy_eigvec_matrix_ogbasis_rotated
            del rotated_term_to_add

            print('sum of phase over all plaquets: {}'.format(np.sum(np.imag(calculate_haldane_current_around_plaquet_sublatticebasis(new_t2mat, plaq_nnn_hopping_isites, plaq_nnn_hopping_jsites)))))
            print('Max real component: {}'.format(np.max(np.abs(np.real(new_t2mat)))))
            total_magnetic_flux_values = np.append(total_magnetic_flux_values,
                                                   np.sum(np.imag(calculate_haldane_current_around_plaquet_sublatticebasis(new_t2mat, plaq_nnn_hopping_isites, plaq_nnn_hopping_jsites))))
            max_real_components = np.append(max_real_components, np.max(np.abs(np.real(new_t2mat))))
            converge_status = self_consistent_check_2d_imag(new_t2mat, old_t2mat, tolerance)
            converge_progress_count += 1
            print(converge_progress_count, np.max(np.abs(np.imag(new_t2mat - old_t2mat))), 'Time taken: {}'.format(time.time() - start_time))

            ham = initial_tight_bind_ham - (hfcoeff * np.imag(new_t2mat))*1j ###Only difference for "DiscardReal"

            old_t2mat = np.copy(new_t2mat)

        print('Convergence done')
        (final_results_dict['V2={}'.format(hfcoeff)])[0,:] = nnn_hopping_indices[0]
        (final_results_dict['V2={}'.format(hfcoeff)])[1,:] = nnn_hopping_indices[1]
        t2values_flattened_ordered = np.array([])
        for ncii in range(len(nnn_hopping_indices[0])):
            t2values_flattened_ordered = np.append(t2values_flattened_ordered, new_t2mat[(nnn_hopping_indices[0])[ncii], (nnn_hopping_indices[1])[ncii]])
        (final_results_dict['V2={}'.format(hfcoeff)])[2, :] = t2values_flattened_ordered

    #Extra bit of code to stack imaginary eigenvalue trackers
    imag_eigvals_bounds_tracker = np.vstack((min_imag_eigvals, max_imag_eigvals))

    return(final_results_dict, total_magnetic_flux_values, max_real_components, imag_eigvals_bounds_tracker, smallest_real_eigvals)

#######################################################################################################################

from Axial_Magnetic_Field.symmetry import c3symmetry_site_label_transform_sequenceofsites_hyperbolicq3
from Fundamental.NonHermitian_Hamiltonian import site_assignment

def selfconsist_hartreefock_Hermitian_SublatticeBasis_DisorderAlreadyAdded_SaveRawOutputs_MoreEfficient_CAverage(p, num_levels, initial_tight_bind_ham, initial_guess, tolerance, hfcoeff_list):
    total_magnetic_flux_values = np.array([])
    real_component_tracker = np.array([])
    nnn_hopping_indices = get_nnn_hopping_indices_hyperbolicq3(p, num_levels)
    nnn_hopping_indices_pos, nnn_hopping_indices_neg = get_nnn_hopping_indices_signspecific_hyperbolicq3(p, num_levels)
    total_num_sites = int(points(p, 3, num_levels)[1])
    final_results_dict = {}

    plaq_boundary_sites = get_plaquet_boundary_sites_hyperbolicq3(p, num_levels)
    plaq_isites, plaq_jsites = get_nnn_hoppings_around_plaquet_sublatticebasis_hyperbolicq3(p, num_levels, plaq_boundary_sites)

    rotbasismat = c3symmetry_site_label_transform_sequenceofsites_hyperbolicq3(p, num_levels)
    nnn_hopping_indices_ogbasis = get_nnn_hopping_indices_hyperbolicq3_ogbasis(p, num_levels)
    asites, bsites = site_assignment(p, 3, num_levels, H0(p, 3, num_levels))
    sublatticebasis = np.concatenate((asites, bsites))
    sublatticebasis = np.array([int(sbl) for sbl in sublatticebasis])
    reversebasis = np.array([], dtype=int)
    for i in range(np.size(initial_tight_bind_ham, 0)):
        reversebasis = np.append(reversebasis, np.where(i == sublatticebasis)[0])

    for hfcoeff in hfcoeff_list:  # Go over all desired V2 values
        print(hfcoeff)
        final_results_dict['V2={}'.format(hfcoeff)] = np.zeros((3, len(nnn_hopping_indices[0])), dtype=np.complex_)
        converge_status = False
        old_t2mat = np.zeros((total_num_sites, total_num_sites), dtype=np.complex_)
        old_t2mat[nnn_hopping_indices_pos] = 1j * initial_guess
        old_t2mat[nnn_hopping_indices_neg] = -1j * initial_guess
        ham = initial_tight_bind_ham - (hfcoeff * old_t2mat)
        converge_progress_count = 0
        while converge_status == False:
            start_time = time.time()
            # print('Diagonalization Start')
            eners, right_eigvec_matrix = scipy.linalg.eigh(ham)  # diagonalize ham to get energies and eigvecs
            del ham
            # print('Diagonalization Done')
            left_eigvec_matrix = np.conj(right_eigvec_matrix)
            # left_eigvec_matrix, right_eigvec_matrix = biortogonal_normalize(left_eigvec_matrix, right_eigvec_matrix)
            # print('Normalization Done')
            negative_energy_indices = np.where(eners < 0)  # Find indices where E<0
            left_negative_energy_eigvec_matrix = left_eigvec_matrix[:, negative_energy_indices[0]]
            right_negative_energy_eigvec_matrix = right_eigvec_matrix[:, negative_energy_indices[0]]
            del eners
            del left_eigvec_matrix
            del right_eigvec_matrix
            del negative_energy_indices
            # print('Getting filled states done')

            new_t2mat = np.zeros((total_num_sites, total_num_sites), dtype=np.complex_)
            for nhi in range(len(nnn_hopping_indices[0])):
                new_t2mat[(nnn_hopping_indices[1])[nhi], (nnn_hopping_indices[0])[nhi]] = np.sum(left_negative_energy_eigvec_matrix[(nnn_hopping_indices[0])[nhi], :] * right_negative_energy_eigvec_matrix[(nnn_hopping_indices[1])[nhi], :])
            # print('Calculating site probabilities done')
            left_negative_energy_eigvec_matrix_ogbasis = left_negative_energy_eigvec_matrix[reversebasis, :]
            right_negative_energy_eigvec_matrix_ogbasis = right_negative_energy_eigvec_matrix[reversebasis, :]
            for rc in range(np.size(rotbasismat, 0)):
                rotated_term_to_add = np.zeros((total_num_sites, total_num_sites), dtype=np.complex_)
                left_negative_energy_eigvec_matrix_ogbasis_rotated = np.copy(left_negative_energy_eigvec_matrix_ogbasis)[rotbasismat[rc, :], :]
                right_negative_energy_eigvec_matrix_ogbasis_rotated = np.copy(right_negative_energy_eigvec_matrix_ogbasis)[rotbasismat[rc, :], :]
                for nhi in range(len(nnn_hopping_indices_ogbasis[0])):
                    rotated_term_to_add[(nnn_hopping_indices_ogbasis[1])[nhi], (nnn_hopping_indices_ogbasis[0])[nhi]] = np.sum(left_negative_energy_eigvec_matrix_ogbasis_rotated[(nnn_hopping_indices_ogbasis[0])[nhi], :] * right_negative_energy_eigvec_matrix_ogbasis_rotated[(nnn_hopping_indices_ogbasis[1])[nhi], :])
                new_t2mat = new_t2mat + rotated_term_to_add[:, sublatticebasis][sublatticebasis, :]
            new_t2mat = new_t2mat / (np.size(rotbasismat, 0) + 1)

            del left_negative_energy_eigvec_matrix
            del right_negative_energy_eigvec_matrix
            del left_negative_energy_eigvec_matrix_ogbasis
            del right_negative_energy_eigvec_matrix_ogbasis
            del left_negative_energy_eigvec_matrix_ogbasis_rotated
            del right_negative_energy_eigvec_matrix_ogbasis_rotated

            print('Max Real Component: {}'.format(np.max(np.abs(np.real(new_t2mat)))))
            print('Total flux through system: {}'.format(np.sum(np.imag(calculate_haldane_current_around_plaquet_sublatticebasis(new_t2mat, plaq_isites, plaq_jsites)))))
            total_magnetic_flux_values = np.append(total_magnetic_flux_values,
                                                   np.sum(np.imag(calculate_haldane_current_around_plaquet_sublatticebasis(new_t2mat, plaq_isites, plaq_jsites))))
            real_component_tracker = np.append(real_component_tracker, np.max(np.abs(np.real(new_t2mat))))

            converge_status = self_consistent_check_2d_imag(new_t2mat, old_t2mat, tolerance)
            # print('Getting convergence status done')
            ham = initial_tight_bind_ham - (hfcoeff * new_t2mat)
            # print('Getting new hamiltonian done')
            converge_progress_count += 1
            print(converge_progress_count, np.max(np.abs(np.imag(new_t2mat - old_t2mat))), 'Time taken: {}'.format(time.time() - start_time))
            old_t2mat = np.copy(new_t2mat)
        print('Convergence done')
        (final_results_dict['V2={}'.format(hfcoeff)])[0,:] = nnn_hopping_indices[0]
        (final_results_dict['V2={}'.format(hfcoeff)])[1,:] = nnn_hopping_indices[1]
        t2values_flattened_ordered = np.array([])
        for ncii in range(len(nnn_hopping_indices[0])):
            t2values_flattened_ordered = np.append(t2values_flattened_ordered, new_t2mat[(nnn_hopping_indices[0])[ncii], (nnn_hopping_indices[1])[ncii]])
        (final_results_dict['V2={}'.format(hfcoeff)])[2, :] = t2values_flattened_ordered
    return(final_results_dict, total_magnetic_flux_values, real_component_tracker)

#Copy of above function but with a few changes to discard real part of new haldane term after every iteration
def selfconsist_hartreefock_Hermitian_SublatticeBasis_DisorderAlreadyAdded_SaveRawOutputs_MoreEfficient_CAverage_DiscardReal(p, num_levels, initial_tight_bind_ham, initial_guess, tolerance, hfcoeff_list):
    total_magnetic_flux_values = np.array([])
    real_component_tracker = np.array([])
    nnn_hopping_indices = get_nnn_hopping_indices_hyperbolicq3(p, num_levels)
    nnn_hopping_indices_pos, nnn_hopping_indices_neg = get_nnn_hopping_indices_signspecific_hyperbolicq3(p, num_levels)
    total_num_sites = int(points(p, 3, num_levels)[1])
    final_results_dict = {}

    plaq_boundary_sites = get_plaquet_boundary_sites_hyperbolicq3(p, num_levels)
    plaq_isites, plaq_jsites = get_nnn_hoppings_around_plaquet_sublatticebasis_hyperbolicq3(p, num_levels, plaq_boundary_sites)

    rotbasismat = c3symmetry_site_label_transform_sequenceofsites_hyperbolicq3(p, num_levels)
    nnn_hopping_indices_ogbasis = get_nnn_hopping_indices_hyperbolicq3_ogbasis(p, num_levels)
    asites, bsites = site_assignment(p, 3, num_levels, H0(p, 3, num_levels))
    sublatticebasis = np.concatenate((asites, bsites))
    sublatticebasis = np.array([int(sbl) for sbl in sublatticebasis])
    reversebasis = np.array([], dtype=int)
    for i in range(np.size(initial_tight_bind_ham, 0)):
        reversebasis = np.append(reversebasis, np.where(i == sublatticebasis)[0])

    for hfcoeff in hfcoeff_list:  # Go over all desired V2 values
        print(hfcoeff)
        final_results_dict['V2={}'.format(hfcoeff)] = np.zeros((3, len(nnn_hopping_indices[0])), dtype=np.complex_)
        converge_status = False
        old_t2mat = np.zeros((total_num_sites, total_num_sites), dtype=np.complex_)
        old_t2mat[nnn_hopping_indices_pos] = 1j * initial_guess
        old_t2mat[nnn_hopping_indices_neg] = -1j * initial_guess
        ham = initial_tight_bind_ham - (hfcoeff * old_t2mat)
        converge_progress_count = 0
        while converge_status == False:
            start_time = time.time()
            # print('Diagonalization Start')
            eners, right_eigvec_matrix = scipy.linalg.eigh(ham)  # diagonalize ham to get energies and eigvecs
            del ham
            # print('Diagonalization Done')
            left_eigvec_matrix = np.conj(right_eigvec_matrix)
            # left_eigvec_matrix, right_eigvec_matrix = biortogonal_normalize(left_eigvec_matrix, right_eigvec_matrix)
            # print('Normalization Done')
            negative_energy_indices = np.where(eners < 0)  # Find indices where E<0
            left_negative_energy_eigvec_matrix = left_eigvec_matrix[:, negative_energy_indices[0]]
            right_negative_energy_eigvec_matrix = right_eigvec_matrix[:, negative_energy_indices[0]]
            del eners
            del left_eigvec_matrix
            del right_eigvec_matrix
            del negative_energy_indices
            # print('Getting filled states done')

            new_t2mat = np.zeros((total_num_sites, total_num_sites), dtype=np.complex_)
            for nhi in range(len(nnn_hopping_indices[0])):
                new_t2mat[(nnn_hopping_indices[1])[nhi], (nnn_hopping_indices[0])[nhi]] = np.sum(left_negative_energy_eigvec_matrix[(nnn_hopping_indices[0])[nhi], :] * right_negative_energy_eigvec_matrix[(nnn_hopping_indices[1])[nhi], :])
            # print('Calculating site probabilities done')
            left_negative_energy_eigvec_matrix_ogbasis = left_negative_energy_eigvec_matrix[reversebasis, :]
            right_negative_energy_eigvec_matrix_ogbasis = right_negative_energy_eigvec_matrix[reversebasis, :]
            for rc in range(np.size(rotbasismat, 0)):
                rotated_term_to_add = np.zeros((total_num_sites, total_num_sites), dtype=np.complex_)
                left_negative_energy_eigvec_matrix_ogbasis_rotated = np.copy(left_negative_energy_eigvec_matrix_ogbasis)[rotbasismat[rc, :], :]
                right_negative_energy_eigvec_matrix_ogbasis_rotated = np.copy(right_negative_energy_eigvec_matrix_ogbasis)[rotbasismat[rc, :], :]
                for nhi in range(len(nnn_hopping_indices_ogbasis[0])):
                    rotated_term_to_add[(nnn_hopping_indices_ogbasis[1])[nhi], (nnn_hopping_indices_ogbasis[0])[nhi]] = np.sum(left_negative_energy_eigvec_matrix_ogbasis_rotated[(nnn_hopping_indices_ogbasis[0])[nhi], :] * right_negative_energy_eigvec_matrix_ogbasis_rotated[(nnn_hopping_indices_ogbasis[1])[nhi], :])
                new_t2mat = new_t2mat + rotated_term_to_add[:, sublatticebasis][sublatticebasis, :]
            new_t2mat = new_t2mat / (np.size(rotbasismat, 0) + 1)

            del left_negative_energy_eigvec_matrix
            del right_negative_energy_eigvec_matrix
            del left_negative_energy_eigvec_matrix_ogbasis
            del right_negative_energy_eigvec_matrix_ogbasis
            del left_negative_energy_eigvec_matrix_ogbasis_rotated
            del right_negative_energy_eigvec_matrix_ogbasis_rotated

            print('Max Real Component: {}'.format(np.max(np.abs(np.real(new_t2mat)))))
            print('Total flux through system: {}'.format(np.sum(np.imag(calculate_haldane_current_around_plaquet_sublatticebasis(new_t2mat, plaq_isites, plaq_jsites)))))
            total_magnetic_flux_values = np.append(total_magnetic_flux_values,
                                                   np.sum(np.imag(calculate_haldane_current_around_plaquet_sublatticebasis(new_t2mat, plaq_isites, plaq_jsites))))
            real_component_tracker = np.append(real_component_tracker, np.max(np.abs(np.real(new_t2mat))))

            converge_status = self_consistent_check_2d_imag(new_t2mat, old_t2mat, tolerance)
            # print('Getting convergence status done')
            ham = initial_tight_bind_ham - (hfcoeff * 1j*np.imag(new_t2mat)) ###Only changed line to discard real part###
            # print('Getting new hamiltonian done')
            converge_progress_count += 1
            print(converge_progress_count, np.max(np.abs(np.imag(new_t2mat - old_t2mat))), 'Time taken: {}'.format(time.time() - start_time))
            old_t2mat = np.copy(new_t2mat)
        print('Convergence done')
        (final_results_dict['V2={}'.format(hfcoeff)])[0,:] = nnn_hopping_indices[0]
        (final_results_dict['V2={}'.format(hfcoeff)])[1,:] = nnn_hopping_indices[1]
        t2values_flattened_ordered = np.array([])
        for ncii in range(len(nnn_hopping_indices[0])):
            t2values_flattened_ordered = np.append(t2values_flattened_ordered, new_t2mat[(nnn_hopping_indices[0])[ncii], (nnn_hopping_indices[1])[ncii]])
        (final_results_dict['V2={}'.format(hfcoeff)])[2, :] = t2values_flattened_ordered
    return(final_results_dict, total_magnetic_flux_values, real_component_tracker)

# Copy of above function but with small changes to make it approrpiate for NH SCHF run
def selfconsist_hartreefock_NonHermitian_SublatticeBasis_DisorderAlreadyAdded_SaveRawOutputs_MoreEfficient_CAverage_DiscardReal(p, num_levels, initial_tight_bind_ham, initial_guess, tolerance, hfcoeff_list):
    total_magnetic_flux_values = np.array([])
    real_component_tracker = np.array([])

    #Extra trackers to track imaginary part of the eigenvalues since adding NH may mess up realness of eigenvalue spectrum
    min_imag_eigvals = np.array([])
    max_imag_eigvals = np.array([])
    # Also another tracker to track smallest magnitude real component of eigenvalues
    smallest_real_eigvals = np.array([])

    nnn_hopping_indices = get_nnn_hopping_indices_hyperbolicq3(p, num_levels)
    nnn_hopping_indices_pos, nnn_hopping_indices_neg = get_nnn_hopping_indices_signspecific_hyperbolicq3(p, num_levels)
    total_num_sites = int(points(p, 3, num_levels)[1])
    final_results_dict = {}

    plaq_boundary_sites = get_plaquet_boundary_sites_hyperbolicq3(p, num_levels)
    plaq_isites, plaq_jsites = get_nnn_hoppings_around_plaquet_sublatticebasis_hyperbolicq3(p, num_levels, plaq_boundary_sites)

    rotbasismat = c3symmetry_site_label_transform_sequenceofsites_hyperbolicq3(p, num_levels)
    nnn_hopping_indices_ogbasis = get_nnn_hopping_indices_hyperbolicq3_ogbasis(p, num_levels)
    asites, bsites = site_assignment(p, 3, num_levels, H0(p, 3, num_levels))
    sublatticebasis = np.concatenate((asites, bsites))
    sublatticebasis = np.array([int(sbl) for sbl in sublatticebasis])
    reversebasis = np.array([], dtype=int)
    for i in range(np.size(initial_tight_bind_ham, 0)):
        reversebasis = np.append(reversebasis, np.where(i == sublatticebasis)[0])

    for hfcoeff in hfcoeff_list:  # Go over all desired V2 values
        print(hfcoeff)
        final_results_dict['V2={}'.format(hfcoeff)] = np.zeros((3, len(nnn_hopping_indices[0])), dtype=np.complex_)
        converge_status = False
        old_t2mat = np.zeros((total_num_sites, total_num_sites), dtype=np.complex_)
        old_t2mat[nnn_hopping_indices_pos] = 1j * initial_guess
        old_t2mat[nnn_hopping_indices_neg] = -1j * initial_guess
        ham = initial_tight_bind_ham - (hfcoeff * old_t2mat)
        converge_progress_count = 0
        while converge_status == False:
            start_time = time.time()
            # print('Diagonalization Start')

            #This line is changed to use eig instead of eigh #1/2 changes for NH
            eners, left_eigvec_matrix, right_eigvec_matrix = scipy.linalg.eig(ham, left=True, right=True)  # diagonalize ham to get energies and eigvecs

            del ham
            # print('Diagonalization Done')

            #Two lines changed to biorthogonally normalize left and right eigenvectors #2/2 changes for NH
            left_eigvec_matrix = np.conj(left_eigvec_matrix)
            left_eigvec_matrix, right_eigvec_matrix = biortogonal_normalize(left_eigvec_matrix, right_eigvec_matrix)

            # Appending the extra imag part trackers that I now have for NH SCHF calculation loop
            max_imag_eigvals = np.append(max_imag_eigvals, np.max(np.imag(eners)))
            min_imag_eigvals = np.append(min_imag_eigvals, np.min(np.imag(eners)))
            # Also append smallest real component of eigval tracker
            smallest_real_eigvals = np.append(smallest_real_eigvals, np.min(np.abs(np.real(eners))))

            # print('Normalization Done')
            negative_energy_indices = np.where(np.real(eners) < 0)  # Find indices where E<0; add np.real to ensure behaves as expected
            left_negative_energy_eigvec_matrix = left_eigvec_matrix[:, negative_energy_indices[0]]
            right_negative_energy_eigvec_matrix = right_eigvec_matrix[:, negative_energy_indices[0]]
            del eners
            del left_eigvec_matrix
            del right_eigvec_matrix
            del negative_energy_indices
            # print('Getting filled states done')

            new_t2mat = np.zeros((total_num_sites, total_num_sites), dtype=np.complex_)
            for nhi in range(len(nnn_hopping_indices[0])):
                new_t2mat[(nnn_hopping_indices[1])[nhi], (nnn_hopping_indices[0])[nhi]] = np.sum(left_negative_energy_eigvec_matrix[(nnn_hopping_indices[0])[nhi], :] * right_negative_energy_eigvec_matrix[(nnn_hopping_indices[1])[nhi], :])
            # print('Calculating site probabilities done')
            left_negative_energy_eigvec_matrix_ogbasis = left_negative_energy_eigvec_matrix[reversebasis, :]
            right_negative_energy_eigvec_matrix_ogbasis = right_negative_energy_eigvec_matrix[reversebasis, :]
            for rc in range(np.size(rotbasismat, 0)):
                rotated_term_to_add = np.zeros((total_num_sites, total_num_sites), dtype=np.complex_)
                left_negative_energy_eigvec_matrix_ogbasis_rotated = np.copy(left_negative_energy_eigvec_matrix_ogbasis)[rotbasismat[rc, :], :]
                right_negative_energy_eigvec_matrix_ogbasis_rotated = np.copy(right_negative_energy_eigvec_matrix_ogbasis)[rotbasismat[rc, :], :]
                for nhi in range(len(nnn_hopping_indices_ogbasis[0])):
                    rotated_term_to_add[(nnn_hopping_indices_ogbasis[1])[nhi], (nnn_hopping_indices_ogbasis[0])[nhi]] = np.sum(left_negative_energy_eigvec_matrix_ogbasis_rotated[(nnn_hopping_indices_ogbasis[0])[nhi], :] * right_negative_energy_eigvec_matrix_ogbasis_rotated[(nnn_hopping_indices_ogbasis[1])[nhi], :])
                new_t2mat = new_t2mat + rotated_term_to_add[:, sublatticebasis][sublatticebasis, :]
            new_t2mat = new_t2mat / (np.size(rotbasismat, 0) + 1)

            del left_negative_energy_eigvec_matrix
            del right_negative_energy_eigvec_matrix
            del left_negative_energy_eigvec_matrix_ogbasis
            del right_negative_energy_eigvec_matrix_ogbasis
            del left_negative_energy_eigvec_matrix_ogbasis_rotated
            del right_negative_energy_eigvec_matrix_ogbasis_rotated

            print('Max Real Component: {}'.format(np.max(np.abs(np.real(new_t2mat)))))
            print('Total flux through system: {}'.format(np.sum(np.imag(calculate_haldane_current_around_plaquet_sublatticebasis(new_t2mat, plaq_isites, plaq_jsites)))))
            total_magnetic_flux_values = np.append(total_magnetic_flux_values,
                                                   np.sum(np.imag(calculate_haldane_current_around_plaquet_sublatticebasis(new_t2mat, plaq_isites, plaq_jsites))))
            real_component_tracker = np.append(real_component_tracker, np.max(np.abs(np.real(new_t2mat))))

            converge_status = self_consistent_check_2d_imag(new_t2mat, old_t2mat, tolerance)
            # print('Getting convergence status done')
            ham = initial_tight_bind_ham - (hfcoeff * 1j*np.imag(new_t2mat)) ###Only changed line to discard real part###
            # print('Getting new hamiltonian done')
            converge_progress_count += 1
            print(converge_progress_count, np.max(np.abs(np.imag(new_t2mat - old_t2mat))), 'Time taken: {}'.format(time.time() - start_time))
            old_t2mat = np.copy(new_t2mat)
        print('Convergence done')
        (final_results_dict['V2={}'.format(hfcoeff)])[0,:] = nnn_hopping_indices[0]
        (final_results_dict['V2={}'.format(hfcoeff)])[1,:] = nnn_hopping_indices[1]
        t2values_flattened_ordered = np.array([])
        for ncii in range(len(nnn_hopping_indices[0])):
            t2values_flattened_ordered = np.append(t2values_flattened_ordered, new_t2mat[(nnn_hopping_indices[0])[ncii], (nnn_hopping_indices[1])[ncii]])
        (final_results_dict['V2={}'.format(hfcoeff)])[2, :] = t2values_flattened_ordered

    # Extra bit of code to stack imaginary eigenvalue trackers
    imag_eigvals_bounds_tracker = np.vstack((min_imag_eigvals, max_imag_eigvals))

    return(final_results_dict, total_magnetic_flux_values, real_component_tracker, imag_eigvals_bounds_tracker, smallest_real_eigvals)

###########################################################################

from Fundamental.Honeycomb_Lattice import honeycomb_lattice
from Fundamental.Honeycomb_Lattice import honeycomb_lattice_periodic_boundary
from Fundamental.Honeycomb_Lattice import honeycomb_points
from Fundamental.Local import get_plaquet_boundary_sites_honeycomb_PBC
from Fundamental.Local import get_nnn_hoppings_around_plaquet_sublatticebasis_honeycomb_PBC


def haldane_current_termonly_honeycomb_PBC(num_levels, tight_bind_ham_OBC, t2):
    haldanemat = haldane_current_termonly_honeycomb(num_levels, tight_bind_ham_OBC, t2)

    #Later I will flip sign of all b-b hops thus need to reverse original flip in haldane_current_termonly_honeycomb for original b-b hops
    asites, bsites = site_assignment_honeycomb(num_levels, honeycomb_lattice(num_levels))
    bsites = [int(b) for b in bsites]
    haldanemat[bsites, :] = np.conj(haldanemat[bsites, :])

    lastgensites = np.arange(np.sum(honeycomb_points(num_levels)[0][:-1]), np.sum(honeycomb_points(num_levels)[1]), dtype=int)

    honeycomb_OBC = honeycomb_lattice(num_levels)
    honeycomb_PBC = honeycomb_lattice_periodic_boundary(num_levels)
    PBC_links = honeycomb_PBC - honeycomb_OBC

    for row in lastgensites:
        rowindex = np.where(row == lastgensites)[0][0]
        pbcconnsite = np.where(PBC_links[row, :] != 0)[0]
        if len(pbcconnsite) == 0:
            pbcconnsite_oneup = np.where(PBC_links[np.take(lastgensites, rowindex+1, mode='wrap'), :] != 0)[0]
            pbcconnsite_onedown = np.where(PBC_links[np.take(lastgensites, rowindex - 1, mode='wrap'), :] != 0)[0]

            haldanemat[row, pbcconnsite_oneup] = -1j * t2
            haldanemat[pbcconnsite_oneup, row] = 1j * t2
            haldanemat[row, pbcconnsite_onedown] = 1j * t2
            haldanemat[pbcconnsite_onedown, row] = -1j * t2


        #Corner case 1
        elif len(pbcconnsite) == 1 and np.where(PBC_links[np.take(lastgensites, row+1, mode='wrap'), :] != 0)[0] == 1:
            pbcconnsiteindex = np.where(pbcconnsite == lastgensites)[0][0]
            haldanemat[row, np.take(lastgensites, pbcconnsiteindex + 1, mode='wrap')] = -1j * t2
            haldanemat[np.take(lastgensites, pbcconnsiteindex + 1, mode='wrap'), row] = 1j * t2
            haldanemat[row, np.take(lastgensites, pbcconnsiteindex - 1, mode='wrap')] = 1j * t2
            haldanemat[np.take(lastgensites, pbcconnsiteindex - 1, mode='wrap'), row] = -1j * t2

            haldanemat[row, np.where(PBC_links[np.take(lastgensites, rowindex+1, mode='wrap'), :] != 0)[0]] = -1j * t2
            haldanemat[np.where(PBC_links[np.take(lastgensites, rowindex+1, mode='wrap'), :] != 0)[0], row] = 1j * t2


        # Corner case 2
        elif len(pbcconnsite) == 1 and np.where(PBC_links[np.take(lastgensites, row-1, mode='wrap'), :] != 0)[0] == 1:
            pbcconnsiteindex = np.where(pbcconnsite == lastgensites)[0][0]
            haldanemat[row, np.take(lastgensites, pbcconnsiteindex + 1, mode='wrap')] = -1j * t2
            haldanemat[np.take(lastgensites, pbcconnsiteindex + 1, mode='wrap'), row] = 1j * t2
            haldanemat[row, np.take(lastgensites, pbcconnsiteindex - 1, mode='wrap')] = 1j * t2
            haldanemat[np.take(lastgensites, pbcconnsiteindex - 1, mode='wrap'), row] = -1j * t2

            haldanemat[row, np.where(PBC_links[np.take(lastgensites, rowindex - 1, mode='wrap'), :] != 0)[0]] = 1j * t2
            haldanemat[np.where(PBC_links[np.take(lastgensites, rowindex - 1, mode='wrap'), :] != 0)[0], row] = -1j * t2

        else:
            pbcconnsiteindex = np.where(pbcconnsite == lastgensites)[0][0]
            haldanemat[row, np.take(lastgensites, pbcconnsiteindex + 1, mode='wrap')] = -1j * t2
            haldanemat[np.take(lastgensites, pbcconnsiteindex + 1, mode='wrap'), row] = 1j * t2
            haldanemat[row, np.take(lastgensites, pbcconnsiteindex - 1, mode='wrap')] = 1j * t2
            haldanemat[np.take(lastgensites, pbcconnsiteindex - 1, mode='wrap'), row] = -1j * t2

    # Last bit of code to reverse sign of phase for NNN hopping on B sublattice
    haldanemat[bsites, :] = np.conj(haldanemat[bsites, :])

    return(haldanemat)


def haldane_current_termonly_honeycomb_sublatticeblockform_PBC(num_levels, tight_bind_ham_OBC, t2):
    asites, bsites = site_assignment_honeycomb(num_levels, tight_bind_ham_OBC)
    asites = [int(a) for a in asites]
    bsites = [int(b) for b in bsites]
    newbasis = np.concatenate((asites, bsites))
    haldane_mat_ogbasis = haldane_current_termonly_honeycomb_PBC(num_levels, tight_bind_ham_OBC, t2)
    haldane_mat_newbasis = haldane_mat_ogbasis[:, newbasis][newbasis, :]
    return (haldane_mat_newbasis)

def get_nnn_hopping_indices_honeycomb_PBC(num_levels):
    tight_bind_ham_OBC = honeycomb_lattice(num_levels)
    t2 = 1
    return(np.where(haldane_current_termonly_honeycomb_PBC(num_levels, tight_bind_ham_OBC, t2) != 0))

def get_nnn_hopping_indices_honeycomb_sublatticebasis_PBC(num_levels):
    tight_bind_ham_OBC = honeycomb_lattice(num_levels)
    t2 = 1
    return(np.where(haldane_current_termonly_honeycomb_sublatticeblockform_PBC(num_levels, tight_bind_ham_OBC, t2) != 0))

def get_nnn_hopping_indices_signspecific_honeycomb_PBC(num_levels):
    tight_bind_ham_OBC = honeycomb_lattice(num_levels)
    haldane_mat_PBC = haldane_current_termonly_honeycomb_sublatticeblockform_PBC(num_levels, tight_bind_ham_OBC, 1)
    return (np.where(np.imag(haldane_mat_PBC) > 0), np.where(np.imag(haldane_mat_PBC) < 0))

#Copied from analogous function for OBC but now adapted for PBC
def selfconsist_hartreefock_Hermitian_SublatticeBasis_DisorderAlreadyAdded_SaveRawOutputs_MoreEfficient_honeycomb_C3Average_DiscardReal_PBC(num_levels, initial_tight_bind_ham, initial_guess, tolerance, hfcoeff_list):
    total_magnetic_flux_values = np.array([])
    max_real_components = np.array([])
    nnn_hopping_indices = get_nnn_hopping_indices_honeycomb_sublatticebasis_PBC(num_levels) ###use PBC version here instead
    nnn_hopping_indices_pos, nnn_hopping_indices_neg = get_nnn_hopping_indices_signspecific_honeycomb_PBC(num_levels) ###use PBC version here instead
    total_num_sites = int(honeycomb_points(num_levels)[1])
    final_results_dict = {}
    plaq_bound_sites = get_plaquet_boundary_sites_honeycomb_PBC(num_levels) ###use PBC version here instead
    plaq_nnn_hopping_isites, plaq_nnn_hopping_jsites = get_nnn_hoppings_around_plaquet_sublatticebasis_honeycomb_PBC(num_levels, plaq_bound_sites) ###use PBC version here instead

    rotated_basis_one, rotated_basis_two = return_c3symmetry_site_label_transform_basis_honeycomb(num_levels)
    c3basismat = np.vstack((rotated_basis_one, rotated_basis_two))
    del rotated_basis_one
    del rotated_basis_two

    asites, bsites = site_assignment_honeycomb(num_levels, honeycomb_lattice(num_levels))
    sublatticebasis = np.concatenate((asites, bsites))
    sublatticebasis = np.array([int(sbl) for sbl in sublatticebasis])
    reversebasis = np.array([], dtype=int)
    for i in np.arange(np.size(initial_tight_bind_ham, 0)):
        reversebasis = np.append(reversebasis, np.where(i == sublatticebasis)[0])

    nnn_hopping_indices_ogbasis = get_nnn_hopping_indices_honeycomb_PBC(num_levels) ###Use pbc version instead

    for hfcoeff in hfcoeff_list:  # Go over all desired V2 values
        print(hfcoeff)
        final_results_dict['V2={}'.format(hfcoeff)] = np.zeros((3, len(nnn_hopping_indices[0])), dtype=np.complex_)
        converge_status = False
        old_t2mat = np.zeros((total_num_sites, total_num_sites), dtype=np.complex_)

        old_t2mat[nnn_hopping_indices_pos] = 1j * initial_guess
        old_t2mat[nnn_hopping_indices_neg] = -1j * initial_guess

        ham = initial_tight_bind_ham - (hfcoeff * old_t2mat)

        converge_progress_count = 0

        print('sum of phase over all plaquet', np.sum(np.imag(calculate_haldane_current_around_plaquet_sublatticebasis(old_t2mat, plaq_nnn_hopping_isites, plaq_nnn_hopping_jsites))))
        print('Max real component: {}'.format(np.max(np.abs(np.real(old_t2mat)))))

        while converge_status == False:
            start_time = time.time()
            # print('Diagonalization Start')
            eners, right_eigvec_matrix = scipy.linalg.eigh(ham)
            del ham
            # print('Diagonalization Done')
            left_eigvec_matrix = np.conj(right_eigvec_matrix)
            negative_energy_indices = np.where(eners < 0)  # Find indices where E<0
            left_negative_energy_eigvec_matrix = left_eigvec_matrix[:, negative_energy_indices[0]]
            right_negative_energy_eigvec_matrix = right_eigvec_matrix[:, negative_energy_indices[0]]
            del eners
            del left_eigvec_matrix
            del right_eigvec_matrix
            del negative_energy_indices
            # print('Getting filled states done')
            new_t2mat = np.zeros((total_num_sites, total_num_sites), dtype=np.complex_)
            for nhi in range(len(nnn_hopping_indices[0])):
                new_t2mat[(nnn_hopping_indices[0])[nhi], (nnn_hopping_indices[1])[nhi]] = np.sum(left_negative_energy_eigvec_matrix[(nnn_hopping_indices[1])[nhi], :] * right_negative_energy_eigvec_matrix[(nnn_hopping_indices[0])[nhi], :])
            left_negative_energy_eigvec_matrix_ogbasis = left_negative_energy_eigvec_matrix[reversebasis, :]
            right_negative_energy_eigvec_matrix_ogbasis = right_negative_energy_eigvec_matrix[reversebasis, :]
            for rc in range(np.size(c3basismat, 0)):
                rotated_term_to_add = np.zeros((total_num_sites, total_num_sites), dtype=np.complex_)
                left_negative_energy_eigvec_matrix_ogbasis_rotated = np.copy(left_negative_energy_eigvec_matrix_ogbasis)[c3basismat[rc, :], :]
                right_negative_energy_eigvec_matrix_ogbasis_rotated = np.copy(right_negative_energy_eigvec_matrix_ogbasis)[c3basismat[rc, :], :]
                for nhi in range(len(nnn_hopping_indices_ogbasis[0])):
                    rotated_term_to_add[(nnn_hopping_indices_ogbasis[1])[nhi], (nnn_hopping_indices_ogbasis[0])[nhi]] = np.sum(left_negative_energy_eigvec_matrix_ogbasis_rotated[(nnn_hopping_indices_ogbasis[0])[nhi], :] * right_negative_energy_eigvec_matrix_ogbasis_rotated[(nnn_hopping_indices_ogbasis[1])[nhi], :])
                new_t2mat = new_t2mat + rotated_term_to_add[:, sublatticebasis][sublatticebasis, :]
            new_t2mat = new_t2mat / (np.size(c3basismat, 0) + 1)

            del left_negative_energy_eigvec_matrix
            del right_negative_energy_eigvec_matrix
            del left_negative_energy_eigvec_matrix_ogbasis
            del right_negative_energy_eigvec_matrix_ogbasis
            del left_negative_energy_eigvec_matrix_ogbasis_rotated
            del right_negative_energy_eigvec_matrix_ogbasis_rotated
            del rotated_term_to_add

            print('sum of phase over all plaquets: {}'.format(np.sum(np.imag(calculate_haldane_current_around_plaquet_sublatticebasis(new_t2mat, plaq_nnn_hopping_isites, plaq_nnn_hopping_jsites)))))
            print('Max real component: {}'.format(np.max(np.abs(np.real(new_t2mat)))))
            total_magnetic_flux_values = np.append(total_magnetic_flux_values,
                                                   np.sum(np.imag(calculate_haldane_current_around_plaquet_sublatticebasis(new_t2mat, plaq_nnn_hopping_isites, plaq_nnn_hopping_jsites))))
            max_real_components = np.append(max_real_components, np.max(np.abs(np.real(new_t2mat))))
            converge_status = self_consistent_check_2d_imag(new_t2mat, old_t2mat, tolerance)
            converge_progress_count += 1
            print(converge_progress_count, np.max(np.abs(np.imag(new_t2mat - old_t2mat))), 'Time taken: {}'.format(time.time() - start_time))

            ham = initial_tight_bind_ham - (hfcoeff * np.imag(new_t2mat))*1j ###Only difference for "DiscardReal"
            print(np.imag(new_t2mat[0,1]))
            old_t2mat = np.copy(new_t2mat)

        print('Convergence done')
        (final_results_dict['V2={}'.format(hfcoeff)])[0,:] = nnn_hopping_indices[0]
        (final_results_dict['V2={}'.format(hfcoeff)])[1,:] = nnn_hopping_indices[1]
        t2values_flattened_ordered = np.array([])
        for ncii in range(len(nnn_hopping_indices[0])):
            t2values_flattened_ordered = np.append(t2values_flattened_ordered, new_t2mat[(nnn_hopping_indices[0])[ncii], (nnn_hopping_indices[1])[ncii]])
        (final_results_dict['V2={}'.format(hfcoeff)])[2, :] = t2values_flattened_ordered

    return(final_results_dict, total_magnetic_flux_values, max_real_components)




