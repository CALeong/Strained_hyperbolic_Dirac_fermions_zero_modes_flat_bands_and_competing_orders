import numpy as np
from Fundamental.General_Hamiltonian import general_q3_hamiltonian
from Fundamental.General_Hamiltonian import get_points_that_connect_with_prev_layer_q3_general
from Fundamental.General_Hamiltonian import get_if_point_is_connected_with_upper_layer_q3_general
from Fundamental.Number_Points import points
from Fundamental.NonHermitian_Hamiltonian import site_assignment

def fill_in_between_intergen_sites(sites_on_level, intergen_sites, N_from_center_results):
    #     print(sites_on_level)
    for i in range(len(intergen_sites) - 1):
        betweensites = np.arange(intergen_sites[i], intergen_sites[i + 1] + 1)
        if len(betweensites) % 2 == 0:  # On top of three side
            for bs in range(int(len(betweensites) / 2 - 1)):
                N_from_center_results[betweensites[bs + 1]] = N_from_center_results[betweensites[bs]] + 1
                N_from_center_results[betweensites[-bs - 1 - 1]] = N_from_center_results[betweensites[-bs - 1]] + 1
        elif len(betweensites) % 2 == 1:  # On top of four side
            for bs in range(int((len(betweensites) - 1) / 2 - 1)):
                N_from_center_results[betweensites[bs + 1]] = N_from_center_results[betweensites[bs]] + 1
                N_from_center_results[betweensites[-bs - 1 - 1]] = N_from_center_results[betweensites[-bs - 1]] + 1
            N_from_center_results[betweensites[int((len(betweensites) - 1) / 2)]] = N_from_center_results[betweensites[int((len(betweensites) - 1) / 2 - 1)]] + 1

    # Extra bit to account for last in-between region between first intergen and last intergen sites
    betweensites = np.append(np.arange(intergen_sites[-1], sites_on_level[-1] + 1), intergen_sites[0])
    betweensites = np.array([int(i) for i in betweensites])
    if len(betweensites) % 2 == 0:  # On top of three side
        for bs in range(int(len(betweensites) / 2 - 1)):
            N_from_center_results[betweensites[bs + 1]] = N_from_center_results[betweensites[bs]] + 1
            N_from_center_results[betweensites[-bs - 1 - 1]] = N_from_center_results[betweensites[-bs - 1]] + 1
    elif len(betweensites) % 2 == 1:  # On top of four side
        for bs in range(int((len(betweensites) - 1) / 2 - 1)):
            N_from_center_results[betweensites[bs + 1]] = N_from_center_results[betweensites[bs]] + 1
            N_from_center_results[betweensites[-bs - 1 - 1]] = N_from_center_results[betweensites[-bs - 1]] + 1
        N_from_center_results[betweensites[int((len(betweensites) - 1) / 2)]] = N_from_center_results[betweensites[int((len(betweensites) - 1) / 2 - 1)]] + 1

    return (N_from_center_results)


def N_from_center_assignment_hyperbolic_q3(tight_bind_ham, p, num_levels):
    N_from_center_results = np.zeros(np.size(tight_bind_ham, 0))
    points_conected_with_next_level = np.arange(0, p)
    for level in range(1, num_levels):
        first_site_on_level = np.sum((points(p, 3, level + 1)[0])[:-1])
        sites_on_layer = np.arange(np.sum((points(p, 3, level + 1)[0])[:-1]), points(p, 3, level + 1)[1])

        points_conected_with_prev_level = get_points_that_connect_with_prev_layer_q3_general(p, level, first_site_on_level)
        points_conected_with_prev_level = np.array([int(i) for i in points_conected_with_prev_level])
        N_from_center_results[points_conected_with_prev_level] = N_from_center_results[points_conected_with_next_level] + 1

        points_conected_with_next_level = sites_on_layer[get_if_point_is_connected_with_upper_layer_q3_general(p, level)]
        points_conected_with_next_level = np.array([int(i) for i in points_conected_with_next_level])
        N_from_center_results = fill_in_between_intergen_sites(sites_on_layer, points_conected_with_prev_level, N_from_center_results)
    return (N_from_center_results)


def N_from_center_tight_bind_ham_hyperbolic_q3(p, num_levels, tight_bind_ham_sublattice_basis, N_from_center_assignment_list):
    t = 1
    num_sites = np.size(tight_bind_ham_sublattice_basis, 0)
    asites, bsites = site_assignment(p, 3, num_levels, general_q3_hamiltonian(p, num_levels).toarray())
    asites = np.array([int(i) for i in asites])
    bsites = np.array([int(i) for i in bsites])
    N_from_center_assignment_list = N_from_center_assignment_list[np.concatenate((asites, bsites))]
    phase_mod_pos = np.exp(N_from_center_assignment_list ** 2)
    phase_mod_neg = np.exp(-1 * N_from_center_assignment_list ** 2)
    conn_locs = np.where(tight_bind_ham_sublattice_basis[:int(num_sites / 2), :] != 0)
    tight_bind_ham_sublattice_basis[conn_locs] = phase_mod_pos[conn_locs[0]] * t * phase_mod_neg[conn_locs[1]]
    tight_bind_ham_sublattice_basis[int(num_sites / 2):, :int(num_sites / 2)] = np.transpose(tight_bind_ham_sublattice_basis[:int(num_sites / 2), int(num_sites / 2):])
    return (tight_bind_ham_sublattice_basis)