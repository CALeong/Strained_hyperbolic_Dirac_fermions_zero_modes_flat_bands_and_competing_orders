import numpy as np
from Fundamental.Honeycomb_Lattice import honeycomb_lattice_periodic_boundary
from Fundamental.Honeycomb_Lattice import honeycomb_points
from Fundamental.NonHermitian_Hamiltonian import NonHermitian_Honeycomb
from Fundamental.NonHermitian_Hamiltonian import site_assignment_honeycomb
import matplotlib.pyplot as plt
from Fundamental.Eigenvalue_Degeneracy_Fix import small_chaos


def points_on_level(level):
    sites_per_level = honeycomb_points(level)[0]
    #     print(sites_per_level)
    #     print(np.arange(np.sum(sites_per_level[:level-1]), np.sum(sites_per_level[:level])))
    return (np.arange(np.sum(sites_per_level[:level - 1]), np.sum(sites_per_level[:level])))


def corner_points_on_level(level):
    num_points_on_side = 2 * level - 1
    first_site_on_gen = np.sum(honeycomb_points(level)[0][:-1])
    first_corner_point = first_site_on_gen + (num_points_on_side - 1) / 2
    corner_points = np.array([first_corner_point, first_corner_point + 1])
    for i in range(5):
        corner_points = np.append(corner_points, corner_points[-2:] + num_points_on_side)
    return (corner_points)


def connected_to_next_gen_points(level):
    connected_to_next_gen_points_list = np.array([], dtype=int)
    corner_points = corner_points_on_level(level)
    first_site_on_gen = np.sum(honeycomb_points(level)[0][:-1])

    if level % 2 == 0:
        start_point = first_site_on_gen + 1
        for i in range(0, len(corner_points), 2):
            connected_to_next_gen_points_list = np.append(connected_to_next_gen_points_list,
                                                          np.arange(start_point, corner_points[i] + 1, 2, dtype=int))
            start_point = corner_points[i + 1]

        connected_to_next_gen_points_list = np.append(connected_to_next_gen_points_list,
                                                      np.arange(start_point, honeycomb_points(level)[1], 2, dtype=int))

    elif level % 2 == 1:
        start_point = first_site_on_gen
        for i in range(0, len(corner_points), 2):
            connected_to_next_gen_points_list = np.append(connected_to_next_gen_points_list,
                                                          np.arange(start_point, corner_points[i] + 1, 2, dtype=int))
            start_point = corner_points[i + 1]

        connected_to_next_gen_points_list = np.append(connected_to_next_gen_points_list,
                                                      np.arange(start_point, honeycomb_points(level)[1], 2, dtype=int))
    return (connected_to_next_gen_points_list)


def connected_to_prev_gen_points(level):
    points_on_layer = np.arange(np.sum((honeycomb_points(level)[0])[:-1]), honeycomb_points(level)[1])
    points_connected_to_next_gen = connected_to_next_gen_points(level)
    return (np.array([int(i) for i in np.setdiff1d(points_on_layer, points_connected_to_next_gen)]))


def fill_inbetween_intergen_sites(level, N_from_center_assignment_list):
    intergen_sites = connected_to_prev_gen_points(level)

    if level % 2 == 1:  # Handles added step needed for first site on odd numbered levels
        N_from_center_assignment_list[int(intergen_sites[0] - 1)] = N_from_center_assignment_list[int(intergen_sites[0])] + 1
    elif level % 2 == 0 and level != 2:  # Handles added step needed for last site on even numbered levels
        N_from_center_assignment_list[int(intergen_sites[-1] + 1)] = N_from_center_assignment_list[int(intergen_sites[0])] + 1
    elif level % 2 == 0 and level == 2:  # Extra special exception for second level
        N_from_center_assignment_list[int(intergen_sites[-1] + 1)] = N_from_center_assignment_list[int(intergen_sites[-1])] + 1
        N_from_center_assignment_list[int(intergen_sites[-1] + 2)] = N_from_center_assignment_list[int(intergen_sites[0])] + 1

    for igs in range(len(intergen_sites) - 1):
        inbetween_sites = np.arange(intergen_sites[igs], intergen_sites[igs + 1] + 1)
        if len(inbetween_sites) == 3:  # On side
            N_from_center_assignment_list[int(inbetween_sites[1])] = N_from_center_assignment_list[int(inbetween_sites[0])] + 1
        elif len(inbetween_sites) == 4:  # On corner
            N_from_center_assignment_list[int(inbetween_sites[1])] = N_from_center_assignment_list[int(inbetween_sites[0])] + 1
            N_from_center_assignment_list[int(inbetween_sites[2])] = N_from_center_assignment_list[int(inbetween_sites[-1])] + 1
    return (N_from_center_assignment_list)

def N_from_center_assignment_honeycomb(num_levels):
    N_from_center_assignment_list = np.zeros(int(honeycomb_points(num_levels)[1]))
    points_connected_to_next_level = np.arange(0,6)
    for n in range(2, num_levels+1):
        points_connected_to_prev_level = connected_to_prev_gen_points(n)
        N_from_center_assignment_list[points_connected_to_prev_level] = N_from_center_assignment_list[points_connected_to_next_level] + 1
        N_from_center_assignment_list = fill_inbetween_intergen_sites(n, N_from_center_assignment_list)
        points_connected_to_next_level = connected_to_next_gen_points(n)
    return(N_from_center_assignment_list)

def N_from_center_tight_bind_ham(nl, tight_bind_ham_sublattice_basis, N_from_center_assignment_list):
    t = 1
    num_sites = np.size(tight_bind_ham_sublattice_basis, 0)
    asites, bsites = site_assignment_honeycomb(nl, honeycomb_lattice_periodic_boundary(nl))
    asites = np.array([int(i) for i in asites])
    bsites = np.array([int(i) for i in bsites])
    N_from_center_assignment_list = N_from_center_assignment_list[np.concatenate((asites, bsites))]
    phase_mod_pos = np.exp(N_from_center_assignment_list**2)
    phase_mod_neg = np.exp(-1*N_from_center_assignment_list**2)
    conn_locs = np.where(tight_bind_ham_sublattice_basis[:int(num_sites/2), :] != 0)
    tight_bind_ham_sublattice_basis[conn_locs] = phase_mod_pos[conn_locs[0]]*t*phase_mod_neg[conn_locs[1]]
    tight_bind_ham_sublattice_basis[int(num_sites/2):, :int(num_sites/2)] = np.transpose(tight_bind_ham_sublattice_basis[:int(num_sites/2), int(num_sites/2):])
    return(tight_bind_ham_sublattice_basis)
