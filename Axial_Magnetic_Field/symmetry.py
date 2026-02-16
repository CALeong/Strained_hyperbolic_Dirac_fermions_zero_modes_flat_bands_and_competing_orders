import numpy as np
from Fundamental.Honeycomb_Lattice import honeycomb_points, honeycomb_lattice
from Fundamental.NonHermitian_Hamiltonian import site_assignment_honeycomb
from Fundamental.Number_Points import points


############################################
def c3symmetry_site_label_transform_honeycomb(num_levels):
    points_per_level, total_num_points = honeycomb_points(num_levels)
    points_per_level = np.concatenate((np.array([0]), points_per_level))
    sites_on_level = []
    for i in range(len(points_per_level)):
        sites_on_level.append(np.arange(np.sum(points_per_level[:i]), np.sum(points_per_level[:i+1])))

    transformed_sites_on_level_one = []
    transformed_sites_on_level_two = []
    for sol in sites_on_level:
        fundamental_shift = int(len(sol) / 3)
        transformed_sites_on_level_one.append(np.roll(np.array(sol), fundamental_shift))
        transformed_sites_on_level_two.append(np.roll(np.array(sol), 2*fundamental_shift))

    return(transformed_sites_on_level_one, transformed_sites_on_level_two)

def return_c3symmetry_site_label_transform_basis_honeycomb(num_levels):
    transformed_sites_on_level_one, transformed_sites_on_level_two = c3symmetry_site_label_transform_honeycomb(num_levels)
    rotated_basis_one = np.array([], dtype=int)
    rotated_basis_two = np.array([], dtype=int)
    for tsolo in transformed_sites_on_level_one:
        for tsolov in tsolo:
            rotated_basis_one = np.append(rotated_basis_one, int(tsolov))
    for tsolt in transformed_sites_on_level_two:
        for tsoltv in tsolt:
            rotated_basis_two = np.append(rotated_basis_two, int(tsoltv))
    return(rotated_basis_one, rotated_basis_two)

# def c3symmetry_site_label_transform_singlesite_honeycomb(num_levels, site_index):
#     rotated_basis_one, rotated_basis_two = return_c3symmetry_site_label_transform_basis_honeycomb(num_levels)
#     return(rotated_basis_one[site_index], rotated_basis_two[site_index])
#
# def c3symmetry_site_label_transform_sequenceofsites_honeycomb(num_levels, site_indices):
#     rotated_basis_one, rotated_basis_two = return_c3symmetry_site_label_transform_basis_honeycomb(num_levels)
#     return(rotated_basis_one[site_indices], rotated_basis_two[site_indices])

############################################
def c3symmetry_site_label_transform_sequenceofsites_hyperbolicq3(p, num_levels):
    points_per_level, total_num_points = points(p, 3, num_levels)
    points_per_level = np.concatenate((np.array([0]), points_per_level))
    sites_on_level = []
    for i in range(len(points_per_level)):
        sites_on_level.append(np.arange(np.sum(points_per_level[:i]), np.sum(points_per_level[:i+1])))
    rotated_basis_mat = np.zeros((int((p/2) - 1), int(total_num_points)), dtype=int)
    for r in range(1, int((p / 2))):
        basis_entry = np.array([])
        for sol in sites_on_level:
            fundamental_shift = int(r * 2 * len(sol) / p)
            basis_entry = np.append(basis_entry, np.roll(sol, fundamental_shift))
        rotated_basis_mat[r-1, :] = basis_entry
    return(rotated_basis_mat)


############################################
# def c3symmetry_site_label_transform_hyperbolicq3(p, num_levels):
#     points_per_level, total_num_points = points(p, 3, num_levels)
#     points_per_level = np.concatenate((np.array([0]), points_per_level))
#     sites_on_level = []
#     for i in range(len(points_per_level)):
#         sites_on_level.append(np.arange(np.sum(points_per_level[:i]), np.sum(points_per_level[:i+1])))
#
#     transformed_sites_on_level = {}
#     for soli in range(len(sites_on_level)):
#         fundamental_shift = int(len(sites_on_level[soli]) / p)
#         for ri in range(p-1):
#             transformed_sites_on_level['gen{}rot{}'.format(soli, ri)] = np.roll(sites_on_level[soli], (ri+1) * fundamental_shift)
#
#     return(transformed_sites_on_level)
#
# def return_c3symmetry_site_label_transform_basis_hyperbolicq3(p, num_levels):
#     transformed_sites_on_level = c3symmetry_site_label_transform_hyperbolicq3(p, num_levels)
#     rotated_basis = {}
#     for n in range(num_levels):
#         rotated_basis['gen{}'.format(n)] = np.array([])
#         for ri in range(p-1):
#
#     return(rotated_basis_one, rotated_basis_two)
#
# def c3symmetry_site_label_transform_sequenceofsites_hyperbolicq3(p, num_levels, site_indices):
#     rotated_basis_one, rotated_basis_two = return_c3symmetry_site_label_transform_basis_hyperbolicq3(p, num_levels)
#     return(rotated_basis_one[site_indices], rotated_basis_two[site_indices])
