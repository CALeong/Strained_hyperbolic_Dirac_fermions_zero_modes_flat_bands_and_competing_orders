import numpy as np
from Fundamental.Number_Points import points as hyperbolic_points
from Fundamental.NonHermitian_Hamiltonian import site_assignment
from Fundamental.Hamiltonian import H0 as hyperbolic_h0
from Self_Consistent_Hartree_Fock.Self_Consistent_Hartree_Fock import hartreefock_hamiltonian_addition_NonHermitian
import scipy
import pickle

def hyperbolic_generation_trend(p,q,n,raw_data_mat,V_value):
    vdata = raw_data_mat[np.where(raw_data_mat[:,0] == V_value)[0], 1:][0]
    points_per_level, total_num_points = hyperbolic_points(p,q,n)
    ham = hyperbolic_h0(p,q,n)
    asites, bsites = site_assignment(p, q, n, ham)
    sublattice_basis = np.concatenate((asites, bsites))
    deltacdw_eachgen = np.array([])
    for level in range(n):
        if level == 0:
            level_sites_ogbasis_index = np.arange(points_per_level[0])
        else:
            level_sites_ogbasis_index = np.arange(np.sum(points_per_level[:level]), np.sum(points_per_level[:level+1]))
        level_sites_sublattice_basis_index = np.array([],dtype=int)
        for i in level_sites_ogbasis_index:
            level_sites_sublattice_basis_index = np.append(level_sites_sublattice_basis_index, int(np.where(i == sublattice_basis)[0]))
        deltacdw_eachgen = np.append(deltacdw_eachgen, np.average(np.abs(vdata[level_sites_sublattice_basis_index])))
    return(deltacdw_eachgen)

def hyperbolic_generation_trend_SDW(p,q,n,raw_data_mat,V_value):
    vdata = raw_data_mat[np.where(raw_data_mat[:,0] == V_value)[0], 1:][0]
    points_per_level, total_num_points = hyperbolic_points(p,q,n)
    ham = hyperbolic_h0(p,q,n)
    asites, bsites = site_assignment(p, q, n, ham)
    sublattice_basis = np.concatenate((asites, bsites))
    deltacdw_eachgen = np.array([])
    for level in range(n):
        if level == 0:
            level_sites_ogbasis_index = np.arange(points_per_level[0])
        else:
            level_sites_ogbasis_index = np.arange(np.sum(points_per_level[:level]), np.sum(points_per_level[:level+1]))
        level_sites_sublattice_basis_index = np.array([],dtype=int)
        for i in level_sites_ogbasis_index:
            level_sites_sublattice_basis_index = np.append(level_sites_sublattice_basis_index, int(np.where(i == sublattice_basis)[0]))
        #Two extra line needed for SDW case to account for the spin doubling
        other_spin_indices = np.array([int(m) for m in level_sites_sublattice_basis_index + total_num_points])
        level_sites_sublattice_basis_index = np.concatenate((level_sites_sublattice_basis_index, other_spin_indices))
        deltacdw_eachgen = np.append(deltacdw_eachgen, np.average(np.abs(vdata[level_sites_sublattice_basis_index])))
    return(deltacdw_eachgen)

def hyperbolic_generation_trend_manual(p,q,n,raw_data_mat,raw_file_row):
    vdata = raw_data_mat[raw_file_row, 1:]
    points_per_level, total_num_points = hyperbolic_points(p,q,n)
    ham = hyperbolic_h0(p,q,n)
    asites, bsites = site_assignment(p, q, n, ham)
    sublattice_basis = np.concatenate((asites, bsites))
    deltacdw_eachgen = np.array([])
    for level in range(n):
        if level == 0:
            level_sites_ogbasis_index = np.arange(points_per_level[0])
        else:
            level_sites_ogbasis_index = np.arange(np.sum(points_per_level[:level]), np.sum(points_per_level[:level+1]))
        level_sites_sublattice_basis_index = np.array([],dtype=int)
        for i in level_sites_ogbasis_index:
            level_sites_sublattice_basis_index = np.append(level_sites_sublattice_basis_index, int(np.where(i == sublattice_basis)[0]))
        deltacdw_eachgen = np.append(deltacdw_eachgen, np.average(np.abs(vdata[level_sites_sublattice_basis_index])))
    return(deltacdw_eachgen)

def save_eigenvalues_eigenvec_specificv(p,q,n,a,ham,raw_data_mat,V_value,savedir):
    ham_withv = hartreefock_hamiltonian_addition_NonHermitian(p,q,n,ham,raw_data_mat[np.where(raw_data_mat[:,0] == V_value)[0], 1:][0],V_value)
    eigenvalues, lefteigenvectors, righteigenvectors = scipy.linalg.eig(ham_withv, left=True, right=True)
    np.save(savedir + '/' + 'p{}q{}n{}a{}v{}_eigenvalues'.format(p,q,n,a,V_value), eigenvalues)
    np.save(savedir + '/' + 'p{}q{}n{}a{}v{}_lefteigenvectors'.format(p, q, n, a, V_value), lefteigenvectors)
    np.save(savedir + '/' + 'p{}q{}n{}a{}v{}_righteigenvectors'.format(p, q, n, a, V_value), righteigenvectors)

def local_density_states(p,q,n,a,eigenvalues, ls_eigenvec_mat, rs_eigenvec_mat, savedir):
    ldos = {}
    ldos['eigenvalues'] = eigenvalues
    points_per_level, total_num_points = hyperbolic_points(p, q, n)
    ham = hyperbolic_h0(p, q, n)
    asites, bsites = site_assignment(p, q, n, ham)
    sublattice_basis = np.concatenate((asites, bsites))
    for level in range(n):
        if level == 0:
            level_sites_ogbasis_index = np.arange(points_per_level[0])
        else:
            level_sites_ogbasis_index = np.arange(np.sum(points_per_level[:level]), np.sum(points_per_level[:level+1]))

        level_sites_sublattice_basis_index = np.array([], dtype=int)
        for i in level_sites_ogbasis_index:
            level_sites_sublattice_basis_index = np.append(level_sites_sublattice_basis_index, int(np.where(i == sublattice_basis)[0]))

        ldos_values = np.array([])
        for col in range(np.size(ls_eigenvec_mat, 1)):
            ldos_values = np.append(ldos_values, np.sum(ls_eigenvec_mat[level_sites_sublattice_basis_index,col]*rs_eigenvec_mat[level_sites_sublattice_basis_index,col])/len(level_sites_sublattice_basis_index))
        ldos['n{}_ldos'.format(level)] = ldos_values

    with open(savedir + '_ldos_dict.pkl', 'wb') as f:
        pickle.dump(ldos, f)
    return(ldos)

def local_density_states_SDW(p,q,n,a,eigenvalues, ls_eigenvec_mat, rs_eigenvec_mat, savedir):
    ldos = {}
    ldos['eigenvalues'] = eigenvalues
    points_per_level, total_num_points = hyperbolic_points(p, q, n)
    ham = hyperbolic_h0(p, q, n)
    asites, bsites = site_assignment(p, q, n, ham)
    sublattice_basis = np.concatenate((asites, bsites))
    for level in range(n):
        if level == 0:
            level_sites_ogbasis_index = np.arange(points_per_level[0])
        else:
            level_sites_ogbasis_index = np.arange(np.sum(points_per_level[:level]), np.sum(points_per_level[:level+1]))

        level_sites_sublattice_basis_index = np.array([], dtype=int)
        for i in level_sites_ogbasis_index:
            level_sites_sublattice_basis_index = np.append(level_sites_sublattice_basis_index, int(np.where(i == sublattice_basis)[0]))

        # Two extra line needed for SDW case to account for the spin doubling
        other_spin_indices = np.array([int(m) for m in level_sites_sublattice_basis_index + total_num_points])
        level_sites_sublattice_basis_index = np.concatenate((level_sites_sublattice_basis_index, other_spin_indices))

        ldos_values = np.array([])
        for col in range(np.size(ls_eigenvec_mat, 1)):
            ldos_values = np.append(ldos_values, np.sum(ls_eigenvec_mat[level_sites_sublattice_basis_index,col]*rs_eigenvec_mat[level_sites_sublattice_basis_index,col])/len(level_sites_sublattice_basis_index))
        ldos['n{}_ldos'.format(level)] = ldos_values

    with open(savedir + '_ldos_dict.pkl', 'wb') as f:
        pickle.dump(ldos, f)
    return(ldos)

def save_ldos_astxt_from_pkldict(pkl_dict, savename, savedir):
    keys = pkl_dict.keys()
    ldos_keys = keys
    for lk in ldos_keys:
        data = pkl_dict[lk]
        np.savetxt(savedir + '/' + savename + '_' + lk + '.txt', data)


from Fundamental.Hamiltonian_PeierlsSubstitution import Number_Plaquets
def get_lower_gen_amag(p, q, currentgen, alphamag_highergen):
    tot_num_plaquets_higher_gen = Number_Plaquets(p, q, currentgen + 1)[1]
    tot_num_plaquets_lower_gen = Number_Plaquets(p, q, currentgen)[1]
    return(alphamag_highergen*(tot_num_plaquets_higher_gen/tot_num_plaquets_lower_gen))

def get_higher_gen_amag(p, q, currentgen, alphamag_lowergen):
    tot_num_plaquets_higher_gen = Number_Plaquets(p, q, currentgen)[1]
    tot_num_plaquets_lower_gen = Number_Plaquets(p, q, currentgen-1)[1]
    return (alphamag_lowergen * (tot_num_plaquets_lower_gen / tot_num_plaquets_higher_gen))
