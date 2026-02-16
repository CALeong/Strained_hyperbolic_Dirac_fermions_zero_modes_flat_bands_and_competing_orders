import numpy as np
from Fundamental.Honeycomb_Lattice import honeycomb_lattice_periodic_boundary, honeycomb_lattice
from Fundamental.NonHermitian_Hamiltonian import NonHermitian_Honeycomb, site_assignment_honeycomb
from Axial_Magnetic_Field.honeycomb import N_from_center_assignment_honeycomb
from Self_Consistent_Hartree_Fock.Self_Consistent_Hartree_Fock import site_assignment
from Fundamental.General_Hamiltonian import general_q3_hamiltonian
from Axial_Magnetic_Field.hyperbolicq3 import N_from_center_assignment_hyperbolic_q3
from Fundamental.NonHermitian_Hamiltonian import NonHermitian_Hamiltonian

def N_from_center_tight_bind_ham_honeycomb_Reversed(nl, tight_bind_ham_sublattice_basis, N_from_center_assignment_list):
    t = 1
    num_sites = np.size(tight_bind_ham_sublattice_basis, 0)
    asites, bsites = site_assignment_honeycomb(nl, honeycomb_lattice_periodic_boundary(nl))
    asites = np.array([int(i) for i in asites])
    bsites = np.array([int(i) for i in bsites])
    N_from_center_assignment_list = N_from_center_assignment_list[np.concatenate((asites, bsites))]
    #Only difference is here where phae_mod pos and neg are switched
    phase_mod_neg = np.exp(N_from_center_assignment_list**2)
    phase_mod_pos = np.exp(-1*N_from_center_assignment_list**2)
    conn_locs = np.where(tight_bind_ham_sublattice_basis[:int(num_sites/2), :] != 0)
    tight_bind_ham_sublattice_basis[conn_locs] = phase_mod_pos[conn_locs[0]]*t*phase_mod_neg[conn_locs[1]]
    tight_bind_ham_sublattice_basis[int(num_sites/2):, :int(num_sites/2)] = np.transpose(tight_bind_ham_sublattice_basis[:int(num_sites/2), int(num_sites/2):])
    return(tight_bind_ham_sublattice_basis)

def N_from_center_tight_bind_ham_nonhermitian_honeycomb_Reversed(num_levels, alpha, axial_strength):
    t=1
    ham = honeycomb_lattice(num_levels)
    num_sites = np.size(ham, 0)
    nfromcenterlist = N_from_center_assignment_honeycomb(num_levels)*axial_strength
    tight_bind_sublattice_basis = NonHermitian_Honeycomb(num_levels, t, alpha)
    #Only difference is now used N_from_center_tight_bind_ham_honeycomb_Reversed
    ham_with_axial = N_from_center_tight_bind_ham_honeycomb_Reversed(num_levels, tight_bind_sublattice_basis, nfromcenterlist)
    ham_with_axial[:int(num_sites/2), int(num_sites/2):] = (1+alpha)*ham_with_axial[:int(num_sites/2), int(num_sites/2):]
    ham_with_axial[int(num_sites/2):, :int(num_sites/2)] = (1-alpha)*ham_with_axial[int(num_sites/2):, :int(num_sites/2)]
    return(ham_with_axial)

def N_from_center_tight_bind_ham_hyperbolic_q3_Reversed(p, num_levels, tight_bind_ham_sublattice_basis, N_from_center_assignment_list):
    t = 1
    num_sites = np.size(tight_bind_ham_sublattice_basis, 0)
    asites, bsites = site_assignment(p, 3, num_levels, general_q3_hamiltonian(p, num_levels).toarray())
    asites = np.array([int(i) for i in asites])
    bsites = np.array([int(i) for i in bsites])
    N_from_center_assignment_list = N_from_center_assignment_list[np.concatenate((asites, bsites))]
    #Only difference is here where phae_mod pos and neg are switched
    phase_mod_pos = np.exp(-1 * N_from_center_assignment_list ** 2)
    phase_mod_neg = np.exp(N_from_center_assignment_list ** 2)
    conn_locs = np.where(tight_bind_ham_sublattice_basis[:int(num_sites / 2), :] != 0)
    tight_bind_ham_sublattice_basis[conn_locs] = phase_mod_pos[conn_locs[0]] * t * phase_mod_neg[conn_locs[1]]
    tight_bind_ham_sublattice_basis[int(num_sites / 2):, :int(num_sites / 2)] = np.transpose(tight_bind_ham_sublattice_basis[:int(num_sites / 2), int(num_sites / 2):])
    return (tight_bind_ham_sublattice_basis)

def N_from_center_tight_bind_ham_nonhermitian_hyperbolic_q3_Reversed(p, num_levels, alpha, axial_strength):
    t=1
    ham = general_q3_hamiltonian(p, num_levels).toarray()
    num_sites = np.size(ham, 0)
    nfromcenterlist = N_from_center_assignment_hyperbolic_q3(ham, p, num_levels)*axial_strength
    tight_bind_sublattice_basis = NonHermitian_Hamiltonian(p, 3, num_levels, alpha, t)
    #Only difference is now used N_from_center_tight_bind_ham_hyperbolic_q3_Reversed
    ham_with_axial = N_from_center_tight_bind_ham_hyperbolic_q3_Reversed(p, num_levels, tight_bind_sublattice_basis, nfromcenterlist)
    ham_with_axial[:int(num_sites/2), int(num_sites/2):] = (1+alpha)*ham_with_axial[:int(num_sites/2), int(num_sites/2):]
    ham_with_axial[int(num_sites/2):, :int(num_sites/2)] = (1-alpha)*ham_with_axial[int(num_sites/2):, :int(num_sites/2)]
    return(ham_with_axial)
