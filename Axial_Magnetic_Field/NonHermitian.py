from Axial_Magnetic_Field.hyperbolicq3 import N_from_center_assignment_hyperbolic_q3
from Axial_Magnetic_Field.hyperbolicq3 import N_from_center_tight_bind_ham_hyperbolic_q3
from Axial_Magnetic_Field.honeycomb import N_from_center_assignment_honeycomb
from Axial_Magnetic_Field.honeycomb import N_from_center_tight_bind_ham as N_from_center_tight_bind_ham_honeycomb
from Fundamental.General_Hamiltonian import general_q3_hamiltonian
from Fundamental.NonHermitian_Hamiltonian import NonHermitian_Hamiltonian, NonHermitian_Honeycomb
import numpy as np
from Fundamental.Honeycomb_Lattice import honeycomb_lattice

def N_from_center_tight_bind_ham_nonhermitian_hyperbolic_q3(p, num_levels, alpha, axial_strength):
    t=1
    ham = general_q3_hamiltonian(p, num_levels).toarray()
    num_sites = np.size(ham, 0)
    nfromcenterlist = N_from_center_assignment_hyperbolic_q3(ham, p, num_levels)*axial_strength
    tight_bind_sublattice_basis = NonHermitian_Hamiltonian(p, 3, num_levels, alpha, t)
    ham_with_axial = N_from_center_tight_bind_ham_hyperbolic_q3(p, num_levels, tight_bind_sublattice_basis, nfromcenterlist)
    ham_with_axial[:int(num_sites/2), int(num_sites/2):] = (1+alpha)*ham_with_axial[:int(num_sites/2), int(num_sites/2):]
    ham_with_axial[int(num_sites/2):, :int(num_sites/2)] = (1-alpha)*ham_with_axial[int(num_sites/2):, :int(num_sites/2)]
    return(ham_with_axial)

def N_from_center_tight_bind_ham_nonhermitian_honeycomb(num_levels, alpha, axial_strength):
    t=1
    ham = honeycomb_lattice(num_levels)
    num_sites = np.size(ham, 0)
    nfromcenterlist = N_from_center_assignment_honeycomb(num_levels)*axial_strength
    tight_bind_sublattice_basis = NonHermitian_Honeycomb(num_levels, t, alpha)
    ham_with_axial = N_from_center_tight_bind_ham_honeycomb(num_levels, tight_bind_sublattice_basis, nfromcenterlist)
    ham_with_axial[:int(num_sites/2), int(num_sites/2):] = (1+alpha)*ham_with_axial[:int(num_sites/2), int(num_sites/2):]
    ham_with_axial[int(num_sites/2):, :int(num_sites/2)] = (1-alpha)*ham_with_axial[int(num_sites/2):, :int(num_sites/2)]
    return(ham_with_axial)


