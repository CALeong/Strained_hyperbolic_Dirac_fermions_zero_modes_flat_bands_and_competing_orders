# Import dependencies
import numpy as np
from Axial_Magnetic_Field.NonHermitian import N_from_center_tight_bind_ham_nonhermitian_honeycomb as honeycomb_strained_hamiltonian
from Fundamental.Eigenvalue_Degeneracy_Fix import small_chaos as small_disorder
from Self_Consistent_Hartree_Fock.Self_Consistent_Hartree_Fock import selfconsist_hartreefock_NonHermitian_Honeycomb_PBC_DisorderAlreadyAdded_SaveRawOutputs as SCHF_calculate

# Define system parameters
nval = 20
alpha_NH = 0.4
small_disorder_strength = 10**(-3)
coulomb_V = 0.3
qaxial = 0.1

# Form the spin-polarized tight-binding Hamiltonian and add small disorder for stabilization
hamiltonian = honeycomb_strained_hamiltonian(nval, alpha_NH, qaxial)
hamiltonian_withDisorder = hamiltonian + small_disorder(small_disorder_strength, hamiltonian.shape[0])

# Perform the SCHF computation
raw_data, system_data, _ = SCHF_calculate(nval, hamiltonian_withDisorder,
                                          initial_guess=0.1, tolerance=0.001,
                                          hfcoeff_list=np.array([coulomb_V]))

# Here may want to save raw_data and system_data

# Above code can be generalized inside of for loops to compute multiple qaxial / multiple coulomb_V in one run



