# Import dependencies
import numpy as np
from Axial_Magnetic_Field.Hubbard import spin_doubled_nonhermitian_axialfield_hyperbolicq3 as hyperbolicq3_strained_hamiltonian
from Fundamental.Eigenvalue_Degeneracy_Fix import small_chaos as small_disorder
from Fundamental.Hubbard import SCHF_onsite_hubbard_disorderalreadyadded_saverawoutput as SCHF_calculate
from Self_Consistent_Hartree_Fock.SCHF_C3Symmetry import SCHF_onsite_hubbard_disorderalreadyadded_saverawoutput_C3Average as SCHF_Calculate_Symmetrize

# Define system parameters
pval = 10
nval = 4
alpha_NH = 0.4
small_disorder_strength = 10**(-3)
hubbard_U = 0.5
qaxial = 0.1
symmetrize = False

# Form the spin-polarized tight-binding Hamiltonian and add small disorder for stabilization
hamiltonian = hyperbolicq3_strained_hamiltonian(pval, nval, alpha_NH, qaxial)
hamiltonian_withDisorder = hamiltonian + small_disorder(small_disorder_strength, hamiltonian.shape[0])

# Perform the SCHF computation
if symmetrize:
    raw_data, system_data = SCHF_Calculate_Symmetrize(pval, 3, nval, hamiltonian_withDisorder,
                                                      U_list=np.array([hubbard_U]),
                                                      initial_guess=0.1, tolerance=0.001)

else:
    raw_data, system_data = SCHF_calculate(pval, 3, nval, hamiltonian_withDisorder,
                                           U_list=np.array([hubbard_U]),
                                           initial_guess=0.1, tolerance=0.001)

# Here may want to save raw_data and system_data

# Above code can be generalized inside of for loops to compute multiple qaxial / multiple hubbard_U in one run
