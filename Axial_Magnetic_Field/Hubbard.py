from Axial_Magnetic_Field.NonHermitian import N_from_center_tight_bind_ham_nonhermitian_hyperbolic_q3
from Axial_Magnetic_Field.NonHermitian import N_from_center_tight_bind_ham_nonhermitian_honeycomb
import numpy as np

def spin_doubled_nonhermitian_axialfield_hyperbolicq3(p, num_levels, alpha, axial_strength):
    ham = N_from_center_tight_bind_ham_nonhermitian_hyperbolic_q3(p, num_levels, alpha, axial_strength)
    zeroblock = np.zeros((np.size(ham, 0), np.size(ham, 1)))
    return(np.block([[ham, zeroblock],[zeroblock, ham]]))

def spin_doubled_nonhermitian_axialfield_honeycomb(num_levels, alpha, axial_strength):
    ham = N_from_center_tight_bind_ham_nonhermitian_honeycomb(num_levels, alpha, axial_strength)
    zeroblock = np.zeros((np.size(ham, 0), np.size(ham, 1)))
    return(np.block([[ham, zeroblock],[zeroblock, ham]]))

