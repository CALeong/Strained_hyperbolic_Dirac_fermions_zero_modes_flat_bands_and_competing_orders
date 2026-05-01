# Import dependencies
import numpy as np
from Fundamental.General_Hamiltonian_Strain import general_q3_hamiltonian_with_strain
from KPM.KPM import moments_ADOS_general as KPM_compute_ADOS
from KPM.measure import rescale_operator_unity_window
from KPM.measure import calculate_ADOS_from_moments
from KPM.parallel_compute import random_vectors_arr_generate
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Define system parameters
pval = 10
nval = 9
qaxial = 0.015
Nm = 16384

# Generate hamiltonian
hamiltonian = general_q3_hamiltonian_with_strain(pval, nval, qaxial)

# Rescale Hamiltonian in preparation for KPM
hamiltonian = rescale_operator_unity_window(hamiltonian, eigval_min=-3.12, eigval_max=3.12)

# Generate random vectors for stochastic trace evaluation used in KPM
random_vectors_array = random_vectors_arr_generate(12, hamiltonian.shape[0])

# Perform KPM
computed_moments = KPM_compute_ADOS(hamiltonian, Nm, random_vectors_array)

# Since this computation is expensive, will likely want to save computed moments
# This can be done in many ways such as
# np.save("address to save to", computed_moments)

# Reconstruct ADOS from moments (with Jackson kernel)
energy_values = np.linspace(-3.12, 3.12, 21)
reconstructed_ados = calculate_ADOS_from_moments(computed_moments, energy_values,
                                                 eigval_min=-3.12, eigval_max=3.12)

# Plot of ADOS from reconstruction with interpolation smoothening
interpolation_func = make_interp_spline(energy_values, reconstructed_ados, k=3)
plt.plot(np.linspace(-3.12, 3.12, 51),
         interpolation_func(np.linspace(-3.12, 3.12, 51)))
plt.show()

# User may want to change how many energy values sampled for reconstructed ADOS
# as well as how many energy values to sample from interpolation function
