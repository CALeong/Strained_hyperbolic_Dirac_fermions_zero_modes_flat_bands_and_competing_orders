# Import dependencies
import numpy as np
from Axial_Magnetic_Field.NonHermitian import N_from_center_tight_bind_ham_nonhermitian_honeycomb as honeycomb_strained_hamiltonian
from Axial_Magnetic_Field.reverse import N_from_center_tight_bind_ham_nonhermitian_honeycomb_Reversed as honeycomb_strained_hamiltonian_reversed
from Fundamental.Eigenvalue_Degeneracy_Fix import small_chaos as small_disorder
from Fundamental.Haldane import selfconsist_hartreefock_NonHermitian_SublatticeBasis_DisorderAlreadyAdded_SaveRawOutputs_MoreEfficient_honeycomb_C3Average_DiscardReal as SCHF_calculate
from Fundamental.Haldane import selfconsist_hartreefock_Hermitian_SublatticeBasis_DisorderAlreadyAdded_SaveRawOutputs_MoreEfficient_honeycomb_C3Average_DiscardReal as SCHF_calculate_Hermitian

# Define system parameters
nval = 20
alpha_NH = 0.4
small_disorder_strength = 10**(-3)
coulomb_V2 = 0.3
qaxial = 0.1
schf_tolerance = 0.001

# Form the spin-polarized tight-binding Hamiltonian and add small disorder for stabilization
hamiltonian = honeycomb_strained_hamiltonian(nval, alpha_NH, qaxial)

if alpha_NH != 0.0:
    hamiltonian_withDisorder = hamiltonian + small_disorder(small_disorder_strength, hamiltonian.shape[0])

    # Perform the SCHF computation
    results = SCHF_calculate(nval, hamiltonian_withDisorder,
                             initial_guess=0.1, tolerance=schf_tolerance,
                             hfcoeff_list=np.array([coulomb_V2]))

    eta_values_dict, total_mag_flux, max_real_comp, imag_ev_bounds, smallest_real_ev = results
else:
    # Perform the SCHF computation
    results = SCHF_calculate_Hermitian(nval, hamiltonian,
                                       initial_guess=0.1, tolerance=schf_tolerance,
                                       hfcoeff_list=np.array([coulomb_V2]))

    eta_values_dict, total_mag_flux, max_real_comp = results

# Repeat above but with reversed axial magnetic field
hamiltonian = honeycomb_strained_hamiltonian_reversed(nval, alpha_NH, qaxial)

if alpha_NH != 0.0:
    hamiltonian_withDisorder = hamiltonian + small_disorder(small_disorder_strength, hamiltonian.shape[0])

    results_reverse = SCHF_calculate(nval, hamiltonian_withDisorder,
                                     initial_guess=0.1, tolerance=schf_tolerance,
                                     hfcoeff_list=np.array([coulomb_V2]))

    eta_values_dict_rev, total_mag_flux_rev, max_real_comp_rev, imag_ev_bounds_rev, smallest_real_ev_rev = results_reverse
else:
    results_reverse = SCHF_calculate_Hermitian(nval, hamiltonian,
                                               initial_guess=0.1, tolerance=schf_tolerance,
                                               hfcoeff_list=np.array([coulomb_V2]))

    eta_values_dict_rev, total_mag_flux_rev, max_real_comp_rev = results_reverse

# Here may want to save the above results

# Above code can be generalized inside of for loops to compute multiple qaxial / multiple coulomb_V in one run

# check total magnetic flux from adding both configurations is negligible
print('Maximum magnetic flux: {}'.format(np.max(np.abs(total_mag_flux + total_mag_flux_rev))))

# check that maximum real component of NNN correlations remained small
print('Maximum real component of eta: {}'.format(np.max(np.abs(max_real_comp))))
print('Maximum real component of eta for reversed axial field: {}'.format(np.max(np.abs(max_real_comp_rev))))

if alpha_NH != 0.0:
    # check that maximum imaginary eigval components remained small compared to the smallest real eigenvalue components
    print('Max imaginary eigval component / Min real eigval component: {}'
          .format(np.max(np.max(np.abs(imag_ev_bounds), axis=0) / smallest_real_ev)))
    print('Max imaginary eigval component / Min real eigval component for reverse axial field: {}'
          .format(np.max(np.max(np.abs(imag_ev_bounds), axis=0) / smallest_real_ev)))

# compute final Haldane order parameter for given coulomb_V2 value
delta_HO = np.average(np.abs(np.imag(eta_values_dict['V2={}'.format(coulomb_V2)][2, :])))
delta_HO_rev = np.average(np.abs(np.imag(eta_values_dict_rev['V2={}'.format(coulomb_V2)][2, :])))

final_delta_HO = (delta_HO + delta_HO_rev) / 2

