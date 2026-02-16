import numpy as np
from Fundamental.Number_Points import points
import scipy
from Fundamental.Biorthogonal import biortogonal_normalize
from Fundamental.Eigenvalue_Degeneracy_Fix import small_chaos
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from Fundamental.NonHermitian_Hamiltonian import site_assignment
import time
from Fundamental.Honeycomb_Lattice import honeycomb_points

# def site_assignment(p,q,num_levels,ham):
#
#     totalnum_points = points(p,q,num_levels)[1]
#
#     a_sites = np.array([])
#     b_sites = np.array([])
#     #Know sites of first gen; let first site be a_site then they just alternate
#     first_gen_sites = np.arange(points(p,q,num_levels)[0][0])
#     a_sites = np.append(a_sites, np.where(first_gen_sites%2 == 0))
#     b_sites = np.append(b_sites, np.where(first_gen_sites%2 == 1))
#
#     site_assign_progress_count = 0
#     while len(a_sites) != (totalnum_points/2) or len(b_sites) != (totalnum_points/2):
#         for i in a_sites:
#             b_sites = np.append(b_sites, np.where(ham[int(i)] != 0))
#             b_sites = np.unique(b_sites)
#         for i in b_sites:
#             a_sites = np.append(a_sites, np.where(ham[int(i)] != 0))
#             a_sites = np.unique(a_sites)
#         site_assign_progress_count += 1
#         print(site_assign_progress_count)
#     print('site assignment while loop done')
#     return(a_sites, b_sites)

def site_assignment_NonHermitian(p,q,num_levels):
    total_num_sites = points(p,q,num_levels)[1]
    a_sites = np.linspace(0,int(total_num_sites/2)-1,int(total_num_sites/2))
    b_sites = np.linspace(int(total_num_sites/2),int(total_num_sites)-1,int(total_num_sites/2))
    return(a_sites, b_sites)

def site_assignment_NonHermitian_Honeycomb(num_levels):
    from Fundamental.Honeycomb_Lattice import honeycomb_points
    total_num_sites = honeycomb_points(num_levels)[1]
    a_sites = np.linspace(0,int(total_num_sites/2)-1,int(total_num_sites/2))
    b_sites = np.linspace(int(total_num_sites/2),int(total_num_sites)-1,int(total_num_sites/2))
    return(a_sites, b_sites)

def self_consistent_check(new_result, prev_result, tolerance):
    check = np.where(np.abs(new_result-prev_result) > tolerance)
    if np.size(check[0],0) == 0:
        converge = True
    else:
        converge = False
    return(converge)

def hartreefock_hamiltonian_addition(p, q, num_levels, ham, deltas, hfcoeff):
    hf_addterm_diagvals = np.array([])
    for row in range(np.size(ham,0)):
        near_neighbor_indices = np.where(ham[row,:] != 0)[0]
        diag_vals = np.array([])
        for nni in near_neighbor_indices:
            diag_vals = np.append(diag_vals, deltas[nni])
        hf_addterm_diagvals = np.append(hf_addterm_diagvals, np.sum(diag_vals))
    hf_addterm_matrix = hfcoeff*np.diag(hf_addterm_diagvals)
    final_ham = ham + hf_addterm_matrix
    return(final_ham)
def hartreefock_hamiltonian_addition_NonHermitian(p, q, num_levels, ham, deltas, hfcoeff):
    hf_addterm_diagvals = np.array([])
    for row in range(np.size(ham,0)):
        near_neighbor_indices = np.where(ham[row,:] != 0)[0] #np.around because small disorder term confuses this code
        totalnum_points = points(p,q,num_levels)[1]
        if row < totalnum_points/2: #Need this because code is getting confused due to small disorder
            near_neighbor_indices = near_neighbor_indices[np.where(near_neighbor_indices >= totalnum_points/2)[0]]
        else:
            near_neighbor_indices = near_neighbor_indices[np.where(near_neighbor_indices < totalnum_points/2)[0]]
        diag_vals = np.array([])
        for nni in near_neighbor_indices:
            diag_vals = np.append(diag_vals, deltas[nni])
        hf_addterm_diagvals = np.append(hf_addterm_diagvals, np.sum(diag_vals))
    hf_addterm_matrix = hfcoeff*np.diag(hf_addterm_diagvals)
    final_ham = ham + hf_addterm_matrix
    return(final_ham)

def hartreefock_hamiltonian_addition_NonHermitian_Honeycomb(nl, ham, deltas, hfcoeff):
    hf_addterm_diagvals = np.array([])
    for row in range(np.size(ham,0)):
        near_neighbor_indices = np.where(ham[row,:] != 0)[0]
        totalnum_points = honeycomb_points(nl)[1]
        if row < totalnum_points/2: #Need this because code is getting confused due to small disorder
            near_neighbor_indices = near_neighbor_indices[np.where(near_neighbor_indices >= totalnum_points/2)[0]]
        else:
            near_neighbor_indices = near_neighbor_indices[np.where(near_neighbor_indices < totalnum_points/2)[0]]
        diag_vals = np.array([])
        for nni in near_neighbor_indices:
            diag_vals = np.append(diag_vals, deltas[nni])
        hf_addterm_diagvals = np.append(hf_addterm_diagvals, np.sum(diag_vals))
    hf_addterm_matrix = hfcoeff*np.diag(hf_addterm_diagvals)
    final_ham = ham + hf_addterm_matrix
    return(final_ham)

# def hartreefock_hamiltonian_addition_NonHermitian_Honeycomb(num_levels, ham, deltas, hfcoeff):
#     from Fundamental.Honeycomb_Lattice import honeycomb_points
#     hf_addterm_diagvals = np.array([])
#     for row in range(np.size(ham,0)):
#         near_neighbor_indices = np.where(ham[row,:] != 0)[0] #np.around because small disorder term confuses this code
#         totalnum_points = honeycomb_points(num_levels)[1]
#         if row < totalnum_points/2: #Need this because code is getting confused due to small disorder
#             near_neighbor_indices = near_neighbor_indices[np.where(near_neighbor_indices >= totalnum_points/2)[0]]
#         else:
#             near_neighbor_indices = near_neighbor_indices[np.where(near_neighbor_indices < totalnum_points/2)[0]]
#         diag_vals = np.array([])
#         for nni in near_neighbor_indices:
#             diag_vals = np.append(diag_vals, deltas[nni])
#         hf_addterm_diagvals = np.append(hf_addterm_diagvals, np.sum(diag_vals))
#     hf_addterm_matrix = hfcoeff*np.diag(hf_addterm_diagvals)
#     final_ham = ham + hf_addterm_matrix
#     return(final_ham)

def selfconsist_hartreefock(p, q, num_levels, initial_ham, initial_guess, tolerance, hfcoeff_list):
    a_sites, b_sites = site_assignment(p, q, num_levels, initial_ham) #Label sites for two basis lattice
    deltas_list = np.array([]) #Holds final delta value for each hfcoeff
    deltas_center_list = np.array([])
    for hfcoeff in hfcoeff_list: #Go over all desired V values
        print(hfcoeff)
        converge_status = False
        deltas = np.array([])
        for d in range(np.size(initial_ham,0)):
            if np.any(a_sites == d):
                deltas = np.append(deltas, initial_guess)
            else:
                deltas = np.append(deltas, -initial_guess)
        # deltas = np.repeat(initial_guess, np.size(initial_ham, 0)) #Initial guess
        ham = hartreefock_hamiltonian_addition(p,q,num_levels,initial_ham,deltas, hfcoeff) #HF Hamiltonian with initial guess
        converge_progress_count = 0
        while converge_status == False:
            eners, eigvec_matrix = np.linalg.eigh(ham) #diagonalize ham to get energies and eigvecs
            negative_energy_indices = np.where(eners < 0)[0] #Find indices where E<0
            negative_energy_eigvec_matrix = eigvec_matrix[:,negative_energy_indices]
            # negative_energy_eigvec_matrix = np.empty((np.size(eigvec_matrix,0),1))
            # for i in negative_energy_indices:
            #     negative_energy_eigvec_matrix = np.concatenate((negative_energy_eigvec_matrix, eigvec_matrix[:,i]), axis=1)
            # negative_energy_eigvec_matrix = negative_energy_eigvec_matrix[:,1:]
            site_probs = np.array([])
            for row in range(np.size(negative_energy_eigvec_matrix,0)):
                row_vals = negative_energy_eigvec_matrix[row,:]
                mod_square = np.sum(row_vals*np.conjugate(row_vals))
                site_probs = np.append(site_probs, mod_square)
            prev_delta = deltas
            deltas = site_probs - 0.5
            converge_status = self_consistent_check(deltas,prev_delta,tolerance)
            ham = hartreefock_hamiltonian_addition(p,q,num_levels,initial_ham,deltas, hfcoeff)
            converge_progress_count += 1
            print(converge_progress_count, np.max(np.abs(deltas-prev_delta)))
        print('Convergence done')
        a_deltas = np.array([])
        b_deltas = np.array([])
        for d in range(len(deltas)):
            if np.any(a_sites == d):
                a_deltas = np.append(a_deltas, deltas[d])
            else:
                b_deltas = np.append(b_deltas, deltas[d])
        a_delta_avg = np.abs(np.average(a_deltas))
        b_delta_avg = np.abs(np.average(b_deltas))
        deltas_list = np.append(deltas_list, 0.5*(a_delta_avg+b_delta_avg))
        a_deltas_center = np.array([])
        b_deltas_center = np.array([])
        for i in range(p):
            if np.any(i == a_sites):
                a_deltas_center = np.append(a_deltas_center, deltas[i])
            else:
                b_deltas_center = np.append(b_deltas_center, deltas[i])
        a_deltas_center_avg = np.abs(np.average(a_deltas_center))
        b_deltas_center_avg = np.abs(np.average(b_deltas_center))
        deltas_center_list = np.append(deltas_center_list, 0.5*(a_deltas_center_avg+b_deltas_center_avg))
    return(deltas_list, deltas_center_list)

def selfconsist_hartreefock_NonHermitian(p, q, num_levels, initial_ham, initial_guess, tolerance, hfcoeff_list):
    initial_ham = initial_ham + small_chaos(10**(-3),points(p,q,num_levels)[1])
    a_sites, b_sites = site_assignment_NonHermitian(p, q, num_levels)  # Label sites for two basis lattice
    deltas_list = np.array([])  # Holds final delta value for each hfcoeff
    deltas_center_list = np.array([])
    for hfcoeff in hfcoeff_list:  # Go over all desired V values
        print(hfcoeff)
        converge_status = False
        deltas = np.array([])
        for d in range(np.size(initial_ham, 0)):
            if np.any(a_sites == d):
                deltas = np.append(deltas, initial_guess)
            else:
                deltas = np.append(deltas, -initial_guess)
        # deltas = np.repeat(initial_guess, np.size(initial_ham, 0)) #Initial guess
        ham = hartreefock_hamiltonian_addition_NonHermitian(p, q, num_levels, initial_ham, deltas, hfcoeff)  # HF Hamiltonian with initial guess
        converge_progress_count = 0
        while converge_status == False:
            start_time = time.time()
            # print('Diagonalization Start')
            eners, left_eigvec_matrix, right_eigvec_matrix = scipy.linalg.eig(ham,left=True,right=True)  # diagonalize ham to get energies and eigvecs
            # print('Diagonalization Done')
            left_eigvec_matrix = np.conj(left_eigvec_matrix)
            left_eigvec_matrix, right_eigvec_matrix = biortogonal_normalize(left_eigvec_matrix, right_eigvec_matrix)
            # print('Normalization Done')
            negative_energy_indices = np.where(eners < 0)  # Find indices where E<0
            left_negative_energy_eigvec_matrix = left_eigvec_matrix[:,negative_energy_indices[0]]
            right_negative_energy_eigvec_matrix = right_eigvec_matrix[:,negative_energy_indices[0]]
            # print('Getting filled states done')
            site_probs = np.array([])
            for row in range(np.size(left_negative_energy_eigvec_matrix, 0)):
                site_probs = np.append(site_probs, np.sum(left_negative_energy_eigvec_matrix[row,:]*right_negative_energy_eigvec_matrix[row,:]))
            # print('Calculating site probabilities done')
            prev_delta = deltas
            deltas = site_probs - 0.5
            converge_status = self_consistent_check(deltas, prev_delta, tolerance)
            # print('Getting convergence status done')
            ham = hartreefock_hamiltonian_addition_NonHermitian(p, q, num_levels, initial_ham, deltas, hfcoeff)
            # print('Getting new hamiltonian done')
            converge_progress_count += 1
            print(converge_progress_count, np.max(np.abs(deltas - prev_delta)), 'Time taken: {}'.format(time.time()-start_time))
        print('Convergence done')
        a_deltas = np.array([])
        b_deltas = np.array([])
        for d in range(len(deltas)):
            if np.any(a_sites == d):
                a_deltas = np.append(a_deltas, deltas[d])
            else:
                b_deltas = np.append(b_deltas, deltas[d])
        a_delta_avg = np.abs(np.average(a_deltas))
        b_delta_avg = np.abs(np.average(b_deltas))
        deltas_list = np.append(deltas_list, 0.5 * (a_delta_avg + b_delta_avg))
        a_deltas_center = np.array([])
        b_deltas_center = np.array([])
        a_site_counter = 0
        b_site_counter = 1
        for i in range(p):
            # print(i, i%2==0)
            if i%2 == 0:
                a_deltas_center = np.append(a_deltas_center, deltas[int(a_sites[i-1*a_site_counter])])
                print(a_sites[i-1*a_site_counter])
                a_site_counter += 1
            else:
                b_deltas_center = np.append(b_deltas_center, deltas[int(b_sites[i-1*b_site_counter])])
                print(b_sites[i-1*b_site_counter])
                b_site_counter += 1
        a_deltas_center_avg = np.abs(np.average(a_deltas_center))
        b_deltas_center_avg = np.abs(np.average(b_deltas_center))
        deltas_center_list = np.append(deltas_center_list, 0.5 * (a_deltas_center_avg + b_deltas_center_avg))
    return (deltas_list, deltas_center_list)

# def selfconsist_hartreefock_NonHermitian_PeierlsSub(p, q, num_levels, alpha, t, initial_guess, tolerance, hfcoeff_list, alphamag_list):
#     from Fundamental.NonHermitian_Hamiltonian import NonHermitian_PeierlsSub_Hamiltonian as nhpsh
#     disorder = small_chaos(10**(-3),points(p,q,num_levels)[1])
#     for alphamag in alphamag_list:
#         initial_ham = nhpsh(p,q,num_levels,alpha,t,alphamag) + disorder
#         a_sites, b_sites = site_assignment_NonHermitian(p, q, num_levels)  # Label sites for two basis lattice
#         deltas_final_list = np.empty((1,len(hfcoeff_list)))  # Holds final delta value for each hfcoeff
#         deltas_center_final_list = np.empty((1,len(hfcoeff_list)))
#
#         deltas_list = np.array([])
#         deltas_center_list = np.array([])
#
#         for hfcoeff in hfcoeff_list:  # Go over all desired V values
#             print(hfcoeff)
#             converge_status = False
#             deltas = np.array([])
#             for d in range(np.size(initial_ham, 0)):
#                 if np.any(a_sites == d):
#                     deltas = np.append(deltas, initial_guess)
#                 else:
#                     deltas = np.append(deltas, -initial_guess)
#             # deltas = np.repeat(initial_guess, np.size(initial_ham, 0)) #Initial guess
#             ham = hartreefock_hamiltonian_addition_NonHermitian(p, q, num_levels, initial_ham, deltas, hfcoeff)  # HF Hamiltonian with initial guess
#             converge_progress_count = 0
#             while converge_status == False:
#                 eners, left_eigvec_matrix, right_eigvec_matrix = scipy.linalg.eig(ham,left=True,right=True)  # diagonalize ham to get energies and eigvecs
#                 left_eigvec_matrix = np.conj(left_eigvec_matrix)
#                 left_eigvec_matrix, right_eigvec_matrix = biortogonal_normalize(left_eigvec_matrix, right_eigvec_matrix)
#                 negative_energy_indices = np.where(eners < 0)  # Find indices where E<0
#                 left_negative_energy_eigvec_matrix = left_eigvec_matrix[:,negative_energy_indices[0]]
#                 right_negative_energy_eigvec_matrix = right_eigvec_matrix[:,negative_energy_indices[0]]
#                 site_probs = np.array([])
#                 for row in range(np.size(left_negative_energy_eigvec_matrix, 0)):
#                     site_probs = np.append(site_probs, np.sum(left_negative_energy_eigvec_matrix[row,:]*right_negative_energy_eigvec_matrix[row,:]))
#                 prev_delta = deltas
#                 deltas = site_probs - 0.5
#                 converge_status = self_consistent_check(deltas, prev_delta, tolerance)
#                 ham = hartreefock_hamiltonian_addition_NonHermitian(p, q, num_levels, initial_ham, deltas, hfcoeff)
#                 converge_progress_count += 1
#                 print(converge_progress_count, np.max(np.abs(deltas - prev_delta)))
#             print('Convergence done')
#             a_deltas = np.array([])
#             b_deltas = np.array([])
#             for d in range(len(deltas)):
#                 if np.any(a_sites == d):
#                     a_deltas = np.append(a_deltas, deltas[d])
#                 else:
#                     b_deltas = np.append(b_deltas, deltas[d])
#             a_delta_avg = np.abs(np.average(a_deltas))
#             b_delta_avg = np.abs(np.average(b_deltas))
#             deltas_list = np.append(deltas_list, 0.5 * (a_delta_avg + b_delta_avg))
#             a_deltas_center = np.array([])
#             b_deltas_center = np.array([])
#             b_site_counter = 1
#             for i in range(p):
#                 if i%2 == 0:
#                     a_deltas_center = np.append(a_deltas_center, deltas[np.int(a_sites[i])])
#                 else:
#                     b_deltas_center = np.append(b_deltas_center, deltas[np.int(b_sites[i-1*b_site_counter])])
#                     print(b_sites[i-1*b_site_counter])
#                     b_site_counter += 1
#             a_deltas_center_avg = np.abs(np.average(a_deltas_center))
#             b_deltas_center_avg = np.abs(np.average(b_deltas_center))
#             deltas_center_list = np.append(deltas_center_list, 0.5 * (a_deltas_center_avg + b_deltas_center_avg))
#         deltas_final_list = np.vstack((deltas_final_list,deltas_list))
#         deltas_center_final_list = np.vstack((deltas_center_final_list,deltas_center_list))
#     deltas_final_list = deltas_final_list[1:,:]
#     deltas_center_final_list = deltas_center_final_list[1:,:]
#     return (deltas_final_list, deltas_center_final_list)

# def after_critical_point_region(v_values, delta_values, cutoff):
#     v_values = v_values[np.where(delta_values > cutoff)]
#     delta_values = delta_values[np.where(delta_values > cutoff)]
#     return(v_values, delta_values)

# def critical_value_curve_fit(v_values, delta_values, cutoff):
#     def match_func(v, vcrit, scale_factor):
#         return(scale_factor*np.sqrt(v-vcrit))
#     old_v_values = v_values
#     old_delta_values = delta_values
#     v_values, delta_values = after_critical_point_region(v_values, delta_values, cutoff)
#     vcrit, scale_factor = curve_fit(f=match_func, xdata=v_values, ydata=delta_values, p0=[v_values[0],0.36])[0]
#     print('Cutoff V was: {}'.format(v_values[0]))
#     print('V Critical = {}'.format(vcrit))
#     print('Scale Factor: {}'.format(scale_factor))
#     plt.scatter(old_v_values,old_delta_values,s=15)
#     # plt.scatter(v_values, delta_values)
#     plt.plot(v_values,match_func(v_values,vcrit,scale_factor))
#     plt.show()
#     return(vcrit)

# def selfconsist_hartreefock_NonHermitian_Honeycomb(num_levels, initial_ham, initial_guess, tolerance, hfcoeff_list):
#     from Fundamental.Honeycomb_Lattice import honeycomb_points
#     initial_ham = initial_ham + small_chaos(10**(-3),honeycomb_points(num_levels)[1])
#     a_sites, b_sites = site_assignment_NonHermitian_Honeycomb(num_levels)  # Label sites for two basis lattice
#     deltas_list = np.array([])  # Holds final delta value for each hfcoeff
#     deltas_center_list = np.array([])
#     for hfcoeff in hfcoeff_list:  # Go over all desired V values
#         print(hfcoeff)
#         converge_status = False
#         deltas = np.array([])
#         for d in range(np.size(initial_ham, 0)):
#             if np.any(a_sites == d):
#                 deltas = np.append(deltas, initial_guess)
#             else:
#                 deltas = np.append(deltas, -initial_guess)
#         # deltas = np.repeat(initial_guess, np.size(initial_ham, 0)) #Initial guess
#         ham = hartreefock_hamiltonian_addition_NonHermitian_Honeycomb(num_levels, initial_ham, deltas, hfcoeff)  # HF Hamiltonian with initial guess
#         converge_progress_count = 0
#         while converge_status == False:
#             eners, left_eigvec_matrix, right_eigvec_matrix = scipy.linalg.eig(ham,left=True,right=True)  # diagonalize ham to get energies and eigvecs
#             left_eigvec_matrix = np.conj(left_eigvec_matrix)
#             left_eigvec_matrix, right_eigvec_matrix = biortogonal_normalize(left_eigvec_matrix, right_eigvec_matrix)
#             negative_energy_indices = np.where(eners < 0)  # Find indices where E<0
#             left_negative_energy_eigvec_matrix = left_eigvec_matrix[:,negative_energy_indices[0]]
#             right_negative_energy_eigvec_matrix = right_eigvec_matrix[:,negative_energy_indices[0]]
#             site_probs = np.array([])
#             for row in range(np.size(left_negative_energy_eigvec_matrix, 0)):
#                 site_probs = np.append(site_probs, np.sum(left_negative_energy_eigvec_matrix[row,:]*right_negative_energy_eigvec_matrix[row,:]))
#             prev_delta = deltas
#             deltas = site_probs - 0.5
#             converge_status = self_consistent_check(deltas, prev_delta, tolerance)
#             ham = hartreefock_hamiltonian_addition_NonHermitian_Honeycomb(num_levels, initial_ham, deltas, hfcoeff)
#             converge_progress_count += 1
#             print(converge_progress_count, np.max(np.abs(deltas - prev_delta)))
#         print('Convergence done')
#         a_deltas = np.array([])
#         b_deltas = np.array([])
#         for d in range(len(deltas)):
#             if np.any(a_sites == d):
#                 a_deltas = np.append(a_deltas, deltas[d])
#             else:
#                 b_deltas = np.append(b_deltas, deltas[d])
#         a_delta_avg = np.abs(np.average(a_deltas))
#         b_delta_avg = np.abs(np.average(b_deltas))
#         deltas_list = np.append(deltas_list, 0.5 * (a_delta_avg + b_delta_avg))
#         a_deltas_center = np.array([])
#         b_deltas_center = np.array([])
#         b_site_counter = 1
#         for i in range(6):
#             if i%2 == 0:
#                 a_deltas_center = np.append(a_deltas_center, deltas[np.int(a_sites[i])])
#             else:
#                 b_deltas_center = np.append(b_deltas_center, deltas[np.int(b_sites[i-1*b_site_counter])])
#                 print(b_sites[i-1*b_site_counter])
#                 b_site_counter += 1
#         a_deltas_center_avg = np.abs(np.average(a_deltas_center))
#         b_deltas_center_avg = np.abs(np.average(b_deltas_center))
#         deltas_center_list = np.append(deltas_center_list, 0.5 * (a_deltas_center_avg + b_deltas_center_avg))
#     return (deltas_list, deltas_center_list)

def selfconsist_hartreefock_NonHermitian_DisorderAlreadyAdded(p, q, num_levels, initial_ham, initial_guess, tolerance, hfcoeff_list):
    a_sites, b_sites = site_assignment_NonHermitian(p, q, num_levels)  # Label sites for two basis lattice
    deltas_list = np.array([])  # Holds final delta value for each hfcoeff
    deltas_center_list = np.array([])
    for hfcoeff in hfcoeff_list:  # Go over all desired V values
        print(hfcoeff)
        converge_status = False
        deltas = np.array([])
        for d in range(np.size(initial_ham, 0)):
            if np.any(a_sites == d):
                deltas = np.append(deltas, initial_guess)
            else:
                deltas = np.append(deltas, -initial_guess)
        # deltas = np.repeat(initial_guess, np.size(initial_ham, 0)) #Initial guess
        ham = hartreefock_hamiltonian_addition_NonHermitian(p, q, num_levels, initial_ham, deltas, hfcoeff)  # HF Hamiltonian with initial guess
        converge_progress_count = 0
        while converge_status == False:
            start_time = time.time()
            # print('Diagonalization Start')
            eners, left_eigvec_matrix, right_eigvec_matrix = scipy.linalg.eig(ham,left=True,right=True)  # diagonalize ham to get energies and eigvecs
            # print('Diagonalization Done')
            left_eigvec_matrix = np.conj(left_eigvec_matrix)
            left_eigvec_matrix, right_eigvec_matrix = biortogonal_normalize(left_eigvec_matrix, right_eigvec_matrix)
            # print('Normalization Done')
            negative_energy_indices = np.where(eners < 0)  # Find indices where E<0
            left_negative_energy_eigvec_matrix = left_eigvec_matrix[:,negative_energy_indices[0]]
            right_negative_energy_eigvec_matrix = right_eigvec_matrix[:,negative_energy_indices[0]]
            # print('Getting filled states done')
            site_probs = np.array([])
            for row in range(np.size(left_negative_energy_eigvec_matrix, 0)):
                site_probs = np.append(site_probs, np.sum(left_negative_energy_eigvec_matrix[row,:]*right_negative_energy_eigvec_matrix[row,:]))
            # print('Calculating site probabilities done')
            prev_delta = deltas
            deltas = site_probs - 0.5
            converge_status = self_consistent_check(deltas, prev_delta, tolerance)
            # print('Getting convergence status done')
            ham = hartreefock_hamiltonian_addition_NonHermitian(p, q, num_levels, initial_ham, deltas, hfcoeff)
            # print('Getting new hamiltonian done')
            converge_progress_count += 1
            print(converge_progress_count, np.max(np.abs(deltas - prev_delta)), 'Time taken: {}'.format(time.time()-start_time))
        print('Convergence done')
        a_deltas = np.array([])
        b_deltas = np.array([])
        for d in range(len(deltas)):
            if np.any(a_sites == d):
                a_deltas = np.append(a_deltas, deltas[d])
            else:
                b_deltas = np.append(b_deltas, deltas[d])
        a_delta_avg = np.abs(np.average(a_deltas))
        b_delta_avg = np.abs(np.average(b_deltas))
        deltas_list = np.append(deltas_list, 0.5 * (a_delta_avg + b_delta_avg))
        a_deltas_center = np.array([])
        b_deltas_center = np.array([])
        a_site_counter = 0
        b_site_counter = 1
        for i in range(p):
            # print(i, i%2==0)
            if i % 2 == 0:
                a_deltas_center = np.append(a_deltas_center, deltas[int(a_sites[i - 1 * a_site_counter])])
                print(a_sites[i - 1 * a_site_counter])
                a_site_counter += 1
            else:
                b_deltas_center = np.append(b_deltas_center, deltas[int(b_sites[i - 1 * b_site_counter])])
                print(b_sites[i - 1 * b_site_counter])
                b_site_counter += 1
        a_deltas_center_avg = np.abs(np.average(a_deltas_center))
        b_deltas_center_avg = np.abs(np.average(b_deltas_center))
        deltas_center_list = np.append(deltas_center_list, 0.5 * (a_deltas_center_avg + b_deltas_center_avg))
    return (deltas_list, deltas_center_list)

def selfconsist_hartreefock_NonHermitian_DisorderAlreadyAdded_SaveRawOutputs(p, q, num_levels, initial_ham, initial_guess, tolerance, hfcoeff_list):
    a_sites, b_sites = site_assignment_NonHermitian(p, q, num_levels)  # Label sites for two basis lattice
    deltas_raw_list = np.zeros((1, np.size(initial_ham,1)+1))
    deltas_list = np.array([])  # Holds final delta value for each hfcoeff
    deltas_center_list = np.array([])
    for hfcoeff in hfcoeff_list:  # Go over all desired V values
        print(hfcoeff)
        converge_status = False
        deltas = np.array([])
        for d in range(np.size(initial_ham, 0)):
            if np.any(a_sites == d):
                deltas = np.append(deltas, initial_guess)
            else:
                deltas = np.append(deltas, -initial_guess)
        # deltas = np.repeat(initial_guess, np.size(initial_ham, 0)) #Initial guess
        ham = hartreefock_hamiltonian_addition_NonHermitian(p, q, num_levels, initial_ham, deltas,
                                                            hfcoeff)  # HF Hamiltonian with initial guess
        converge_progress_count = 0
        while converge_status == False:
            start_time = time.time()
            # print('Diagonalization Start')
            eners, left_eigvec_matrix, right_eigvec_matrix = scipy.linalg.eig(ham, left=True,
                                                                              right=True)  # diagonalize ham to get energies and eigvecs
            # print('Diagonalization Done')
            left_eigvec_matrix = np.conj(left_eigvec_matrix)
            left_eigvec_matrix, right_eigvec_matrix = biortogonal_normalize(left_eigvec_matrix, right_eigvec_matrix)
            # print('Normalization Done')
            negative_energy_indices = np.where(eners < 0)  # Find indices where E<0
            left_negative_energy_eigvec_matrix = left_eigvec_matrix[:, negative_energy_indices[0]]
            right_negative_energy_eigvec_matrix = right_eigvec_matrix[:, negative_energy_indices[0]]
            # print('Getting filled states done')
            site_probs = np.array([])
            for row in range(np.size(left_negative_energy_eigvec_matrix, 0)):
                site_probs = np.append(site_probs, np.sum(
                    left_negative_energy_eigvec_matrix[row, :] * right_negative_energy_eigvec_matrix[row, :]))
            # print('Calculating site probabilities done')
            prev_delta = deltas
            deltas = site_probs - 0.5
            converge_status = self_consistent_check(deltas, prev_delta, tolerance)
            # print('Getting convergence status done')
            ham = hartreefock_hamiltonian_addition_NonHermitian(p, q, num_levels, initial_ham, deltas, hfcoeff)
            # print('Getting new hamiltonian done')
            converge_progress_count += 1
            print(converge_progress_count, np.max(np.abs(deltas - prev_delta)),
                  'Time taken: {}'.format(time.time() - start_time))
        print('Convergence done')
        a_deltas = np.array([])
        b_deltas = np.array([])
        for d in range(len(deltas)):
            if np.any(a_sites == d):
                a_deltas = np.append(a_deltas, deltas[d])
            else:
                b_deltas = np.append(b_deltas, deltas[d])
        a_delta_avg = np.abs(np.average(a_deltas))
        b_delta_avg = np.abs(np.average(b_deltas))
        deltas_list = np.append(deltas_list, 0.5 * (a_delta_avg + b_delta_avg))
        a_deltas_center = np.array([])
        b_deltas_center = np.array([])
        a_site_counter = 0
        b_site_counter = 1
        for i in range(p):
            # print(i, i%2==0)
            if i % 2 == 0:
                a_deltas_center = np.append(a_deltas_center, deltas[int(a_sites[i - 1 * a_site_counter])])
                print(a_sites[i - 1 * a_site_counter])
                a_site_counter += 1
            else:
                b_deltas_center = np.append(b_deltas_center, deltas[int(b_sites[i - 1 * b_site_counter])])
                print(b_sites[i - 1 * b_site_counter])
                b_site_counter += 1
        a_deltas_center_avg = np.abs(np.average(a_deltas_center))
        b_deltas_center_avg = np.abs(np.average(b_deltas_center))
        deltas_center_list = np.append(deltas_center_list, 0.5 * (a_deltas_center_avg + b_deltas_center_avg))
        deltas_raw_list = np.vstack((deltas_raw_list,np.concatenate((np.array([hfcoeff]),deltas))))
    deltas_raw_list = deltas_raw_list[1:,:]
    return (deltas_raw_list, deltas_list, deltas_center_list)

def calc_from_raw_delta_data_NonHermitian(p,q,n,raw_delta_data):
    a_sites, b_sites = site_assignment_NonHermitian(p, q, n)
    deltas_list = np.array([])
    deltas_center_list = np.array([])
    for row in range(np.size(raw_delta_data,0)):
        deltas = raw_delta_data[row,1:]
        a_deltas = np.array([])
        b_deltas = np.array([])
        for d in range(len(deltas)):
            if np.any(a_sites == d):
                a_deltas = np.append(a_deltas, deltas[d])
            else:
                b_deltas = np.append(b_deltas, deltas[d])
        a_delta_avg = np.abs(np.average(a_deltas))
        b_delta_avg = np.abs(np.average(b_deltas))
        deltas_list = np.append(deltas_list, 0.5 * (a_delta_avg + b_delta_avg))
        a_deltas_center = np.array([])
        b_deltas_center = np.array([])
        a_site_counter = 0
        b_site_counter = 1
        for i in range(p):
            # print(i, i%2==0)
            if i % 2 == 0:
                a_deltas_center = np.append(a_deltas_center, deltas[int(a_sites[i - 1 * a_site_counter])])
                print(a_sites[i - 1 * a_site_counter])
                a_site_counter += 1
            else:
                b_deltas_center = np.append(b_deltas_center, deltas[int(b_sites[i - 1 * b_site_counter])])
                print(b_sites[i - 1 * b_site_counter])
                b_site_counter += 1
        a_deltas_center_avg = np.abs(np.average(a_deltas_center))
        b_deltas_center_avg = np.abs(np.average(b_deltas_center))
        deltas_center_list = np.append(deltas_center_list, 0.5 * (a_deltas_center_avg + b_deltas_center_avg))
    return(deltas_list, deltas_center_list)

def selfconsist_hartreefock_NonHermitian_Honeycomb_PBC_DisorderAlreadyAdded_SaveRawOutputs(nl, initial_ham, initial_guess, tolerance, hfcoeff_list):
    a_sites, b_sites = site_assignment_NonHermitian_Honeycomb(nl)  # Label sites for two basis lattice
    deltas_raw_list = np.zeros((1, np.size(initial_ham,1)+1))
    deltas_list = np.array([])  # Holds final delta value for each hfcoeff
    deltas_center_list = np.array([])
    for hfcoeff in hfcoeff_list:  # Go over all desired V values
        print(hfcoeff)
        converge_status = False
        deltas = np.array([])
        for d in range(np.size(initial_ham, 0)):
            if np.any(a_sites == d):
                deltas = np.append(deltas, initial_guess)
            else:
                deltas = np.append(deltas, -initial_guess)
        # deltas = np.repeat(initial_guess, np.size(initial_ham, 0)) #Initial guess
        ham = hartreefock_hamiltonian_addition_NonHermitian_Honeycomb(nl, initial_ham, deltas, hfcoeff)  # HF Hamiltonian with initial guess
        converge_progress_count = 0
        while converge_status == False:
            start_time = time.time()
            # print('Diagonalization Start')
            eners, left_eigvec_matrix, right_eigvec_matrix = scipy.linalg.eig(ham, left=True, right=True)  # diagonalize ham to get energies and eigvecs
            # print('Diagonalization Done')
            left_eigvec_matrix = np.conj(left_eigvec_matrix)
            left_eigvec_matrix, right_eigvec_matrix = biortogonal_normalize(left_eigvec_matrix, right_eigvec_matrix)
            # print('Normalization Done')
            negative_energy_indices = np.where(eners < 0)  # Find indices where E<0
            left_negative_energy_eigvec_matrix = left_eigvec_matrix[:, negative_energy_indices[0]]
            right_negative_energy_eigvec_matrix = right_eigvec_matrix[:, negative_energy_indices[0]]
            # print('Getting filled states done')
            site_probs = np.array([])
            for row in range(np.size(left_negative_energy_eigvec_matrix, 0)):
                site_probs = np.append(site_probs, np.sum(left_negative_energy_eigvec_matrix[row, :] * right_negative_energy_eigvec_matrix[row, :]))
            # print('Calculating site probabilities done')
            prev_delta = deltas
            deltas = site_probs - 0.5
            converge_status = self_consistent_check(deltas, prev_delta, tolerance)
            # print('Getting convergence status done')
            ham = hartreefock_hamiltonian_addition_NonHermitian_Honeycomb(nl, initial_ham, deltas, hfcoeff)
            # print('Getting new hamiltonian done')
            converge_progress_count += 1
            print(converge_progress_count, np.max(np.abs(deltas - prev_delta)), 'Time taken: {}'.format(time.time() - start_time))
        print('Convergence done')
        a_deltas = np.array([])
        b_deltas = np.array([])
        for d in range(len(deltas)):
            if np.any(a_sites == d):
                a_deltas = np.append(a_deltas, deltas[d])
            else:
                b_deltas = np.append(b_deltas, deltas[d])
        a_delta_avg = np.abs(np.average(a_deltas))
        b_delta_avg = np.abs(np.average(b_deltas))
        deltas_list = np.append(deltas_list, 0.5 * (a_delta_avg + b_delta_avg))
        a_deltas_center = np.array([])
        b_deltas_center = np.array([])
        a_site_counter = 0
        b_site_counter = 1
        for i in range(6):
            # print(i, i%2==0)
            if i % 2 == 0:
                a_deltas_center = np.append(a_deltas_center, deltas[int(a_sites[i - 1 * a_site_counter])])
                print(a_sites[i - 1 * a_site_counter])
                a_site_counter += 1
            else:
                b_deltas_center = np.append(b_deltas_center, deltas[int(b_sites[i - 1 * b_site_counter])])
                print(b_sites[i - 1 * b_site_counter])
                b_site_counter += 1
        a_deltas_center_avg = np.abs(np.average(a_deltas_center))
        b_deltas_center_avg = np.abs(np.average(b_deltas_center))
        deltas_center_list = np.append(deltas_center_list, 0.5 * (a_deltas_center_avg + b_deltas_center_avg))
        deltas_raw_list = np.vstack((deltas_raw_list,np.concatenate((np.array([hfcoeff]),deltas))))
    deltas_raw_list = deltas_raw_list[1:,:]
    return (deltas_raw_list, deltas_list, deltas_center_list)

from Fundamental.Square_Lattice import number_square_points

def hartreefock_hamiltonian_addition_NonHermitian_SquareLattice(nl, ham, deltas, hfcoeff):
    hf_addterm_diagvals = np.array([])
    for row in range(np.size(ham,0)):
        near_neighbor_indices = np.where(ham[row,:] != 0)[0]
        totalnum_points = number_square_points(nl)
        if row < totalnum_points/2: #Need this because code is getting confused due to small disorder
            near_neighbor_indices = near_neighbor_indices[np.where(near_neighbor_indices >= totalnum_points/2)[0]]
        else:
            near_neighbor_indices = near_neighbor_indices[np.where(near_neighbor_indices < totalnum_points/2)[0]]
        diag_vals = np.array([])
        for nni in near_neighbor_indices:
            diag_vals = np.append(diag_vals, deltas[nni])
        hf_addterm_diagvals = np.append(hf_addterm_diagvals, np.sum(diag_vals))
    hf_addterm_matrix = hfcoeff*np.diag(hf_addterm_diagvals)
    final_ham = ham + hf_addterm_matrix
    return(final_ham)

def site_assignment_NonHermitian_SquareLattice(num_levels):
    total_num_sites = number_square_points(num_levels)
    a_sites = np.linspace(0, int(total_num_sites / 2) - 1, int(total_num_sites / 2))
    b_sites = np.linspace(int(total_num_sites / 2), int(total_num_sites) - 1, int(total_num_sites / 2))
    return (a_sites, b_sites)

def selfconsist_hartreefock_NonHermitian_SquareLattice_DisorderAlreadyAdded_SaveRawOutputs(nl, initial_ham, initial_guess, tolerance, hfcoeff_list):
    a_sites, b_sites = site_assignment_NonHermitian_SquareLattice(nl)  # Label sites for two basis lattice
    deltas_raw_list = np.zeros((1, np.size(initial_ham,1)+1))
    deltas_list = np.array([])  # Holds final delta value for each hfcoeff
    for hfcoeff in hfcoeff_list:  # Go over all desired V values
        print(hfcoeff)
        converge_status = False
        deltas = np.array([])
        for d in range(np.size(initial_ham, 0)):
            if np.any(a_sites == d):
                deltas = np.append(deltas, initial_guess)
            else:
                deltas = np.append(deltas, -initial_guess)
        # deltas = np.repeat(initial_guess, np.size(initial_ham, 0)) #Initial guess
        ham = hartreefock_hamiltonian_addition_NonHermitian_SquareLattice(nl, initial_ham, deltas, hfcoeff)  # HF Hamiltonian with initial guess
        converge_progress_count = 0
        while converge_status == False:
            start_time = time.time()
            # print('Diagonalization Start')
            eners, left_eigvec_matrix, right_eigvec_matrix = scipy.linalg.eig(ham, left=True, right=True)  # diagonalize ham to get energies and eigvecs
            # print('Diagonalization Done')
            left_eigvec_matrix = np.conj(left_eigvec_matrix)
            left_eigvec_matrix, right_eigvec_matrix = biortogonal_normalize(left_eigvec_matrix, right_eigvec_matrix)
            # print('Normalization Done')
            negative_energy_indices = np.where(eners < 0)  # Find indices where E<0
            left_negative_energy_eigvec_matrix = left_eigvec_matrix[:, negative_energy_indices[0]]
            right_negative_energy_eigvec_matrix = right_eigvec_matrix[:, negative_energy_indices[0]]
            # print('Getting filled states done')
            site_probs = np.array([])
            for row in range(np.size(left_negative_energy_eigvec_matrix, 0)):
                site_probs = np.append(site_probs, np.sum(left_negative_energy_eigvec_matrix[row, :] * right_negative_energy_eigvec_matrix[row, :]))
            # print('Calculating site probabilities done')
            prev_delta = deltas
            deltas = site_probs - 0.5
            converge_status = self_consistent_check(deltas, prev_delta, tolerance)
            # print('Getting convergence status done')
            ham = hartreefock_hamiltonian_addition_NonHermitian_SquareLattice(nl, initial_ham, deltas, hfcoeff)
            # print('Getting new hamiltonian done')
            converge_progress_count += 1
            print(converge_progress_count, np.max(np.abs(deltas - prev_delta)), 'Time taken: {}'.format(time.time() - start_time))
        print('Convergence done')
        a_deltas = np.array([])
        b_deltas = np.array([])
        for d in range(len(deltas)):
            if np.any(a_sites == d):
                a_deltas = np.append(a_deltas, deltas[d])
            else:
                b_deltas = np.append(b_deltas, deltas[d])
        a_delta_avg = np.abs(np.average(a_deltas))
        b_delta_avg = np.abs(np.average(b_deltas))
        deltas_list = np.append(deltas_list, 0.5 * (a_delta_avg + b_delta_avg))
        deltas_raw_list = np.vstack((deltas_raw_list,np.concatenate((np.array([hfcoeff]),deltas))))
    deltas_raw_list = deltas_raw_list[1:,:]
    return (deltas_raw_list, deltas_list)

def site_assignment_nonHermitian_bilayer_honeycomb(nl):
    totnumsites = 2*honeycomb_points(nl)[1]
    asites = np.arange(0, totnumsites/4)
    bsites = np.arange(totnumsites/4, totnumsites/2)
    asites_otherlayer = np.arange(totnumsites/2, 3*totnumsites/4)
    bsites_otherlayer = np.arange(3*totnumsites/4, totnumsites)
    return(asites, bsites, asites_otherlayer, bsites_otherlayer)

def hartreefock_hamiltonian_addition_NonHermitian_Bilayer_Honeycomb(nl, ham, deltas, hfcoeff):
    hf_addterm_diagvals = np.array([])
    ham_nodiag = ham - np.diag(np.diag(ham))  # removes disorder so code does not get confused by it
    for row in range(np.size(ham,0)):
        near_neighbor_indices = np.where(ham_nodiag[row,:] != 0)[0]
        if row < int(np.size(ham,0) / 2):
            near_neighbor_indices = near_neighbor_indices[np.where(near_neighbor_indices < int(np.size(ham,0) / 2))[0]]
        else:
            near_neighbor_indices = near_neighbor_indices[np.where(near_neighbor_indices >= int(np.size(ham,0) / 2))[0]]
        diag_vals = deltas[near_neighbor_indices]
        hf_addterm_diagvals = np.append(hf_addterm_diagvals, np.sum(diag_vals))
    hf_addterm_matrix = hfcoeff*np.diag(hf_addterm_diagvals)
    final_ham = ham + hf_addterm_matrix
    return(final_ham)

def selfconsist_hartreefock_NonHermitian_Bilayer_Honeycomb_DisorderAlreadyAdded_SaveRawOutputs(nl, initial_ham, initial_guess, tolerance, hfcoeff_list):
    a_sites, b_sites, asites_otherlayer, bsites_otherlayer = site_assignment_nonHermitian_bilayer_honeycomb(nl)
    deltas_raw_list = np.zeros((1, np.size(initial_ham, 1) + 1))
    deltas_list = np.array([])  # Holds final delta value for each hfcoeff
    deltas_list_center = np.array([])
    for hfcoeff in hfcoeff_list:  # Go over all desired V values
        print(hfcoeff)
        converge_status = False
        deltas = np.array([])
        for d in range(np.size(initial_ham, 0)):
            if np.any(a_sites == d) or np.any(bsites_otherlayer == d):
                deltas = np.append(deltas, initial_guess)
            else:
                deltas = np.append(deltas, -initial_guess)
        # deltas = np.repeat(initial_guess, np.size(initial_ham, 0)) #Initial guess
        ham = hartreefock_hamiltonian_addition_NonHermitian_Bilayer_Honeycomb(nl, initial_ham, deltas, hfcoeff)  # HF Hamiltonian with initial guess
        converge_progress_count = 0
        while converge_status == False:
            start_time = time.time()
            # print('Diagonalization Start')
            eners, left_eigvec_matrix, right_eigvec_matrix = scipy.linalg.eig(ham, left=True, right=True)  # diagonalize ham to get energies and eigvecs
            # print('Diagonalization Done')
            left_eigvec_matrix = np.conj(left_eigvec_matrix)
            left_eigvec_matrix, right_eigvec_matrix = biortogonal_normalize(left_eigvec_matrix, right_eigvec_matrix)
            # print('Normalization Done')
            negative_energy_indices = np.where(eners < 0)  # Find indices where E<0
            left_negative_energy_eigvec_matrix = left_eigvec_matrix[:, negative_energy_indices[0]]
            right_negative_energy_eigvec_matrix = right_eigvec_matrix[:, negative_energy_indices[0]]
            # print('Getting filled states done')
            site_probs = np.array([])
            for row in range(np.size(left_negative_energy_eigvec_matrix, 0)):
                site_probs = np.append(site_probs, np.sum(left_negative_energy_eigvec_matrix[row, :] * right_negative_energy_eigvec_matrix[row, :]))
            # print('Calculating site probabilities done')
            prev_delta = deltas
            deltas = site_probs - 0.5
            converge_status = self_consistent_check(deltas, prev_delta, tolerance)
            # print('Getting convergence status done')
            ham = hartreefock_hamiltonian_addition_NonHermitian_Bilayer_Honeycomb(nl, initial_ham, deltas, hfcoeff)
            # print('Getting new hamiltonian done')
            converge_progress_count += 1
            print(converge_progress_count, np.max(np.abs(deltas - prev_delta)), 'Time taken: {}'.format(time.time() - start_time))
        print('Convergence done')
        a_deltas = np.array([])
        b_deltas = np.array([])
        for d in range(len(deltas)):
            if np.any(a_sites == d) or np.any(asites_otherlayer == d):
                a_deltas = np.append(a_deltas, np.abs(deltas[int(d)]))
            else:
                b_deltas = np.append(b_deltas, np.abs(deltas[int(d)]))
        a_delta_avg = np.abs(np.average(a_deltas))
        b_delta_avg = np.abs(np.average(b_deltas))
        deltas_list = np.append(deltas_list, 0.5 * (a_delta_avg + b_delta_avg))
        a_deltas_center = np.array([])
        b_deltas_center = np.array([])
        for d in np.concatenate((np.array([0,1,2]), np.array([0,1,2]) + np.size(initial_ham,0)/2, np.array([0,1,2]) + np.size(initial_ham,0)/4, np.array([0,1,2]) + 3*np.size(initial_ham,0)/4)):
            if np.any(a_sites == d) or np.any(asites_otherlayer == d):
                a_deltas_center = np.append(a_deltas_center, np.abs(deltas[int(d)]))
            else:
                b_deltas_center = np.append(b_deltas_center, np.abs(deltas[int(d)]))
        a_deltas_center_avg = np.abs(np.average(a_deltas_center))
        b_deltas_center_avg = np.abs(np.average(b_deltas_center))
        deltas_list_center = np.append(deltas_list_center, 0.5*(a_deltas_center_avg + b_deltas_center_avg))
        deltas_raw_list = np.vstack((deltas_raw_list, np.concatenate((np.array([hfcoeff]), deltas))))
    deltas_raw_list = deltas_raw_list[1:, :]
    return (deltas_raw_list, deltas_list, deltas_list_center)

def selfconsist_hartreefock_NonHermitian_DisorderAlreadyAdded_SaveRawOutputs_MoreEfficient(p, q, num_levels, initial_ham, initial_guess, tolerance, hfcoeff_list):
    a_sites, b_sites = site_assignment_NonHermitian(p, q, num_levels)  # Label sites for two basis lattice
    deltas_raw_list = np.zeros((1, np.size(initial_ham,1)+1))
    deltas_list = np.array([])  # Holds final delta value for each hfcoeff
    deltas_center_list = np.array([])
    for hfcoeff in hfcoeff_list:  # Go over all desired V values
        print(hfcoeff)
        converge_status = False
        deltas = np.array([])
        for d in range(np.size(initial_ham, 0)):
            if np.any(a_sites == d):
                deltas = np.append(deltas, initial_guess)
            else:
                deltas = np.append(deltas, -initial_guess)
        # deltas = np.repeat(initial_guess, np.size(initial_ham, 0)) #Initial guess
        ham = hartreefock_hamiltonian_addition_NonHermitian(p, q, num_levels, initial_ham, deltas, hfcoeff)  # HF Hamiltonian with initial guess
        converge_progress_count = 0
        while converge_status == False:
            start_time = time.time()
            # print('Diagonalization Start')
            eners, left_eigvec_matrix, right_eigvec_matrix = scipy.linalg.eig(ham, left=True, right=True)  # diagonalize ham to get energies and eigvecs
            del ham
            # print('Diagonalization Done')
            left_eigvec_matrix = np.conj(left_eigvec_matrix)
            left_eigvec_matrix, right_eigvec_matrix = biortogonal_normalize(left_eigvec_matrix, right_eigvec_matrix)
            # print('Normalization Done')
            negative_energy_indices = np.where(eners < 0)  # Find indices where E<0
            left_negative_energy_eigvec_matrix = left_eigvec_matrix[:, negative_energy_indices[0]]
            right_negative_energy_eigvec_matrix = right_eigvec_matrix[:, negative_energy_indices[0]]
            del eners
            del left_eigvec_matrix
            del right_eigvec_matrix
            del negative_energy_indices
            # print('Getting filled states done')
            site_probs = np.array([])
            for row in range(np.size(left_negative_energy_eigvec_matrix, 0)):
                site_probs = np.append(site_probs, np.sum(left_negative_energy_eigvec_matrix[row, :] * right_negative_energy_eigvec_matrix[row, :]))
            # print('Calculating site probabilities done')
            prev_delta = deltas
            deltas = site_probs - 0.5
            del left_negative_energy_eigvec_matrix
            del right_negative_energy_eigvec_matrix
            del site_probs
            converge_status = self_consistent_check(deltas, prev_delta, tolerance)
            # print('Getting convergence status done')
            ham = hartreefock_hamiltonian_addition_NonHermitian(p, q, num_levels, initial_ham, deltas, hfcoeff)
            # print('Getting new hamiltonian done')
            converge_progress_count += 1
            print(converge_progress_count, np.max(np.abs(deltas - prev_delta)), 'Time taken: {}'.format(time.time() - start_time))
        print('Convergence done')
        a_deltas = np.array([])
        b_deltas = np.array([])
        for d in range(len(deltas)):
            if np.any(a_sites == d):
                a_deltas = np.append(a_deltas, deltas[d])
            else:
                b_deltas = np.append(b_deltas, deltas[d])
        a_delta_avg = np.abs(np.average(a_deltas))
        b_delta_avg = np.abs(np.average(b_deltas))
        deltas_list = np.append(deltas_list, 0.5 * (a_delta_avg + b_delta_avg))
        a_deltas_center = np.array([])
        b_deltas_center = np.array([])
        a_site_counter = 0
        b_site_counter = 1
        for i in range(p):
            # print(i, i%2==0)
            if i % 2 == 0:
                a_deltas_center = np.append(a_deltas_center, deltas[int(a_sites[i - 1 * a_site_counter])])
                print(a_sites[i - 1 * a_site_counter])
                a_site_counter += 1
            else:
                b_deltas_center = np.append(b_deltas_center, deltas[int(b_sites[i - 1 * b_site_counter])])
                print(b_sites[i - 1 * b_site_counter])
                b_site_counter += 1
        a_deltas_center_avg = np.abs(np.average(a_deltas_center))
        b_deltas_center_avg = np.abs(np.average(b_deltas_center))
        deltas_center_list = np.append(deltas_center_list, 0.5 * (a_deltas_center_avg + b_deltas_center_avg))
        deltas_raw_list = np.vstack((deltas_raw_list,np.concatenate((np.array([hfcoeff]),deltas))))
    deltas_raw_list = deltas_raw_list[1:,:]
    return (deltas_raw_list, deltas_list, deltas_center_list)