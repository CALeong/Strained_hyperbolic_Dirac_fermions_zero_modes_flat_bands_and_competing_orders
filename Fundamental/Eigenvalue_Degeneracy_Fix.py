import scipy
import numpy as np
from Fundamental.Hamiltonian import H0
from Fundamental.NonHermitian_Hamiltonian import NonHermitian_Hamiltonian as nhh
from Check.Orthogonality_Check import biorthogonality_matrix

def small_chaos(chaos_order, num_sites):
    chaos = np.array([])
    for n in range(int(num_sites)):
        chaos = np.append(chaos, np.random.uniform(-chaos_order/2,chaos_order/2))
    chaos_mat = np.diag(chaos)
    chaos_correction_val = np.trace(chaos_mat)/num_sites
    chaos_correction_mat = np.eye(int(num_sites))*chaos_correction_val
    return(chaos_mat-chaos_correction_mat)

def small_chaos_specifyseed(chaos_order, num_sites, seed):
    random_generator = np.random.default_rng(seed)
    chaos = random_generator.uniform(-chaos_order/2, chaos_order/2, size=int(num_sites))
    chaos_mat = np.diag(chaos)
    chaos_correction_val = np.trace(chaos_mat)/num_sites
    chaos_correction_mat = np.eye(int(num_sites))*chaos_correction_val
    return(chaos_mat-chaos_correction_mat)

# def gram_schmidt_projection_vector(v1, v2):
#     numerator = np.sum(np.conj(v1) * v2)
#     denominator = np.sum(np.conj(v2) * v2)
#     return((numerator/denominator) * v2)
#
# def gram_schmidt_orthogonalize(eigvec_matrix):
#     new_eigvecs = np.zeros(np.shape(eigvec_matrix))
#     new_eigvecs[:, 0] = eigvec_matrix[:, 0]
#     for col in range(1, np.size(eigvec_matrix, 1)):
#         projection_vectors = np.zeros((np.size(eigvec_matrix, 0), col))
#         for i in range(col):
#             projection_vectors[:, i] = gram_schmidt_projection_vector(eigvec_matrix[:, col], new_eigvecs[:, i])
#         new_entry = eigvec_matrix[:, col]
#         for pvi in range(np.size(projection_vectors, 1)):
#             new_entry = new_entry - projection_vectors[:, pvi]
#         new_eigvecs[:, col] = new_entry
#     return(new_eigvecs)
