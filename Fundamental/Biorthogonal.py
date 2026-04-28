import numpy as np
import scipy
# from Fundamental.NonHermitian_Hamiltonian import NonHermitian_Hamiltonian_Dagger as nhh_dagger
from Fundamental.NonHermitian_Hamiltonian import NonHermitian_Hamiltonian as nhh
#import pymor.algorithms.gram_schmidt as gs

# def biorthogonal_norm(ls_mat,rs_mat):
#     norm_factors = np.array([])
#     for i in range(np.size(ls_mat,0)):
#         norm_factors = np.append(norm_factors, np.sqrt(np.sum(ls_mat[:,i]*rs_mat[:,i])))
#     return(norm_factors)

def biortogonal_normalize(ls_mat,rs_mat):
    norm_factors = np.array([])
    for i in range(np.size(ls_mat, 0)):
        # print('i:{}'.format(i))
        norm_factors = np.append(norm_factors, np.sqrt(np.sum(ls_mat[:, i] * rs_mat[:, i]), dtype=np.clongdouble))

    # ls_mat_norm = np.empty((np.size(ls_mat,0), 1))
    # rs_mat_norm = np.empty((np.size(ls_mat,0), 1))
    # for col in range(np.size(ls_mat, 1)):
    #     print('col:{}'.format(col))
    #     ls_mat_norm = np.concatenate((ls_mat_norm, (ls_mat[:, col] / norm_factors[col]).reshape(np.size(ls_mat,0), 1)), axis=1)
    #     rs_mat_norm = np.concatenate((rs_mat_norm, (rs_mat[:, col] / norm_factors[col]).reshape(np.size(rs_mat, 0), 1)),axis=1)
    # ls_mat_norm = ls_mat_norm[:, 1:]
    # rs_mat_norm = rs_mat_norm[:, 1:]

    ls_mat_norm_new = ls_mat/norm_factors
    rs_mat_norm_new = rs_mat/norm_factors

    return(ls_mat_norm_new, rs_mat_norm_new)

# def left_eigvecs_ham(ham):
#     ham_conjtrans = np.conj(np.transpose(ham))
#     eners, ls_mat_pre = scipy.linalg.eig(ham_conjtrans)
#     ls_mat_pre = ls_mat_pre[:,np.argsort(eners)]
#     eners = np.sort(eners)
#     ls_mat = np.empty((len(eners), 1))
#     for col in range(np.size(ls_mat_pre, 1)):
#         ls_mat = np.concatenate((ls_mat, np.conj(ls_mat_pre[:, col]).reshape(len(eners), 1)), axis=1)
#     ls_mat = ls_mat[:, 1:]
#     return(ls_mat)
#
# def left_eigvecs(p,q,n,alpha,t):
#     ham_conjtrans = nhh_dagger(p,q,n,alpha,t)
#     eners, ls_mat_pre = scipy.linalg.eig(ham_conjtrans)
#     ls_mat_pre = ls_mat_pre[:,np.argsort(eners)]
#     eners = np.sort(eners)
#     ls_mat = np.conj(ls_mat_pre)
#     return(ls_mat)
#
# def check_lefts(p,q,n,alpha,t):
#     ls_theory = left_eigvecs(p,q,n,alpha,t)
#     ham = nhh(p,q,n,alpha,t)
#     eners, ls_scipy, rs_scipy = scipy.linalg.eig(ham,left=True,right=True)
#     ls_scipy = ls_scipy[:,np.argsort(eners)]
#     left_check_mat = np.empty((len(eners),len(eners)))
#     for e1 in range(len(eners)):
#         for e2 in range(len(eners)):
#             # left_check_mat[e1,e2] = len(np.unique(np.around(ls_theory[:,e1],5)/np.around(ls_scipy[:,e2],5)))
#             left_check_mat[e1,e2] = np.dot(ls_theory[:,e1],ls_scipy[:,e2])
#     return(left_check_mat)

# def biorthogonal_gramschmidt(ls_mat,rs_mat):
#     def inner_product(v1,v2):
#         return(np.sum(v1*v2))
#     def b_algo(k, akp1, g_list, c_list):
#         sum_terms = np.array([])
#         for i in range(k):
#             sum_terms = np.append(sum_terms, inner_product(akp1,g_list[i])*c_list[i])
#         return(akp1-np.sum(sum_terms))
#     def c_value()
#
#     g_list = np.array([])
#     c_list = np.array([])
#     b_list = np.array([]).reshape((np.size(ls_mat,1),1))
#     for i in range(np.size(ls_mat,0)):
#         b_list = np.append(b_list, b_algo(i,ls_mat[:,i+1],g_list,c_list))

# def biortogonal_normalize_check(ls_mat,rs_mat):
#     norm_factors = np.array([])
#     for i in range(np.size(ls_mat, 0)):
#         # print('i:{}'.format(i))
#         norm_factors = np.append(norm_factors, np.sqrt(np.sum(ls_mat[:, i] * rs_mat[:, i]), dtype=np.clongdouble))
#
#     ls_mat_norm = np.empty((np.size(ls_mat,0), 1))
#     rs_mat_norm = np.empty((np.size(ls_mat,0), 1))
#     for col in range(np.size(ls_mat, 1)):
#         print('col:{}'.format(col))
#         ls_mat_norm = np.concatenate((ls_mat_norm, (ls_mat[:, col] / norm_factors[col]).reshape(np.size(ls_mat,0), 1)), axis=1)
#         rs_mat_norm = np.concatenate((rs_mat_norm, (rs_mat[:, col] / norm_factors[col]).reshape(np.size(rs_mat, 0), 1)),axis=1)
#     ls_mat_norm = ls_mat_norm[:, 1:]
#     rs_mat_norm = rs_mat_norm[:, 1:]
#
#     ls_mat_norm_new = ls_mat/norm_factors
#     rs_mat_norm_new = rs_mat/norm_factors
#
#     return(ls_mat_norm_new, rs_mat_norm_new, ls_mat_norm, rs_mat_norm)
#
