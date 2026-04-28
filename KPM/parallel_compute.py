from joblib import parallel_config, Parallel, delayed
from KPM.KPM import moments_ADOS_general
import numpy as np
from KPM.measure import rescale_disorder_unity_window, rescale_operator_unity_window
import scipy.sparse
import pickle


def random_vectors_arr_generate(num_rand_vecs, vec_dim):
    return np.random.uniform(-np.sqrt(3), np.sqrt(3), size=(vec_dim, num_rand_vecs))


def rescaled_onsite_disorder_generate(W, num_sites, eigval_min, eigval_max, epsilon=0.01):
    return rescale_disorder_unity_window(np.random.uniform(-W/2, W/2, size=num_sites),
                                         eigval_min,
                                         eigval_max,
                                         epsilon=epsilon)


def run_KPM_ADOS_parallel_disorder(n_jobs, sparse_ham, W_list, number_moments,
                                   num_rand_vecs, rand_vec_dim,
                                   save_dir, save_name, save_index_list):

    def run_routine(sparse_ham, number_moments, random_vecs_arr, save_dir, save_name, save_index):
        result = moments_ADOS_general(sparse_ham, number_moments, random_vecs_arr)
        np.save(save_dir + '/' + save_name + '_' + save_index, result)

    with parallel_config(backend='loky'):
        Parallel(n_jobs=n_jobs)(
            delayed(run_routine)
            (rescale_operator_unity_window(sparse_ham + scipy.sparse.diags(np.random.uniform(-W/2, W/2,
                                                                                             size=rand_vec_dim),
                                                                           format='csr'),
                                           eigval_min=-3.12, eigval_max=3.12
                                           ),
            number_moments,
            random_vectors_arr_generate(num_rand_vecs, rand_vec_dim),
            save_dir,
            save_name,
            save_index)
            for (W, save_index) in zip(W_list, save_index_list)
        )





