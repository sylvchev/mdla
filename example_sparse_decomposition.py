"""Example of sparse decomposition with MOMP on random dataset"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from mdla import MultivariateDictLearning, MiniBatchMultivariateDictLearning
from mdla import multivariate_sparse_encode
from dict_metrics import hausdorff, emd, detectionRate
from numpy.linalg import norm
from numpy import array, arange, zeros
from time import time
from multiprocessing import cpu_count
from sklearn.utils import check_random_state

def _generate_testbed(kernel_init_len, n_nonzero_coefs, n_kernels, rng=None,
                      n_samples=10, n_features=5, n_dims=3, snr=1000,
                      Gaussian=False):
    """Generate a dataset from a random dictionary

    Generate a random dictionary and a dataset, where samples are combination of
    n_nonzero_coefs dictionary atoms. Noise is added, based on SNR value, with
    1000 indicated that no noise should be added.
    Return the dictionary, the dataset and an array indicated how atoms are 
    combined to obtain each sample
    """
    rng = check_random_state(rng)

    if Gaussian:
        print('Dictionary of size (', kernel_init_len, ',', n_dims,
              ') sampled from Gaussian distribution')
        dico = [rng.randn(kernel_init_len, n_dims) for i in range(n_kernels)]
    else:
        print('Dictionary of size (', kernel_init_len, ',', n_dims,
              ') sampled from uniform distribution')
        dico = [rng.rand(kernel_init_len, n_dims) for i in range(n_kernels)]
        
    for i in range(len(dico)):
        dico[i] /= norm(dico[i], 'fro')
    
    signals = list()
    decomposition = list()
    for i in range(n_samples):
        s = np.zeros(shape=(n_features, n_dims))
        d = np.zeros(shape=(n_nonzero_coefs, 3))
        rk = rng.permutation(range(n_kernels))
        for j in range(n_nonzero_coefs):
            k_idx = rk[j]
            k_amplitude = 3. * rng.rand() + 1.
            k_offset = rng.randint(n_features - kernel_init_len + 1)
            s[k_offset:k_offset+kernel_init_len, :] += (k_amplitude *
                                                        dico[k_idx])
            d[j, :] = array([k_amplitude, k_offset, k_idx])
        decomposition.append(d)
        noise = rng.randn(n_features, n_dims)
        if snr == 1000: alpha = 0
        else:
            ps = norm(s, 'fro')
            pn = norm(noise, 'fro')
            alpha = ps / (pn*10**(snr/20.))
        signals.append(s+alpha*noise)

    return dico, signals, decomposition

rng_global = np.random.RandomState(1)
n_nonzero_coefs = 15

def decomposition_random_dictionary(Gaussian=True, rng=None, n_features=65,
                                    n_dims=1):
    """Generate a dataset from a random dictionary and compute decomposition

    A dataset of n_samples examples is generated from a random dictionary,
    each sample containing a random mixture of n_nonzero_coef atoms and has
    a dimension of n_features by n_dims. All the examples are decomposed with 
    sparse multivariate OMP, written as:
    (Eq. 1) min_a ||x - Da ||^2 s.t. ||a||_0 <= k 
    with x in R^(n_features x n_dims), D in R^(n_features x n_kernels) and
    a in R^n_kernels. 

    Returns a ndarray of (n_nonzero_coefs, n_samples) containing all the
    root mean square error (RMSE) computed as the residual of the decomposition
    for all samples for sparsity constraint values of (Eq. 1) going from 1
    to n_nonzero_coefs.
    """
    n_samples = 100
    kernel_init_len = n_features
    n_kernels = 50
    n_jobs = 1

    dictionary, X, code = _generate_testbed(kernel_init_len=kernel_init_len,
            n_nonzero_coefs=n_nonzero_coefs, n_kernels=n_kernels,
            n_samples=n_samples, n_features=n_features, n_dims=n_dims,
            rng=rng_global, Gaussian=Gaussian)
    rmse = zeros(shape=(n_nonzero_coefs, n_samples))
    for k in range(n_nonzero_coefs):
        for idx, s in enumerate(X):
            r, _ = multivariate_sparse_encode(array(s, ndmin=3), dictionary,
                        n_nonzero_coefs=k+1, n_jobs=n_jobs, verbose=1)
            rmse[k, idx] = norm(r[0], 'fro')/norm(s, 'fro')*100
    return rmse

rmse_gaussian = decomposition_random_dictionary(Gaussian=True, rng=rng_global,
                                                n_features=65, n_dims=1)

rmse_uniform = decomposition_random_dictionary(Gaussian=False, rng=rng_global,
                                                n_features=65, n_dims=1)

fig = plt.figure(figsize=(10,8))
plt.title(r'Initialization of the random univariate dictionary')
km = fig.add_subplot(1,1,1)
km.errorbar(range(1, n_nonzero_coefs+1), rmse_uniform.mean(1),
            yerr=rmse_uniform.std(1), label='Uniform')
km.errorbar(range(1, n_nonzero_coefs+1), rmse_gaussian.mean(1),
            yerr=rmse_gaussian.std(1), color='r', label='Gaussian')
km.plot(range(n_nonzero_coefs+2), np.zeros(n_nonzero_coefs+2), 'k')
km.axis([0, n_nonzero_coefs+1, 0, 90])
km.set_xticks(range(0, n_nonzero_coefs+2, 5))
km.set_xlabel('k')
km.set_ylabel('rRMSE (%)')
km.legend(loc='upper right')
plt.savefig('sparse_decomposition_univariate.png')

rmse_gaussian = decomposition_random_dictionary(Gaussian=True, rng=rng_global,
                                                n_features=65, n_dims=5)

rmse_uniform = decomposition_random_dictionary(Gaussian=False, rng=rng_global,
                                                n_features=65, n_dims=5)

fig = plt.figure(figsize=(10,8))
plt.title(r'Initialization of the random multivariate dictionary')
km = fig.add_subplot(1,1,1)
km.errorbar(range(1, n_nonzero_coefs+1), rmse_uniform.mean(1),
            yerr=rmse_uniform.std(1), label='Uniform')
km.errorbar(range(1, n_nonzero_coefs+1), rmse_gaussian.mean(1),
            yerr=rmse_gaussian.std(1), color='r', label='Gaussian')
km.plot(range(n_nonzero_coefs+2), np.zeros(n_nonzero_coefs+2), 'k')
km.axis([0, n_nonzero_coefs+1, 0, 90])
km.set_xticks(range(0, n_nonzero_coefs+2, 5))
km.set_xlabel('k')
km.set_ylabel('rRMSE (%)')
km.legend(loc='upper right')
plt.savefig('sparse_decomposition_multivariate.png')


# test uniform vs gaussian, for univariate, multivariate and kernelized version
# function avec Gaussian, n_dims return 2 rmse_uniform, rmse_gaussian


# Gaussian case
# gaussian_dict, X, code = _generate_testbed(kernel_init_len=kernel_init_len,
#                                              n_nonzero_coefs=n_nonzero_coefs,
#                                              n_kernels=n_kernels,
#                                              n_samples=n_samples,
#                                              n_features=n_features,
#                                              n_dims=n_dims,
#                                              rng=rng_global,
#                                              Gaussian=True)
# rmse_gaussian = zeros(shape=(n_nonzero_coefs, n_samples))
# for k in range(n_nonzero_coefs):
#     for idx, s in enumerate(X):
#         r, _ = multivariate_sparse_encode(array(s, ndmin=3), gaussian_dict,
#                                       n_nonzero_coefs=k+1,
#                                       n_jobs=n_jobs,
#                                       verbose=1)
#         rmse_gaussian[k, idx] = norm(r[0], 'fro')/norm(s, 'fro')*100

# # Uniform case
# uniform_dict, X, code = _generate_testbed(kernel_init_len=kernel_init_len,
#                                           n_nonzero_coefs=n_nonzero_coefs,
#                                           n_kernels=n_kernels,
#                                           n_samples=n_samples,
#                                           n_features=n_features,
#                                           n_dims=n_dims,
#                                           rng=rng_global,
#                                           Gaussian=False)
# rmse_uniform = zeros(shape=(n_nonzero_coefs, n_samples))
# for k in range(n_nonzero_coefs):
#     for idx, s in enumerate(X):
#         r, _ = multivariate_sparse_encode(array(s, ndmin=3), uniform_dict,
#                                       n_nonzero_coefs=k+1,
#                                       n_jobs=n_jobs,
#                                       verbose=1)
#         rmse_uniform[k, idx] = norm(r[0], 'fro')/norm(s, 'fro')*100

# fig = plt.figure(figsize=(10,8))
# plt.title(r'Initialization of the generating dictionary')
# km = fig.add_subplot(1,1,1)
# km.errorbar(range(1, n_nonzero_coefs+1), rmse_uniform.mean(1),
#             yerr=rmse_uniform.std(1), label='Uniform')
# km.errorbar(range(1, n_nonzero_coefs+1), rmse_gaussian.mean(1),
#             yerr=rmse_gaussian.std(1), color='r', label='Gaussian')
# km.plot(range(n_nonzero_coefs+2), np.zeros(n_nonzero_coefs+2), 'k')
# km.axis([0, n_nonzero_coefs+1, 0, 90])
# km.set_xticks(range(0, n_nonzero_coefs+2, 5))
# km.set_xlabel('k')
# km.set_ylabel('rRMSE (%)')
# km.legend(loc='upper right')
# plt.savefig('omp-rmse-UniformGaussian.png')
        
