"""Dictionary recovering experiment for univariate random dataset"""
import numpy as np
import matplotlib.pyplot as plt
from mdla import MultivariateDictLearning, MiniBatchMultivariateDictLearning
from mdla import multivariate_sparse_encode
from dict_metrics import hausdorff, emd, detectionRate
from numpy.linalg import norm
from numpy import array, arange, zeros
from numpy.random import rand, randn, permutation, randint
from time import time
from multiprocessing import cpu_count()

def _generate_testbed(kernel_init_len, n_nonzero_coefs, n_kernels,
                      n_samples=10, n_features=5, n_dims=3, snr=1000):
    """Generate a dataset from a random dictionary

    Generate a random dictionary and a dataset, where samples are combination of
    n_nonzero_coefs dictionary atoms. Noise is added, based on SNR value, with
    1000 indicated that no noise should be added.
    Return the dictionary, the dataset and an array indicated how atoms are combined
    to obtain each sample
    """
    print('Dictionary sampled from uniform distribution')
    dico = [rand(kernel_init_len, n_dims) for i in range(n_kernels)]
    for i in range(len(dico)):
        dico[i] /= norm(dico[i], 'fro')
    
    signals = list()
    decomposition = list()
    for i in range(n_samples):
        s = np.zeros(shape=(n_features, n_dims))
        d = np.zeros(shape=(n_nonzero_coefs, 3))
        rk = permutation(range(n_kernels))
        for j in range(n_nonzero_coefs):
            k_idx = rk[j]
            k_amplitude = 3. * rand() + 1.
            k_offset = randint(n_features - kernel_init_len + 1)
            s[k_offset:k_offset+kernel_init_len, :] += (k_amplitude *
                                                        dico[k_idx])
            d[j, :] = array([k_amplitude, k_offset, k_idx])
        decomposition.append(d)
        noise = randn(n_features, n_dims)
        if snr == 1000: alpha = 0
        else:
            ps = norm(s, 'fro')
            pn = norm(noise, 'fro')
            alpha = ps / (pn*10**(snr/20.))
        signals.append(s+alpha*noise)
    signals = np.array(signals)

    return dico, signals, decomposition

rng_global = np.random.RandomState(1)
n_samples, n_dims = 1500, 1
n_features = kernel_init_len = 5
n_nonzero_coefs = 3
n_kernels, max_iter, learning_rate = 50, 10, 1.5
n_jobs, batch_size = -1, None

iter_time, plot_separator, it_separator = list(), list(), 0

generating_dict, X, code = _generate_testbed(kernel_init_len, n_nonzero_coefs,
                                             n_kernels, n_samples, n_features,
                                             n_dims)

# Online without mini-batch
batch_size, n_jobs =n_samples, 1
learned_dict = MiniBatchMultivariateDictLearning(n_kernels=n_kernels, 
                                batch_size=batch_size, n_iter=max_iter,
                                n_nonzero_coefs=n_nonzero_coefs,
                                n_jobs=n_jobs, learning_rate=learning_rate,
                                kernel_init_len=kernel_init_len, verbose=1,
                                dict_init=None, random_state=rng_global)
ts = time()
learned_dict = learned_dict.fit(X)
iter_time.append((time()-ts) / max_iter)
it_separator += 1
plot_separator.append(it_separator)

# Online with mini-batch
minibatch_range = [cpu_count()]
minibatch_range.extend([cpu_count()*i for i in range(5, 21, 5)])
n_jobs = -1
for mb in minibatch_range:
    

# Batch learning

# Batch learning with one process

# Create a dictionary
# Update learned dictionary at each iteration and compute a distance
# with the generating dictionary
for i in range(max_iter):
    learned_dict = learned_dict.partial_fit(X)
    # Compute the detection rate
    detection_rate.append(detectionRate(learned_dict.kernels_,
                                        generating_dict, 0.97))
    # Compute the Wasserstein distance
    wasserstein.append(emd(learned_dict.kernels_, generating_dict,
                        'chordal', scale=True))
    # Get the objective error
    objective_error.append(array(learned_dict.error_ ).sum())
    
plot_univariate(array(objective_error), array(detection_rate),
                array(wasserstein), 'univariate-case')
    
