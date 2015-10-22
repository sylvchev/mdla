"""Dictionary recovering experiment for univariate random dataset"""
import numpy as np
import matplotlib.pyplot as plt
from mdla import MultivariateDictLearning, MiniBatchMultivariateDictLearning
from mdla import multivariate_sparse_encode
from dict_metrics import hausdorff, emd, detectionRate
from numpy.linalg import norm
from numpy import array, arange, zeros, min, max
from numpy.random import rand, randn, permutation, randint, RandomState

def plot_univariate(objective_error, detection_rate, wasserstein,
                    n_iter, figname):
    fig = plt.figure(figsize=(15,5))
    if n_iter == 1: step = 5
    else: step = n_iter
    
    # plotting data from objective error
    objerr = fig.add_subplot(1,3,1)
    _ = objerr.plot(step*arange(1, len(objective_error)+1), objective_error, 
                     color='green', label=r'Objective error')
    objerr.axis([0, len(objective_error)-1, min(objective_error),
                 max(objective_error)])
    objerr.set_xticks(arange(0, step*len(objective_error)+1, step))
    objerr.set_xlabel('Iteration')
    objerr.set_ylabel(r'Error (no unit)')
    objerr.legend(loc='upper right')
    
    # plotting data from detection rate 0.99
    detection = fig.add_subplot(1,3,2)        
    _ = detection.plot(step*arange(1,len(detection_rate)+1), detection_rate,
                            color='magenta', label=r'Detection rate 0.99')
    detection.axis([0, len(detection_rate), 0, 100])
    detection.set_xticks(arange(0, step*len(detection_rate)+1, step))
    detection.set_xlabel('Iteration')
    detection.set_ylabel(r'Recovery rate (in %)')
    detection.legend(loc='upper left')
    
    # plotting data from our metric
    met = fig.add_subplot(1,3,3)
    _ = met.plot(step*arange(1, len(wasserstein)+1), 100-wasserstein,
                    label=r'$d_W$', color='red') 
    met.axis([0, len(wasserstein), 0, 100])
    met.set_xticks(arange(0, step*len(wasserstein)+1, step))
    met.set_xlabel('Iteration')
    met.set_ylabel(r'Recovery rate (in %)')
    met.legend(loc='upper left')
    
    plt.tight_layout(.5)
    plt.savefig(figname+'.png')

def _generate_testbed(kernel_init_len, n_nonzero_coefs, n_kernels,
                      n_samples=10, n_features=5, n_dims=3, snr=1000):
    """Generate a dataset from a random dictionary

    Generate a random dictionary and a dataset, where samples are combination of
    n_nonzero_coefs dictionary atoms. Noise is added, based on SNR value, with
    1000 indicated that no noise should be added.
    Return the dictionary, the dataset and an array indicated how atoms are combined
    to obtain each sample
    """
    dico = [randn(kernel_init_len, n_dims) for i in range(n_kernels)]
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

rng_global = RandomState(1)
n_samples, n_dims = 1500, 1
n_features = kernel_init_len = 20
n_nonzero_coefs = 3
n_kernels, max_iter, n_iter, learning_rate = 50, 10, 3, 1.5
n_jobs, batch_size = -1, 10
detection_rate, wasserstein, objective_error = list(), list(), list()

generating_dict, X, code = _generate_testbed(kernel_init_len, n_nonzero_coefs,
                                             n_kernels, n_samples, n_features,
                                             n_dims)

# # Create a dictionary
# dict_init = [rand(kernel_init_len, n_dims) for i in range(n_kernels)]
# for i in range(len(dict_init)):
#     dict_init[i] /= norm(dict_init[i], 'fro')
dict_init = None
    
learned_dict = MiniBatchMultivariateDictLearning(n_kernels=n_kernels, 
                                batch_size=batch_size, n_iter=n_iter,
                                n_nonzero_coefs=n_nonzero_coefs,
                                n_jobs=n_jobs, learning_rate=learning_rate,
                                kernel_init_len=kernel_init_len, verbose=1,
                                dict_init=dict_init, random_state=rng_global)

# Update learned dictionary at each iteration and compute a distance
# with the generating dictionary
for i in range(max_iter):
    learned_dict = learned_dict.partial_fit(X)
    # Compute the detection rate
    detection_rate.append(detectionRate(learned_dict.kernels_,
                                        generating_dict, 0.99))
    # Compute the Wasserstein distance
    wasserstein.append(emd(learned_dict.kernels_, generating_dict,
                        'chordal', scale=True))
    # Get the objective error
    objective_error.append(array(learned_dict.error_ ).sum())
    
plot_univariate(array(objective_error), array(detection_rate),
                array(wasserstein), n_iter, 'univariate-case')
    
# Another possibility is to rely on a callback function such as 
def callback_distance(loc):
    ii, iter_offset = loc['ii'], loc['iter_offset']
    n_batches = loc['n_batches']
    if np.mod((ii-iter_offset)/int(n_batches), n_iter) == 0:
        # Compute distance only every 5 iterations, as in previous case
        d = loc['dict_obj']
        d.wasserstein.append(emd(loc['dictionary'], d.generating_dict, 
                                 'chordal', scale=True))
        d.detection_rate.append(detectionRate(loc['dictionary'],
                                              d.generating_dict, 0.99))
        d.objective_error.append(loc['current_cost']) 

# reinitializing the random generator
learned_dict2 = MiniBatchMultivariateDictLearning(n_kernels=n_kernels, 
                                batch_size=batch_size, n_iter=max_iter*n_iter,
                                n_nonzero_coefs=n_nonzero_coefs,
                                callback=callback_distance,
                                n_jobs=n_jobs, learning_rate=learning_rate,
                                kernel_init_len=kernel_init_len, verbose=1,
                                dict_init=dict_init, random_state=rng_global)
learned_dict2.generating_dict = list(generating_dict)
learned_dict2.wasserstein = list()
learned_dict2.detection_rate = list()
learned_dict2.objective_error = list()

learned_dict2 = learned_dict2.fit(X)

plot_univariate(array(learned_dict2.objective_error),
                array(learned_dict2.detection_rate),
                array(learned_dict2.wasserstein),
                n_iter=1, figname='univariate-case-callback')
    
