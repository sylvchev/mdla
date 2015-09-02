"""Dictionary recovering experiment for univariate random dataset"""
import numpy as np
import matplotlib.pyplot as plt
from mdla import MultivariateDictLearning
from mdla import multivariate_sparse_encode
from dict_metrics import hausdorff, emd, detectionRate
from numpy.linalg import norm
from numpy import array, arange

def plot_univariate(objective_error, detection_rate, wasserstein, figname):
    fig = plt.figure(figsize=(10,6))
    
    # plotting data from objective error
    objerr = fig.add_subplot(1,3,1)
    oe = objerr.plot(arange(1, len(objective_error)+1), objective_error, 
                     color='green', label=r'Objective error')
    # objerr.axis([0, len(objective_error)-1, 0, np.max(objective_error)])
    # objerr.set_xticks(arange(0,len(objective_error)+1,10))
    objerr.set_xlabel('Iteration')
    objerr.set_ylabel(r'Error (no unit)')
    objerr.legend(loc='lower right')
    
    # plotting data from detection rate 0.97
    detection = fig.add_subplot(1,3,2)        
    detrat = detection.plot(arange(1,len(detection_rate)+1), detection_rate,
                            color='magenta', label=r'Detection rate 0.97')
    # detection.axis([0, len(detection_rate), 0, 100])
    # detection.set_xticks(arange(0, len(detection_rate),10))
    # detection.set_xlabel('Iteration')
    detection.set_ylabel(r'Recovery rate (in %)')
    detection.legend(loc='lower right')
    
    # plotting data from our metric
    met = fig.add_subplot(1,3,3)
    wass = met.plot(arange(1, len(wasserstein)+1), 100-wasserstein,
                    label=r'$d_W$', color='red') 
    # met.axis([0, len(wasserstein), 0, 100])
    # met.set_xticks(arange(0,len(wasserstein),10))
    detection.set_xlabel('Iteration')
    detection.set_ylabel(r'Recovery rate (in %)')
    met.legend(loc='lower right')
    
    # plt.tight_layout(.5)
    plt.savefig(figname+'.png')

def _generate_testbed(kernel_init_len, n_nonzero_coefs, n_kernels,
                      n_samples=10, n_features=5, n_dims=3):
    """Generate a dataset from a random dictionary

    Generate a random dictionary and a dataset, where samples are combination of
    n_nonzero_coefs dictionary atoms.
    Return the dictionary, the dataset and an array indicated how atoms are combined
    to obtain each sample
    """
    dico = [np.random.randn(kernel_init_len, n_dims) for i in range(n_kernels)]
    for i in range(len(dico)):
        dico[i] /= np.linalg.norm(dico[i], 'fro')
    
    signals = list()
    decomposition = list()
    for i in range(n_samples):
        s = np.zeros(shape=(n_features, n_dims))
        d = np.zeros(shape=(n_nonzero_coefs, 3))
        rk = np.random.permutation(range(n_kernels))
        for j in range(n_nonzero_coefs):
            k_idx = rk[j]
            k_amplitude = 3. * np.random.rand() + 1.
            k_offset = np.random.randint(n_features - kernel_init_len + 1)
            s[k_offset:k_offset+kernel_init_len, :] += (k_amplitude *
                                                        dico[k_idx])
            d[j, :] = np.array([k_amplitude, k_offset, k_idx])
        decomposition.append(d)
        signals.append(s)
    signals = np.array(signals)

    return dico, signals, decomposition

rng_global = np.random.RandomState(0)
n_samples, n_features, n_dims = 1500, 50, 1
n_nonzero_coefs = 3
n_kernels, max_iter, kernel_init_len, learning_rate = 50, 20, 50, 1.5
n_jobs = -1
detection_rate, wasserstein, objective_error = list(), list(), list()
recov = zeros((max_iter))

generating_dict, X, code = _generate_testbed(kernel_init_len, n_nonzero_coefs,
                                             n_kernels, n_samples, n_features,
                                             n_dims)

# Create a dictionary
learned_dict = MultivariateDictLearning(n_kernels=n_kernels, max_iter=1,
                                n_nonzero_coefs=n_nonzero_coefs,
                                n_jobs=n_jobs, learning_rate=learning_rate,
                                kernel_init_len=kernel_init_len, verbose=1,
                                dict_init=None, random_state=rng_global)
# Update learned dictionary at each iteration and compute a distance
# with the generating dictionary
for i in range(max_iter):
    learned_dict = learned_dict.fit(X)
    # Compute the detection rate
    detection_rate.append(detectionRate(learned_dict.kernels_,
                                        generating_dict, 0.97))
    # Compute the Wasserstein distance
    wasserstein.append(emd(learned_dict.kernels_, generating_dict,
                        'chordal', scale=True))
    # Get the objective error
    objective_error.append(array(learned_dict.error_ ).sum())
    

    


# residual, code = multivariate_sparse_encode(X, dico)
# print ('Objective error for each samples is:')
# for i in range(len(residual)):
#     print ('Sample', i, ':', norm(residual[i], 'fro') + len(code[i]))
    