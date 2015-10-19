"""Dictionary recovering experiment for univariate random dataset"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from mdla import MultivariateDictLearning, MiniBatchMultivariateDictLearning
from mdla import multivariate_sparse_encode
from dict_metrics import hausdorff, emd, detectionRate, betaDist
from numpy.linalg import norm
from numpy import array, arange, zeros, min, max
from numpy.random import rand, randn, permutation, randint
import pickle
from os.path import exists

# TODO: Add SNR, repeat experiments to make stats, make a fast and a 
#       long version, 

def _generate_testbed(kernel_init_len, n_nonzero_coefs, n_kernels,
                      n_samples=10, n_features=5, n_dims=3, snr=1000):
    """Generate a dataset from a random dictionary

    Generate a random dictionary and a dataset, where samples are combination 
    of n_nonzero_coefs dictionary atoms. Noise is added, based on SNR value,
    with 1000 indicated that no noise should be added.
    Return the dictionary, the dataset and an array indicated how atoms are
    combined to obtain each sample
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

def plot_recov(wc, wfs, hc, hfs, bd, dr99, dr97, n_iter, figname):
    fig = plt.figure(figsize=(18,10))
    
    # plotting data from detection rate SNR 30
    detection = fig.add_subplot(3,4,1)
    det99 = detection.boxplot(dr99[0,:,:])
    plt.setp(det99['medians'], color='green')
    plt.setp(det99['caps'], color='green')
    plt.setp(det99['boxes'], color='green')
    plt.setp(det99['fliers'], color='green')
    plt.setp(det99['whiskers'], color='green')
    medianlt99 = [median.get_ydata()[0] for n,median in enumerate(det99['medians'])]
    axlt99 = detection.plot(np.arange(1,n+2), medianlt99, linewidth=1, color='green', label=r'$c_\operatorname{99}$')
    det97 = detection.boxplot(dr97[0,:,:])
    plt.setp(det97['medians'], color='mangenta')
    plt.setp(det97['caps'], color='mangenta')
    plt.setp(det97['boxes'], color='mangenta')
    plt.setp(det97['fliers'], color='mangenta')
    plt.setp(det97['whiskers'], color='mangenta')
    medianlt97 = [median.get_ydata()[0] for n,median in enumerate(det97['medians'])]
    axlt97 = detection.plot(np.arange(1,n+2), medianlt97, linewidth=1, color='magenta', label=r'$c_\operatorname{97}$')
    detection.axis([0, n, 0, 100])
    detection.set_xticks(np.arange(0,n+1,10))
    detection.set_title(r'SNR 30')
    detection.legend(loc='lower right')
    
    # plotting data from hausdorff metric
    methaus = fig.add_subplot(3, 4, 2)
    hausch = methaus.boxplot(1-hc[0,:,:]/100.) 
    plt.setp(hausch['medians'], color='cyan')
    plt.setp(hausch['caps'], color='cyan')
    plt.setp(hausch['boxes'], color='cyan')
    plt.setp(hausch['fliers'], color='cyan')
    plt.setp(hausch['whiskers'], color='cyan')
    medianhc = [median.get_ydata()[0] for n,median in enumerate(hausch['medians'])]
    axhc = methaus.plot(np.arange(1,n+2), medianhc, linewidth=1, label=r'$1-d_H^c$', color='cyan')
    hausfs = methaus.boxplot(1-hfs[0,:,:]/100.) 
    plt.setp(hausfs['medians'], color='yellow')
    plt.setp(hausfs['caps'], color='yellow')
    plt.setp(hausfs['boxes'], color='yellow')
    plt.setp(hausfs['fliers'], color='yellow')
    plt.setp(hausfs['whiskers'], color='yellow')
    medianhfs = [median.get_ydata()[0] for n,median in enumerate(hausfs['medians'])]
    axhfs = methaus.plot(np.arange(1,n+2), medianhfs, linewidth=1, label=r'$1-d_H^{fs}$', color='yellow')
    methaus.axis([0, n, 0, 1])
    methaus.set_xticks(np.arange(0,n+1,10))
    methaus.legend(loc='lower right')
    methaus.set_title(r'SNR 30')

    # plotting data from wasserstein metric
    metwass = fig.add_subplot(3, 4, 3)
    wassch = metwass.boxplot(1-wc[0,:,:]/100.) 
    plt.setp(wassch['medians'], color='red')
    plt.setp(wassch['caps'], color='red')
    plt.setp(wassch['boxes'], color='red')
    plt.setp(wassch['fliers'], color='red')
    plt.setp(wassch['whiskers'], color='red')
    medianwc = [median.get_ydata()[0] for n,median in enumerate(wassch['medians'])]
    axwc = metwass.plot(np.arange(1,n+2), medianwc, linewidth=1, label=r'$1-d_W^c$', color='red')
    wassfs = metwass.boxplot(1-hfs[0,:,:]/100.) 
    plt.setp(wassfs['medians'], color='blue')
    plt.setp(wassfs['caps'], color='blue')
    plt.setp(wassfs['boxes'], color='blue')
    plt.setp(wassfs['fliers'], color='blue')
    plt.setp(wassfs['whiskers'], color='blue')
    medianwfs = [median.get_ydata()[0] for n,median in enumerate(wassfs['medians'])]
    axwfs = metwass.plot(np.arange(1,n+2), medianwfs, linewidth=1, label=r'$1-d_W^{fs}$', color='blue')
    metwass.axis([0, n, 0, 1])
    metwass.set_xticks(np.arange(0,n+1,10))
    metwass.legend(loc='lower right')
    metwass.set_title(r'SNR 30')
    
    # plotting data from Beta SNR 30
    metbeta = fig.add_subplot(3, 4, 4)
    betad = metbeta.boxplot(1-bd[0,:,:]/100.) 
    plt.setp(betad['medians'], color='black')
    plt.setp(betad['caps'], color='black')
    plt.setp(betad['boxes'], color='black')
    plt.setp(betad['fliers'], color='black')
    plt.setp(betad['whiskers'], color='black')
    medianbd = [median.get_ydata()[0] for n,median in enumerate(betad['medians'])]
    axbd = metbeta.plot(np.arange(1,n+2), medianbd, linewidth=1, label=r'$1-d_\beta$', color='black')
    metbeta.axis([0, n, 0, 1])
    metbeta.set_xticks(np.arange(0,n+1,10))
    metbeta.legend(loc='lower right')
    metbeta.set_title(r'SNR 30')

    # plotting data from detection rate for SNR 20
    detection = fig.add_subplot(3, 4, 5)
    det99 = detection.boxplot(dr99[1,:,:])
    plt.setp(det99['medians'], color='green')
    plt.setp(det99['caps'], color='green')
    plt.setp(det99['boxes'], color='green')
    plt.setp(det99['fliers'], color='green')
    plt.setp(det99['whiskers'], color='green')
    medianlt99 = [median.get_ydata()[0] for n,median in enumerate(det99['medians'])]
    axlt99 = detection.plot(np.arange(1,n+2), medianlt99, linewidth=1, color='green', label=r'$c_\operatorname{99}$')
    det97 = detection.boxplot(dr97[1,:,:])
    plt.setp(det97['medians'], color='mangenta')
    plt.setp(det97['caps'], color='mangenta')
    plt.setp(det97['boxes'], color='mangenta')
    plt.setp(det97['fliers'], color='mangenta')
    plt.setp(det97['whiskers'], color='mangenta')
    medianlt97 = [median.get_ydata()[0] for n,median in enumerate(det97['medians'])]
    axlt97 = detection.plot(np.arange(1,n+2), medianlt97, linewidth=1, color='magenta', label=r'$c_\operatorname{97}$')
    detection.axis([0, n, 0, 100])
    detection.set_xticks(np.arange(0,n+1,10))
    detection.set_title(r'SNR 20')
    detection.legend(loc='lower right')
    
    # plotting data from hausdorff metric SNR 20
    methaus = fig.add_subplot(3, 4, 6)
    hausch = methaus.boxplot(1-hc[1,:,:]/100.) 
    plt.setp(hausch['medians'], color='cyan')
    plt.setp(hausch['caps'], color='cyan')
    plt.setp(hausch['boxes'], color='cyan')
    plt.setp(hausch['fliers'], color='cyan')
    plt.setp(hausch['whiskers'], color='cyan')
    medianhc = [median.get_ydata()[0] for n,median in enumerate(hausch['medians'])]
    axhc = methaus.plot(np.arange(1,n+2), medianhc, linewidth=1, label=r'$1-d_H^c$', color='cyan')
    hausfs = methaus.boxplot(1-hfs[1,:,:]/100.) 
    plt.setp(hausfs['medians'], color='yellow')
    plt.setp(hausfs['caps'], color='yellow')
    plt.setp(hausfs['boxes'], color='yellow')
    plt.setp(hausfs['fliers'], color='yellow')
    plt.setp(hausfs['whiskers'], color='yellow')
    medianhfs = [median.get_ydata()[0] for n,median in enumerate(hausfs['medians'])]
    axhfs = methaus.plot(np.arange(1,n+2), medianhfs, linewidth=1, label=r'$1-d_H^{fs}$', color='yellow')
    methaus.axis([0, n, 0, 1])
    methaus.set_xticks(np.arange(0,n+1,10))
    methaus.legend(loc='lower right')
    methaus.set_title(r'SNR 20')

    # plotting data from wasserstein metric SNR 20
    metwass = fig.add_subplot(3, 4, 7)
    wassch = metwass.boxplot(1-wc[1,:,:]/100.) 
    plt.setp(wassch['medians'], color='red')
    plt.setp(wassch['caps'], color='red')
    plt.setp(wassch['boxes'], color='red')
    plt.setp(wassch['fliers'], color='red')
    plt.setp(wassch['whiskers'], color='red')
    medianwc = [median.get_ydata()[0] for n,median in enumerate(wassch['medians'])]
    axwc = metwass.plot(np.arange(1,n+2), medianwc, linewidth=1, label=r'$1-d_W^c$', color='red')
    wassfs = metwass.boxplot(1-hfs[1,:,:]/100.) 
    plt.setp(wassfs['medians'], color='blue')
    plt.setp(wassfs['caps'], color='blue')
    plt.setp(wassfs['boxes'], color='blue')
    plt.setp(wassfs['fliers'], color='blue')
    plt.setp(wassfs['whiskers'], color='blue')
    medianwfs = [median.get_ydata()[0] for n,median in enumerate(wassfs['medians'])]
    axwfs = metwass.plot(np.arange(1,n+2), medianwfs, linewidth=1, label=r'$1-d_W^{fs}$', color='blue')
    metwass.axis([0, n, 0, 1])
    metwass.set_xticks(np.arange(0,n+1,10))
    metwass.legend(loc='lower right')
    metwass.set_title(r'SNR 20')
    
    # plotting data from Beta SNR 20
    metbeta = fig.add_subplot(3, 4, 8)
    betad = metbeta.boxplot(1-bd[1,:,:]/100.) 
    plt.setp(betad['medians'], color='black')
    plt.setp(betad['caps'], color='black')
    plt.setp(betad['boxes'], color='black')
    plt.setp(betad['fliers'], color='black')
    plt.setp(betad['whiskers'], color='black')
    medianbd = [median.get_ydata()[0] for n,median in enumerate(betad['medians'])]
    axbd = metbeta.plot(np.arange(1,n+2), medianbd, linewidth=1, label=r'$1-d_\beta$', color='black')
    metbeta.axis([0, n, 0, 1])
    metbeta.set_xticks(np.arange(0,n+1,10))
    metbeta.legend(loc='lower right')
    metbeta.set_title(r'SNR 20')

           

def callback_recovery(loc):
    d = loc['dict_obj']
    d.wc.append(emd(loc['dictionary'], d.generating_dict, 
                    'chordal', scale=False))
    d.wfs.append(emd(loc['dictionary'], d.generating_dict, 
                     'fubinistudy', scale=False))
    d.hc.append(hausdorff(loc['dictionary'], d.generating_dict, 
                          'chordal', scale=False))
    d.hfs.append(hausdorff(loc['dictionary'], d.generating_dict, 
                           'fubinistudy', scale=False))
    d.bd.append(betaDist(d.generating_dict, loc['dictionary']))
    d.dr99.append(detectionRate(loc['dictionary'],
                                d.generating_dict, 0.99))
    d.dr97.append(detectionRate(loc['dictionary'],
                                d.generating_dict, 0.97))

rng_global = np.random.RandomState(1)
n_samples, n_dims, n_kernels = 1500, 1, 50
n_features = kernel_init_len = 20
n_nonzero_coefs, learning_rate = 3, 1.5
n_experiments, n_iter = 15, 25
snr = [30, 20, 10]
n_snr = len(snr)
n_jobs, batch_size = -1, 10

if exists("expe_reco.pck"):
    with open("expe_reco.pck", "r") as f:
        o = pickle.load(f)
    wc, wfs, hc, hfs = o['wc'], o['wfs'], o['hc'], o['hfs']
    bd, dr99, dr97 = o['bd'], o['dr99'], o['dr97']
    # plot_recov(wc, wfs, hc, hfs, bd, dr99, dr97, n_iter, "univariate_recov")
else:
    wc = zeros((n_snr, n_experiments, n_iter))
    wfs = zeros((n_snr, n_experiments, n_iter))
    hc = zeros((n_snr, n_experiments, n_iter))
    hfs = zeros((n_snr, n_experiments, n_iter))
    bd = zeros((n_snr, n_experiments, n_iter))
    dr99 = zeros((n_snr, n_experiments, n_iter))
    dr97 = zeros((n_snr, n_experiments, n_iter))

    for i, s in enumerate(snr):
        for e in range(n_experiments):
            g, X, code = _generate_testbed(kernel_init_len,
                n_nonzero_coefs, n_kernels, n_samples, n_features,
                n_dims, s)
            d = MiniBatchMultivariateDictLearning(n_kernels=n_kernels, 
                batch_size=batch_size, n_iter=n_iter,
                n_nonzero_coefs=n_nonzero_coefs, callback=callback_recovery,
                n_jobs=n_jobs, learning_rate=learning_rate,
                kernel_init_len=kernel_init_len, verbose=1,
                random_state=rng_global)
            d.generating_dict = list(g)
            d.wc, d.wfs, d.hc, d.hfs = list(), list(), list(), list()
            d.bd, d.dr99, d.dr97 = list(), list(), list()
            print ('\nExperiment', e+1, 'on', n_experiments)
            d = d.fit(X)
            wc[i, e, :] = array(d.wc); wfs[i, e, :] = array(d.wfs)
            hc[i, e, :] = array(d.hc); hfs[i, e, :] = array(d.hfs)
            dr99[i, e, :] = array(d.dr99); dr97[i, e, :] = array(d.dr97)
            bd[i, e,:] = array(d.bd)
    with open("expe_reco.pck", "w") as f:
        o = {'wc':wc, 'wfs':wfs, 'hc':hc, 'hfs':hfs, 'bd':bd, 'dr99':dr99, 'dr97':dr97}
        pickle.dump(o, f)
    # plot_recov(wc, wfs, hc, hfs, bd, dr99, dr97, n_iter, "univariate_recov")
        
