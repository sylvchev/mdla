"""Dictionary recovering experiment for multivariate random dataset"""
from __future__ import print_function
import numpy as np
from mdla import MultivariateDictLearning, MiniBatchMultivariateDictLearning
from mdla import multivariate_sparse_encode
from dict_metrics import hausdorff, emd, detectionRate, betaDist
from numpy.linalg import norm
from numpy import array, arange, zeros, min, max
from numpy.random import rand, randn, permutation, randint, RandomState
import pickle
from os.path import exists

import os
display = os.environ.get('DISPLAY')
if display is None:
    # if launched from a screen
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors

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
        s = zeros(shape=(n_features, n_dims))
        d = zeros(shape=(n_nonzero_coefs, 3))
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
    signals = array(signals)

    return dico, signals, decomposition

def plot_boxes(fig, data, color='blue', n_iter=100, label=""):
    bp = fig.boxplot(data)
    plt.setp(bp['medians'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['fliers'], color=color)
    plt.setp(bp['whiskers'], color=color)
    med = [m.get_ydata()[0] for n, m in enumerate(bp['medians'])]
    ax = fig.plot(arange(1, n_iter+1), med, linewidth=1, color=color,
                                label=label)

def plot_recov_all(wc, wfs, wcpa, wbc, wg, wfb, hc, hfs, hcpa,
                   hbc, hg, hfb, dr99, dr97, n_iter, figname):
    snr = ['30', '20', '10']
    fig = plt.figure(figsize=(18,10))
    for i, s in enumerate(snr):
        # plotting data from detection rate
        detection = fig.add_subplot(3, 3, i*3+1)
        plot_boxes(detection, dr99[i,:,:]/100., 'green',
                   n_iter, r'$c_\operatorname{99}$')
        plot_boxes(detection, dr97[i,:,:]/100., 'magenta',
                   n_iter, r'$c_\operatorname{97}$')
        detection.axis([0, n_iter, 0, 1])
        detection.set_xticks(arange(0, n_iter+1, 10))
        detection.set_xticklabels([])
        detection.legend(loc='lower right')

        methaus = fig.add_subplot(3, 3, i*3+2)
        plot_boxes(methaus, 1-hc[i,:,:], 'chartreuse',
                   n_iter, r'$1-d_H^c$')
        plot_boxes(methaus, 1-hcpa[i,:,:], 'red',
                   n_iter, r'$1-d_H^{cpa}$')
        plot_boxes(methaus, 1-hfs[i,:,:], 'magenta',
                   n_iter, r'$1-d_H^{fs}$')
        plot_boxes(methaus, 1-hbc[i,:,:], 'blue',
                   n_iter, r'$1-d_H^{bc}$')
        plot_boxes(methaus, 1-hg[i,:,:], 'deepskyblue',
                   n_iter, r'$1-d_H^{g}$')
        plot_boxes(methaus, 1-hfb[i,:,:], 'orange',
                   n_iter, r'$1-d_H^{fb}$')
        
        methaus.axis([0, n_iter, 0, 1])
        methaus.set_xticks(arange(0, n_iter+1, 10))
        methaus.set_xticklabels([])
        methaus.set_yticklabels([])
        methaus.legend(loc='lower right')

        metwass = fig.add_subplot(3, 3, i*3+3)
        plot_boxes(metwass, 1-wc[i,:,:], 'chartreuse',
                   n_iter, r'$1-d_W^c$')
        plot_boxes(metwass, 1-wcpa[i,:,:], 'red',
                   n_iter, r'$1-d_W^{cpa}$')
        plot_boxes(metwass, 1-wfs[i,:,:], 'magenta',
                   n_iter, r'$1-d_W^{fs}$')
        plot_boxes(metwass, 1-wbc[i,:,:], 'blue',
                   n_iter, r'$1-d_W^{bc}$')
        plot_boxes(metwass, 1-wg[i,:,:], 'deepskyblue',
                   n_iter, r'$1-d_W^{g}$')
        plot_boxes(metwass, 1-wfb[i,:,:], 'orange',
                   n_iter, r'$1-d_W^{fb}$')
        metwass.axis([0, n_iter, 0, 1])
        metwass.set_xticks(arange(0, n_iter+1, 10))
        metwass.set_xticklabels([])
        metwass.set_yticklabels([])
        metwass.legend(loc='lower right')
        metwass.set_title(' ')

        metwass.annotate('SNR '+s, xy=(.51, 1.-i*1./3.+i*0.01-0.001),
                xycoords='figure fraction',
                horizontalalignment='center', verticalalignment='top',
                fontsize='large')

    detection.set_xticks(arange(0, n_iter+1, 10))
    detection.set_xticklabels(arange(0, n_iter+1, 10))
    methaus.set_xticks(arange(0, n_iter+1, 10))
    methaus.set_xticklabels(arange(0, n_iter+1, 10))
    metwass.set_xticks(arange(0, n_iter+1, 10))
    metwass.set_xticklabels(arange(0, n_iter+1, 10))
    plt.tight_layout(1.2)
    plt.savefig(figname+".png")                        
        
def plot_recov(wc, wfs, hc, hfs, dr99, dr97, n_iter, figname):
    snr = ['30', '20', '10']
    fig = plt.figure(figsize=(18,10))
    for i, s in enumerate(snr):
        # plotting data from detection rate
        detection = fig.add_subplot(3, 3, i*3+1)
        plot_boxes(detection, dr99[i,:,:]/100., 'green',
                   n_iter, r'$c_\operatorname{99}$')
        plot_boxes(detection, dr97[i,:,:]/100., 'magenta',
                   n_iter, r'$c_\operatorname{97}$')
        detection.axis([0, n_iter, 0, 1])
        detection.set_xticks(arange(0, n_iter+1, 10))
        detection.set_xticklabels([])
        detection.legend(loc='lower right')
    
        # plotting data from hausdorff metric
        methaus = fig.add_subplot(3, 3, i*3+2)
        plot_boxes(methaus, 1-hc[i,:,:], 'cyan',
                   n_iter, r'$1-d_H^c$')
        plot_boxes(methaus, 1-hfs[i,:,:], 'yellow',
                   n_iter, r'$1-d_H^{fs}$')
        methaus.axis([0, n_iter, 0, 1])
        methaus.set_xticks(arange(0, n_iter+1, 10))
        methaus.set_xticklabels([])
        methaus.set_yticklabels([])
        methaus.legend(loc='lower right')

        # plotting data from wasserstein metric
        metwass = fig.add_subplot(3, 3, i*3+3)
        plot_boxes(metwass, 1-wc[i,:,:], 'red',
                   n_iter, r'$1-d_W^c$')
        plot_boxes(metwass, 1-wfs[i,:,:], 'blue',
                   n_iter, r'$1-d_W^{fs}$')
        metwass.axis([0, n_iter, 0, 1])
        metwass.set_xticks(arange(0, n_iter+1, 10))
        metwass.set_xticklabels([])
        metwass.set_yticklabels([])
        metwass.legend(loc='lower right')
        metwass.set_title(' ')

        metwass.annotate('SNR '+s, xy=(.51, 1.-i*1./3.+i*0.01-0.001),
                xycoords='figure fraction',
                horizontalalignment='center', verticalalignment='top',
                fontsize='large')

    detection.set_xticks(arange(0, n_iter+1, 10))
    detection.set_xticklabels(arange(0, n_iter+1, 10))
    methaus.set_xticks(arange(0, n_iter+1, 10))
    methaus.set_xticklabels(arange(0, n_iter+1, 10))
    metwass.set_xticks(arange(0, n_iter+1, 10))
    metwass.set_xticklabels(arange(0, n_iter+1, 10))
    plt.tight_layout(1.2)
    plt.savefig(figname+".png")

def callback_recovery(loc):
    d = loc['dict_obj']
    d.wc.append(emd(loc['dictionary'], d.generating_dict, 
                    'chordal', scale=True))
    d.wfs.append(emd(loc['dictionary'], d.generating_dict, 
                     'fubinistudy', scale=True))
    d.wcpa.append(emd(loc['dictionary'], d.generating_dict, 
                     'chordalPA', scale=True))
    d.wbc.append(emd(loc['dictionary'], d.generating_dict, 
                     'binetcauchy', scale=True))
    d.wg.append(emd(loc['dictionary'], d.generating_dict, 
                     'geodesic', scale=True))
    d.wfb.append(emd(loc['dictionary'], d.generating_dict, 
                     'frobeniusBased', scale=True))
    d.hc.append(hausdorff(loc['dictionary'], d.generating_dict, 
                          'chordal', scale=True))
    d.hfs.append(hausdorff(loc['dictionary'], d.generating_dict, 
                           'fubinistudy', scale=True))
    d.hcpa.append(hausdorff(loc['dictionary'], d.generating_dict, 
                           'chordalPA', scale=True))
    d.hbc.append(hausdorff(loc['dictionary'], d.generating_dict, 
                           'binetcauchy', scale=True))
    d.hg.append(hausdorff(loc['dictionary'], d.generating_dict, 
                           'geodesic', scale=True))
    d.hfb.append(hausdorff(loc['dictionary'], d.generating_dict, 
                           'frobeniusBased', scale=True))
    d.dr99.append(detectionRate(loc['dictionary'],
                                d.generating_dict, 0.99))
    d.dr97.append(detectionRate(loc['dictionary'],
                                d.generating_dict, 0.97))

rng_global = RandomState(1)
n_samples, n_dims, n_kernels = 1500, 5, 50
n_features = kernel_init_len = 20
n_nonzero_coefs, learning_rate = 3, 1.5
n_experiments, n_iter = 15, 25
snr = [30, 20, 10]
n_snr = len(snr)
n_jobs, batch_size = -1, 60

backup_fname = "expe_multi_reco_all.pck"

if exists(backup_fname):
    with open(backup_fname, "r") as f:
        o = pickle.load(f)
    wc, wfs, hc, hfs = o['wc'], o['wfs'], o['hc'], o['hfs']
    wcpa, wbc, wg, wfb = o['wcpa'], o['wbc'], o['wg'], o['wfb']
    hcpa, hbc, hg, hfb = o['hcpa'], o['hbc'], o['hg'], o['hfb']
    dr99, dr97 = o['dr99'], o['dr97']
    plot_recov(wc, wfs, hc, hfs, dr99, dr97, n_iter, "multivariate_recov")
else:
    wc = zeros((n_snr, n_experiments, n_iter))
    wfs = zeros((n_snr, n_experiments, n_iter))
    wcpa = zeros((n_snr, n_experiments, n_iter))
    wbc = zeros((n_snr, n_experiments, n_iter))
    wg = zeros((n_snr, n_experiments, n_iter))
    wfb = zeros((n_snr, n_experiments, n_iter))
    hc = zeros((n_snr, n_experiments, n_iter))
    hfs = zeros((n_snr, n_experiments, n_iter))
    hcpa = zeros((n_snr, n_experiments, n_iter))
    hbc = zeros((n_snr, n_experiments, n_iter))
    hg = zeros((n_snr, n_experiments, n_iter))
    hfb = zeros((n_snr, n_experiments, n_iter))
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
            d.wcpa, d.wbc, d.wg, d.wfb = list(), list(), list(), list()
            d.hcpa, d.hbc, d.hg, d.hfb = list(), list(), list(), list()
            d.dr99, d.dr97 = list(), list()
            print ('\nExperiment', e+1, 'on', n_experiments)
            d = d.fit(X)
            wc[i, e, :] = array(d.wc); wfs[i, e, :] = array(d.wfs)
            hc[i, e, :] = array(d.hc); hfs[i, e, :] = array(d.hfs)
            wcpa[i, e, :] = array(d.wcpa); wbc[i, e, :] = array(d.wbc)
            wg[i, e, :] = array(d.wg); wfb[i, e, :] = array(d.wfb)
            hcpa[i, e, :] = array(d.hcpa); hbc[i, e, :] = array(d.hbc)
            hg[i, e, :] = array(d.hg); hfb[i, e, :] = array(d.hfb)
            dr99[i, e, :] = array(d.dr99); dr97[i, e, :] = array(d.dr97)
    with open(backup_fname, "w") as f:
        o = {'wc':wc, 'wfs':wfs, 'hc':hc, 'hfs':hfs, 'dr99':dr99, 'dr97':dr97,
             'wcpa':wcpa, 'wbc':wbc, 'wg':wg, 'wfb':wfb, 'hcpa':hcpa,
             'hbc':hbc, 'hg':hg, 'hfb':hfb}
        pickle.dump(o, f)
    # plot_recov(wc, wfs, hc, hfs, dr99, dr97, n_iter, "multivariate_recov")
    plot_recov_all(wc, wfs, wcpa, wbc, wg, wfb, hc, hfs, hcpa, hbc, hg, hfb, dr99, dr97, n_iter, "multivariate_recov_all")
        
