"""Learning dictionary on BCI Competition dataset"""
from __future__ import print_function

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mdla import MiniBatchMultivariateDictLearning
from dict_metrics import hausdorff, emd, detectionRate
from numpy.linalg import norm
from numpy import array, arange, zeros, zeros_like, min, max, int,real, exp, \
pi, poly, nan_to_num, histogram, hstack
from numpy.random import rand, randn, permutation, randint, RandomState
from scipy.signal import filtfilt, butter, decimate
from scipy.io import loadmat
from os import listdir
from os.path import exists

def notch(Wn, notchWidth):
    # Compute zeros
    nzeros = array([exp(1j*pi*Wn), exp(-1j*pi*Wn)])
    #Compute poles
    poles = (1-notchWidth) * nzeros
    b = poly(nzeros)    # Get moving average filter coefficients
    a = poly(poles)    # Get autoregressive filter coefficients
    return b, a

def read_BCI_signals():
    kppath = '../../datasets/BCIcompetition4/'
    lkp = listdir(kppath)
    sujets = list()
    classes = list()
    signals = list()

    preprocessing = True
    decimation = True

    if preprocessing:
        # Notch filtering
        f0 = 50.              #notch frequency
        notchWidth = 0.1      #width of the notch        
        # Bandpass filtering
        order = 8
        fc = [8., 30.] # [0.1, 100]
    else:
        f0 = -1.
        notchWidth = 0.
        order = 0
        fc = 0
        
    if decimation:
        dfactor = 2.
    else:
        dfactor = 1.
        
    fn = 'bcicompdata'+str(hash(str(preprocessing)+str(f0)+str(notchWidth)+str(order)+str(fc)+str(decimation)+str(dfactor)))+'.pickle'
    
    if exists(fn):
        with open(fn,'r') as f:
            o = pickle.load(f)
            signals = o['signals']
            f.close()
        print ('Previous preprocessing of BCI dataset found, reusing it')
    else:    
        for item in lkp:
            # on prend tous les training, il faut utiliser les labels pour E.
            if item[-9:] == 'T-EOG.mat':
                o = loadmat (kppath+item, struct_as_record=True)
                s = nan_to_num(o['s'])
                SampleRate = o['SampleRate']    
                EVENTTYP = o['EVENTTYP']
                EVENTPOS = o['EVENTPOS']
                EVENTDUR = o['EVENTDUR']
                # Label = o['Label']
                # Classlabel = o['Classlabel']
                if preprocessing:
                    # Use a Notch filter to remove 50Hz power line
                    fs = SampleRate[0,0]  #sampling rate
                    Wn = f0/(fs/2.)       #ratio of notch freq. to Nyquist freq.
                    # notchWidth = 0.1      #width of the notch
                    [b, a] = notch(Wn, notchWidth)
                    ns = zeros_like(s)
                    for e in range(s.shape[1]):
                        ns[:,e] = filtfilt(b, a, s[:,e])
                
                    # Apply a bandpass filter
                    fs = SampleRate[0,0]  #sampling rate
                    [b, a] = butter(order, fc/(fs/2.), 'bandpass')
                    fs = zeros_like(s)
                    for e in range(s.shape[1]):
                        fs[:,e] = filtfilt(real(b), real(a), ns[:,e])

                    # decimate the signal
                    if decimation:
                        dfs = decimate(fs, int(dfactor), axis=0)
                    
                # Choose the class to process
                LeftHand  = 769
                RightHand = 770
                Foot      = 771
                Tongue    = 772
                start = 1*SampleRate[0,0]/dfactor
                stop  = 4*SampleRate[0,0]/dfactor

                # ev = np.concatenate((EVENTPOS[EVENTTYP==LeftHand], EVENTPOS[EVENTTYP==RightHand]))
                ev = EVENTPOS[EVENTTYP==LeftHand] # Only left hand movement
                for i, t in enumerate(ev):
                    tmpfs = fs[t/dfactor+start:t/dfactor+stop,0:22]
                    # center the data
                    signals.append((tmpfs-tmpfs.mean(axis=0)))
                    sujets.append(item[2:3])
                    classes.append('LeftHand')
                ev = EVENTPOS[EVENTTYP==RightHand] # Only left hand movement
                for i, t in enumerate(ev):
                    tmpfs = fs[t/dfactor+start:t/dfactor+stop,0:22]
                    # center the data
                    signals.append((tmpfs-tmpfs.mean(axis=0)))
                    sujets.append(item[2:3])
                    classes.append('RightHand')
                ev = EVENTPOS[EVENTTYP==Foot] # Only left hand movement
                for i, t in enumerate(ev):
                    tmpfs = fs[t/dfactor+start:t/dfactor+stop,0:22]
                    # center the data
                    signals.append((tmpfs-tmpfs.mean(axis=0)))
                    sujets.append(item[2:3])
                    classes.append('Foot')
                ev = EVENTPOS[EVENTTYP==Tongue] # Only left hand movement
                for i, t in enumerate(ev):
                    tmpfs = fs[t/dfactor+start:t/dfactor+stop,0:22]
                    # center the data
                    signals.append((tmpfs-tmpfs.mean(axis=0)))
                    sujets.append(item[2:3])
                    classes.append('Tongue')
                
    with open(fn, 'w+') as f:
        o = {'signals':signals}
        pickle.dump(o,f)
    return signals

def saveKernelPlot(kernels, n_kernels, col = 5, row = -1, order=None, figname = 'allkernels', label=None):
    n_display = idx = 0
    if n_kernels == row*col:
        pass
    elif row == -1:
        row = n_kernels / int(col)
        if n_kernels % int(col) != 0:
            row += 1
    elif col == -1:
        col = n_kernels / int(row)
        if n_kernels % int(row) != 0:
            col += 1
    n_display = row*col
    n_figure = n_kernels / int(n_display)
    if n_kernels % int(n_display) != 0:
        n_figure += 1
    if order == None:
        order = range(n_kernels)
    if label == None:
        label = range(n_kernels)

    for j in range(n_figure):
        fig = plt.figure(figsize=(15,10))
        for i in range(1, n_display+1):
            if idx+i > n_kernels:
                break
            k = fig.add_subplot(row, col, i)
            k.plot(kernels[order[-(idx+i)]])
            k.set_xticklabels([])
            k.set_yticklabels([])
            k.set_title('k %d: %d' % (order[-(idx+i)], label[order[-(idx+i)]]))
        idx += n_display
        plt.tight_layout(.5)
        plt.savefig(figname+'-part'+str(j)+'.png')


X = read_BCI_signals()
X_half1 = array([x[0::2,:] for x in X])
X_half2 = array([x[1::2,:] for x in X])

rng_global = RandomState(1)
n_samples = len(X_half1)
n_dims = X_half1[0].shape[0] # half of the 22 electrodes
n_features = X_half1[0].shape[1] # 375, 3s of decimated signal at 125Hz
kernel_init_len = 60 # kernel size is 60
n_kernels = 90 
n_nonzero_coefs = 6
learning_rate = 5.0
n_iter = 100
n_jobs, batch_size = -1, None # n_cpu, 5*n_cpu
figname="-90ker-K6-klen60-lr5.0-emm-all"

d1 = MiniBatchMultivariateDictLearning(n_kernels=n_kernels,
                batch_size=batch_size, n_iter=n_iter,
                n_nonzero_coefs=n_nonzero_coefs, 
                n_jobs=n_jobs, learning_rate=learning_rate,
                kernel_init_len=kernel_init_len, verbose=1,
                random_state=rng_global)
d1 = d1.fit(X_half1)

d2 = MiniBatchMultivariateDictLearning(n_kernels=n_kernels,
                batch_size=batch_size, n_iter=n_iter,
                n_nonzero_coefs=n_nonzero_coefs, 
                n_jobs=n_jobs, learning_rate=learning_rate,
                kernel_init_len=kernel_init_len, verbose=1,
                random_state=rng_global)
d2 = d2.fit(X_half2)

plt.figure()
plt.plot (array(d1.error_))
plt.savefig('EEG-decomposition-error-half1'+figname+'.png')

plt.figure()
plt.plot (array(d2.error_))
plt.savefig('EEG-decomposition-error-half2'+figname+'.png')

from mdla import multivariate_sparse_encode
from collections import Counter
n_jobs = 4
r, code = multivariate_sparse_encode(X_half1, d1.kernels_,
                                     n_nonzero_coefs=n_nonzero_coefs,
                                     n_jobs=n_jobs, verbose=2)

decomposition_weight = hstack([code[i][:,2] for i in range(len(code))])
decomposition_weight.sort()
weight, _ = histogram(decomposition_weight, n_kernels, normed=False)
order = weight.argsort()
saveKernelPlot(d1.kernels_, d1.n_kernels, order=order, label=weight, figname='EEG-kernels-half1'+figname, row=6)

correlation = Counter(decomposition_weight).items()
correlation.sort(key=lambda x: x[1])
labels, values = zip(*correlation)
indexes = arange(len(correlation))

plt.figure()
width = 1
plt.bar(indexes, values, width, linewidth=0)
plt.savefig('EEG-coeff_hist_sorted-half1'+figname+'.png')

r, code = multivariate_sparse_encode(X_half2, d2.kernels_,
                                     n_nonzero_coefs=n_nonzero_coefs,
                                     n_jobs=n_jobs, verbose=2)

decomposition_weight = hstack([code[i][:,2] for i in range(len(code))])
decomposition_weight.sort()
weight, _ = histogram(decomposition_weight, n_kernels, normed=False)
order = weight.argsort()
saveKernelPlot(d2.kernels_, d2.n_kernels, order=order, label=weight, figname='EEG-kernels-half2'+figname, row=6)

correlation = Counter(decomposition_weight).items()
correlation.sort(key=lambda x: x[1])
labels, values = zip(*correlation)
indexes = arange(len(correlation))

plt.figure()
width = 1
plt.bar(indexes, values, width, linewidth=0)
plt.savefig('EEG-coeff_hist_sorted-half2'+figname+'.png')

with open('EEG-savedico-half1'+figname+'.pkl', 'w+') as f:
    o = {'kernels':d1.kernels_, 'error':d1.error_, 'kernel_init_len':d1.kernel_init_len, 'learning_rate':d1.learning_rate, 'n_iter':d1.n_iter, 'n_jobs':d1.n_jobs, 'n_kernels':d1.n_kernels, 'n_nonzero_coefs':d1.n_nonzero_coefs}
    pickle.dump(o,f)

with open('EEG-savedico-half2'+figname+'.pkl', 'w+') as f:
    o = {'kernels':d2.kernels_, 'error':d2.error_, 'kernel_init_len':d2.kernel_init_len, 'learning_rate':d2.learning_rate, 'n_iter':d2.n_iter, 'n_jobs':d2.n_jobs, 'n_kernels':d2.n_kernels, 'n_nonzero_coefs':d2.n_nonzero_coefs}
    pickle.dump(o,f)


