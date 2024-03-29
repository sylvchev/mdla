"""Learning dictionary on BCI Competition IV-2a dataset"""
import pickle
from os import listdir
from os.path import exists

from numpy import array, exp, int, nan_to_num, pi, poly, real, zeros_like
from numpy.random import RandomState
from plot_bci_dict import plot_atom_usage, plot_objective_func
from scipy.io import loadmat
from scipy.signal import butter, decimate, filtfilt

from mdla import MiniBatchMultivariateDictLearning


def notch(Wn, notchWidth):
    # Compute zeros
    nzeros = array([exp(1j * pi * Wn), exp(-1j * pi * Wn)])
    # Compute poles
    poles = (1 - notchWidth) * nzeros
    b = poly(nzeros)  # Get moving average filter coefficients
    a = poly(poles)  # Get autoregressive filter coefficients
    return b, a


def read_BCI_signals():
    kppath = "../../datasets/BCIcompetition4/"
    lkp = listdir(kppath)
    sujets = list()
    classes = list()
    signals = list()

    preprocessing = True
    decimation = True

    if preprocessing:
        # Notch filtering
        f0 = 50.0  # notch frequency
        notchWidth = 0.1  # width of the notch
        # Bandpass filtering
        order = 8
        fc = array([8.0, 30.0])  # [0.1, 100]
        sr = 250  # sampling rate, o['SampleRate'][0,0] from loadmat
        Wn = f0 / (sr / 2.0)  # ratio of notch freq. to Nyquist freq.
        [bn, an] = notch(Wn, notchWidth)
        [bb, ab] = butter(order, fc / (sr / 2.0), "bandpass")
    else:
        f0 = -1.0
        notchWidth = 0.0
        order = 0
        fc = 0

    if decimation:
        dfactor = 2.0
    else:
        dfactor = 1.0

    fn = (
        "bcicompdata"
        + str(
            hash(
                str(preprocessing)
                + str(f0)
                + str(notchWidth)
                + str(order)
                + str(fc)
                + str(decimation)
                + str(dfactor)
            )
        )
        + ".pickle"
    )

    if exists(fn):
        with open(fn, "rb") as f:
            o = pickle.load(f)
            signals = o["signals"]
            classes = o["classes"]
        print("Previous preprocessing of BCI dataset found, reusing it")
    else:
        for item in lkp:
            if item[-8:] == "-EOG.mat":
                print("loading", item)
                o = loadmat(kppath + item, struct_as_record=True)
                s = nan_to_num(o["s"])
                # sample_rate = o['SampleRate']
                event_type = o["EVENTTYP"]
                event_pos = o["EVENTPOS"]
                class_label = o["Classlabel"]
                if preprocessing:
                    # Use a Notch filter to remove 50Hz power line
                    ns = zeros_like(s)
                    for e in range(s.shape[1]):
                        ns[:, e] = filtfilt(bn, an, s[:, e])
                    # Apply a bandpass filter
                    fs = zeros_like(s)
                    for e in range(s.shape[1]):
                        fs[:, e] = filtfilt(real(bb), real(ab), ns[:, e])
                    # decimate the signal
                    if decimation:
                        fs = decimate(fs, int(dfactor), axis=0, zero_phase=True)

                # Event Type
                trial_begin = 768

                start = 3 * sr / dfactor  # 2s fixation, 1s after cue
                stop = 6 * sr / dfactor  # 4s after cue, 3s of EEG

                trials = event_pos[event_type == trial_begin]
                for i, t in enumerate(trials):
                    tmpfs = fs[int(t / dfactor + start) : int(t / dfactor + stop), 0:22]
                    signals.append((tmpfs - tmpfs.mean(axis=0)))  # center data
                    sujets.append(item[2:3])
                    classes.append(class_label[i])

        with open(fn, "wb") as f:
            o = {"signals": signals, "classes": classes}
            pickle.dump(o, f)
    return signals, classes


X, classes = read_BCI_signals()

rng_global = RandomState(1)
n_samples = len(X)
n_dims = X[0].shape[0]  # 22 electrodes
n_features = X[0].shape[1]  # 375, 3s of decimated signal at 125Hz
kernel_init_len = 80  # kernel size is 50
n_kernels = 60
n_nonzero_coefs = 2
learning_rate = 5.0
n_iter = 40  # 100
n_jobs, batch_size = -1, None  # n_cpu, 5*n_cpu
figname = "-60ker-K3-klen80-lr5.0-emm-all"

d = MiniBatchMultivariateDictLearning(
    n_kernels=n_kernels,
    batch_size=batch_size,
    n_iter=n_iter,
    n_nonzero_coefs=n_nonzero_coefs,
    n_jobs=n_jobs,
    learning_rate=learning_rate,
    kernel_init_len=kernel_init_len,
    verbose=1,
    random_state=rng_global,
)
d = d.fit(X)

plot_objective_func(d.error_, n_iter, figname)

n_jobs = 4
plot_atom_usage(X, d.kernels_, n_nonzero_coefs, n_jobs, figname)

with open("EEG-savedico" + figname + ".pkl", "wb") as f:
    o = {
        "kernels": d.kernels_,
        "error": d.error_,
        "kernel_init_len": d.kernel_init_len,
        "learning_rate": d.learning_rate,
        "n_iter": d.n_iter,
        "n_jobs": d.n_jobs,
        "n_kernels": d.n_kernels,
        "n_nonzero_coefs": d.n_nonzero_coefs,
    }
    pickle.dump(o, f)
