"""Benchmarking dictionary learning algorithms on random dataset"""

from multiprocessing import cpu_count
from time import time

import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from numpy.linalg import norm
from numpy.random import permutation, rand, randint, randn

from mdla import MiniBatchMultivariateDictLearning, MultivariateDictLearning


# TODO:
# investigate perf break from pydico


def benchmarking_plot(figname, pst, plot_sep, minibatchRange, mprocessRange):
    _ = plt.figure(figsize=(15, 10))
    bar_width = 0.35
    _ = plt.bar(
        np.array([0]),
        pst[0],
        bar_width,
        color="b",
        label="Online, no multiprocessing (baseline)",
    )
    index = [0]
    for i in range(1, plot_sep[1]):
        if i == 1:
            _ = plt.bar(
                np.array([i + 1]),
                pst[i],
                bar_width,
                color="r",
                label="Online with minibatch",
            )
        else:
            _ = plt.bar(np.array([i + 1]), pst[i], bar_width, color="r")
        index.append(i + 1)
    for _ in range(plot_sep[1], plot_sep[2]):
        if i == plot_sep[1]:
            _ = plt.bar(
                np.array([i + 2]),
                pst[i],
                bar_width,
                label="Batch with multiprocessing",
                color="magenta",
            )
        else:
            _ = plt.bar(np.array([i + 2]), pst[i], bar_width, color="magenta")
        index.append(i + 2)

    plt.ylabel("Time per iteration (s)")
    plt.title("Processing time for online and batch processing")
    tick = [""]
    tick.extend(map(str, minibatchRange))
    tick.extend(map(str, mprocessRange))
    plt.xticks(index, tuple(tick))
    plt.legend()
    plt.savefig(figname + ".png")


def _generate_testbed(
    kernel_init_len,
    n_nonzero_coefs,
    n_kernels,
    n_samples=10,
    n_features=5,
    n_dims=3,
    snr=1000,
):
    """Generate a dataset from a random dictionary

    Generate a random dictionary and a dataset, where samples are combination of
    n_nonzero_coefs dictionary atoms. Noise is added, based on SNR value, with
    1000 indicated that no noise should be added.
    Return the dictionary, the dataset and an array indicated how atoms are combined
    to obtain each sample
    """
    print("Dictionary sampled from uniform distribution")
    dico = [rand(kernel_init_len, n_dims) for i in range(n_kernels)]
    for i in range(len(dico)):
        dico[i] /= norm(dico[i], "fro")

    signals = list()
    decomposition = list()
    for _ in range(n_samples):
        s = np.zeros(shape=(n_features, n_dims))
        d = np.zeros(shape=(n_nonzero_coefs, 3))
        rk = permutation(range(n_kernels))
        for j in range(n_nonzero_coefs):
            k_idx = rk[j]
            k_amplitude = 3.0 * rand() + 1.0
            k_offset = randint(n_features - kernel_init_len + 1)
            s[k_offset : k_offset + kernel_init_len, :] += k_amplitude * dico[k_idx]
            d[j, :] = array([k_amplitude, k_offset, k_idx])
        decomposition.append(d)
        noise = randn(n_features, n_dims)
        if snr == 1000:
            alpha = 0
        else:
            ps = norm(s, "fro")
            pn = norm(noise, "fro")
            alpha = ps / (pn * 10 ** (snr / 20.0))
        signals.append(s + alpha * noise)
    signals = np.array(signals)

    return dico, signals, decomposition


rng_global = np.random.RandomState(1)
n_samples, n_dims = 1500, 1
n_features = kernel_init_len = 5
n_nonzero_coefs = 3
n_kernels, max_iter, learning_rate = 50, 10, 1.5
n_jobs, batch_size = -1, None

iter_time, plot_separator, it_separator = list(), list(), 0

generating_dict, X, code = _generate_testbed(
    kernel_init_len, n_nonzero_coefs, n_kernels, n_samples, n_features, n_dims
)

# Online without mini-batch
print(
    "Processing ",
    max_iter,
    "iterations in online mode, " "without multiprocessing:",
    end="",
)
batch_size, n_jobs = n_samples, 1
learned_dict = MiniBatchMultivariateDictLearning(
    n_kernels=n_kernels,
    batch_size=batch_size,
    n_iter=max_iter,
    n_nonzero_coefs=n_nonzero_coefs,
    n_jobs=n_jobs,
    learning_rate=learning_rate,
    kernel_init_len=kernel_init_len,
    verbose=1,
    dict_init=None,
    random_state=rng_global,
)
ts = time()
learned_dict = learned_dict.fit(X)
iter_time.append((time() - ts) / max_iter)
it_separator += 1
plot_separator.append(it_separator)

# Online with mini-batch
minibatch_range = [cpu_count()]
minibatch_range.extend([cpu_count() * i for i in range(3, 10, 2)])
n_jobs = -1
for mb in minibatch_range:
    print(
        "\nProcessing ",
        max_iter,
        "iterations in online mode, with ",
        "minibatch size",
        mb,
        "and",
        cpu_count(),
        "processes:",
        end="",
    )
    batch_size = mb
    learned_dict = MiniBatchMultivariateDictLearning(
        n_kernels=n_kernels,
        batch_size=batch_size,
        n_iter=max_iter,
        n_nonzero_coefs=n_nonzero_coefs,
        n_jobs=n_jobs,
        learning_rate=learning_rate,
        kernel_init_len=kernel_init_len,
        verbose=1,
        dict_init=None,
        random_state=rng_global,
    )
    ts = time()
    learned_dict = learned_dict.fit(X)
    iter_time.append((time() - ts) / max_iter)
    it_separator += 1
plot_separator.append(it_separator)

# Batch learning
mp_range = range(1, cpu_count() + 1)
for p in mp_range:
    print(
        "\nProcessing ",
        max_iter,
        "iterations in batch mode, with",
        p,
        "processes:",
        end="",
    )
    n_jobs = p
    learned_dict = MultivariateDictLearning(
        n_kernels=n_kernels,
        max_iter=max_iter,
        verbose=1,
        n_nonzero_coefs=n_nonzero_coefs,
        n_jobs=n_jobs,
        learning_rate=learning_rate,
        kernel_init_len=kernel_init_len,
        dict_init=None,
        random_state=rng_global,
    )
    ts = time()
    learned_dict = learned_dict.fit(X)
    iter_time.append((time() - ts) / max_iter)
    it_separator += 1
plot_separator.append(it_separator)
print("Done benchmarking")

figname = "minibatch-performance"
print("Plotting results in", figname)
benchmarking_plot(figname, iter_time, plot_separator, minibatch_range, mp_range)

print("Exiting.")
