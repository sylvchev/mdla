from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
from numpy import abs, arange, argsort, histogram, hstack, percentile, zeros
from numpy.linalg import norm

from mdla import multivariate_sparse_encode, reconstruct_from_code


matplotlib.use("Agg")

# TODO: use sum of decomposition weight instead of number of atom usage in
#       plot_atom_usage
#       - subplot(3,n*2, ) = X, residual, reconst for n best/worst reconstruction.
#


def plot_kernels(
    kernels,
    n_kernels,
    col=5,
    row=-1,
    order=None,
    amp=None,
    figname="allkernels",
    label=None,
):
    n_display = idx = 0
    if n_kernels == row * col:
        pass
    elif row == -1:
        row = n_kernels / int(col)
        if n_kernels % int(col) != 0:
            row += 1
    elif col == -1:
        col = n_kernels / int(row)
        if n_kernels % int(row) != 0:
            col += 1
    n_display = row * col
    n_figure = int(n_kernels / n_display)
    if n_kernels % int(n_display) != 0:
        n_figure += 1
    if order is None:
        order = range(n_kernels)
    if label is None:
        label = range(n_kernels)
    if amp is None:
        amp = range(n_kernels)

    for j in range(int(n_figure)):
        fig = plt.figure(figsize=(15, 10))
        for i in range(1, n_display + 1):
            if idx + i > n_kernels:
                break
            k = fig.add_subplot(row, col, i)
            k.plot(kernels[order[-(idx + i)]])
            k.set_xticklabels([])
            k.set_yticklabels([])
            k.set_title(
                "k %d: %d-%g"
                % (order[-(idx + i)], label[order[-(idx + i)]], amp[order[-(idx + i)]])
            )
        idx += n_display
        plt.tight_layout(0.5)
        plt.savefig(figname + "-part" + str(j) + ".png")


def plot_reconstruction_samples(X, r, code, kernels, n, figname):
    n_features = X[0].shape[0]
    energy_residual = zeros(len(r))
    for i in range(len(r)):
        energy_residual[i] = norm(r[i], "fro")
    energy_sample = zeros(len(X))
    for i in range(len(X)):
        energy_sample[i] = norm(X[i], "fro")

    energy_explained = energy_residual / energy_sample
    index = argsort(energy_explained)  # 0 =worse, end=best
    fig = plt.figure(figsize=(15, 9))
    k = fig.add_subplot(3, 2 * n, 1)
    k.set_xticklabels([])
    k.set_yticklabels([])
    for i in range(n):
        if i != 0:
            ka = fig.add_subplot(3, 2 * n, i + 1, sharex=k, sharey=k)
        else:
            ka = k
        ka.plot(X[index[i]])
        ka.set_title("s%d: %.1f%%" % (index[i], 100.0 * (1 - energy_explained[index[i]])))
        ka = fig.add_subplot(3, 2 * n, 2 * n + i + 1, sharex=k, sharey=k)
        ka.plot(r[index[i]])
        ka = fig.add_subplot(3, 2 * n, 4 * n + i + 1, sharex=k, sharey=k)
        s = reconstruct_from_code([code[index[i]]], kernels, n_features)
        ka.plot(s[0, :, :])
    for j, i in zip(range(n, 2 * n), range(n, 0, -1)):
        ka = fig.add_subplot(3, 2 * n, j + 1, sharex=k, sharey=k)
        ka.plot(X[index[-i]])
        ka.set_title(
            "s%d: %.1f%%" % (index[-i], 100.0 * (1 - energy_explained[index[-i]]))
        )
        ka = fig.add_subplot(3, 2 * n, 2 * n + j + 1, sharex=k, sharey=k)
        ka.plot(r[index[-i]])
        ka = fig.add_subplot(3, 2 * n, 4 * n + j + 1, sharex=k, sharey=k)
        s = reconstruct_from_code([code[index[-i]]], kernels, n_features)
        ka.plot(s[0, :, :])
    plt.tight_layout(0.5)
    plt.savefig("EEG-reconstruction" + figname + ".png")


def plot_objective_func_box(error, n_iter, figname):
    fig = plt.figure()
    objf = fig.add_subplot(1, 1, 1)
    ofun = objf.boxplot(error.T)
    medianof = [median.get_ydata()[0] for n, median in enumerate(ofun["medians"])]
    _ = objf.plot(arange(1, n_iter + 1), medianof, linewidth=1)
    plt.savefig("EEG-decomposition-error" + figname + ".png")


def plot_objective_func(error, n_iter, figname):
    fig = plt.figure()
    objf = fig.add_subplot(1, 1, 1)
    p0, p25, med, p75, p100 = percentile(error, (0, 25, 50, 75, 100), axis=1)
    objf.fill_between(
        arange(1, n_iter + 1), p0, p100, facecolor="blue", alpha=0.1, interpolate=True
    )
    objf.fill_between(
        arange(1, n_iter + 1), p25, p75, facecolor="blue", alpha=0.3, interpolate=True
    )
    objf.plot(arange(1, n_iter + 1), med, linewidth=2.5, color="blue")
    objf.set_xlabel("Iterations")
    objf.set_ylabel("Objective function")
    plt.tight_layout(0.5)
    plt.savefig("EEG-decomposition-error" + figname + ".png")


def plot_coef_hist(decomposition_weight, figname, width=1):
    correlation = sorted(Counter(decomposition_weight).items())
    labels, values = zip(*correlation)
    indexes = arange(len(correlation))
    plt.figure()
    plt.bar(indexes, values, width, linewidth=0)
    plt.savefig("EEG-coeff_hist_sorted" + figname + ".png")


def plot_weight_hist(amplitudes, figname, width=1):
    amplitudes.sort()
    indexes = arange(len(amplitudes))
    plt.figure()
    width = 1
    plt.bar(indexes, amplitudes, width, linewidth=0)
    plt.savefig("EEG-weight_sorted" + figname + ".png")


def plot_atom_usage(X, kernels, n_nonzero_coefs, n_jobs, figname):
    r, code = multivariate_sparse_encode(
        X, kernels, n_nonzero_coefs=n_nonzero_coefs, n_jobs=n_jobs, verbose=2
    )
    n_kernels = len(kernels)
    amplitudes = zeros(n_kernels)
    for i in range(len(code)):
        for s in range(n_nonzero_coefs):
            amplitudes[int(code[i][s, 2])] += abs(code[i][s, 0])

    decomposition_weight = hstack([code[i][:, 2] for i in range(len(code))])
    decomposition_weight.sort()
    weight, _ = histogram(decomposition_weight, len(kernels), normed=False)
    order = weight.argsort()
    plot_kernels(
        kernels,
        len(kernels),
        order=order,
        label=weight,
        amp=amplitudes,
        figname="EEG-kernels" + figname,
        row=6,
    )
    plot_coef_hist(decomposition_weight, figname)
    plot_weight_hist(amplitudes, figname)
    plot_reconstruction_samples(X, r, code, kernels, 3, figname)
