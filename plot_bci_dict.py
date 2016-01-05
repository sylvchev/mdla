import matplotlib.pyplot as plt
from mdla import multivariate_sparse_encode
from collections import Counter
from numpy import arange, hstack, histogram

# TODO: use sum of decomposition weight instead of number of atom usage in
#       plot_atom_usage

def plot_kernels(kernels, n_kernels, col = 5, row = -1,
                 order=None, figname = 'allkernels', label=None):
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

def plot_objective_func(error, n_iter, figname):
    fig = plt.figure()
    objf = fig.add_subplot(1, 1, 1)
    ofun = objf.boxplot(error.T)
    medianof = [median.get_ydata()[0]
                for n, median in enumerate(ofun['medians'])]
    axof = objf.plot(arange(1, n_iter+1), medianof, linewidth=1)
    plt.savefig('EEG-decomposition-error'+figname+'.png')

def plot_atom_usage(X, kernels, n_nonzero_coefs, n_jobs, figname):
    r, code = multivariate_sparse_encode(X, kernels,
                                         n_nonzero_coefs=n_nonzero_coefs,
                                         n_jobs=n_jobs, verbose=2)

    decomposition_weight = hstack([code[i][:,2] for i in range(len(code))])
    decomposition_weight.sort()
    weight, _ = histogram(decomposition_weight, len(kernels), normed=False)
    order = weight.argsort()
    plot_kernels(kernels, len(kernels), order=order, label=weight,
                 figname='EEG-kernels'+figname, row=6)

    correlation = Counter(decomposition_weight).items()
    correlation.sort(key=lambda x: x[1])
    labels, values = zip(*correlation)
    indexes = arange(len(correlation))
    plt.figure()
    width = 1
    plt.bar(indexes, values, width, linewidth=0)
    plt.savefig('EEG-coeff_hist_sorted'+figname+'.png')
