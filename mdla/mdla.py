""" Multivariate dictionary learning"""
# Author: Sylvain Chevallier
# License: GPL v3

import itertools
import sys
from time import time

import numpy as np

# Pour array3d
import scipy.sparse as sp
from joblib import Parallel, cpu_count, delayed
from numpy import floor
from sklearn.base import BaseEstimator, TransformerMixin

# from sklearn.externals.six.moves import zip
from sklearn.utils import assert_all_finite, check_random_state, gen_even_slices


# TODO:
# - speed up by grouping decomposition+update as in pydico.
# - adding extension for shift invariant dico


def find(condition):
    (res,) = np.nonzero(np.ravel(condition))
    return res


def _shift_and_extend(signal, extended_length, shift_offset):
    """_shift_and_extend put a copy of signal in a new container of size
    extended_length and at position shift_offset.
    """
    extended_signal = np.zeros(shape=(extended_length, signal.shape[1]))
    extended_signal[shift_offset : shift_offset + signal.shape[0], :] = signal
    return extended_signal


def _normalize(dictionary):
    """Normalize all dictionary elements to have a unit norm"""
    for i in range(len(dictionary)):
        dictionary[i] /= np.linalg.norm(dictionary[i], "fro")
    return dictionary


def _get_learning_rate(iteration, max_iteration, learning_rate):
    # TODO: change to have last_iterations=max_iterations
    # TODO: verify that max_iter=1 is not a problem for partial_fit
    if learning_rate == 0.0:
        return 0.0
    last_iterations = np.floor(max_iteration * 2.0 / 3.0)
    if iteration >= last_iterations:
        return last_iterations ** learning_rate
    else:
        return (iteration + 1) ** learning_rate


def _multivariate_OMP(signal, dictionary, n_nonzero_coefs=None, verbose=False):
    """Sparse coding multivariate signal with OMP

    Returns residual and a decomposition array (n_nonzero_coefs, 3),
    each line indicating (amplitude, offset, kernel).

    Parameters
    ----------
    signal: array of shape (n_features, n_dims)
        Data matrix.
        Each sample is a matrix of shape (n_features, n_dims) with
        n_features >= n_dims.

    dictionary: list of arrays
        The dictionary against which to solve the sparse coding of
        the data. The dictionary learned is a list of n_kernels
        elements. Each element is a convolution kernel, i.e. an
        array of shape (k, n_dims), where k <= n_features and is
        kernel specific. The algorithm normalizes the kernels.

    n_nonzero_coefs : int
        Sparsity controller parameter for multivariate variant
        of OMP

    verbose:
        Degree of output the procedure will print.

    Returns
    -------
    residual: array of (n_features, n_dims)
        Reconstruction error residual.

    decomposition: array of shape (n_nonzero_coefs, 3)
        The sparse code decomposition : (amplitude, offset, kernel)
        for each n_nonzero_coefs.

    See also
    --------
    SparseCoder
    """
    n_features, n_dims = signal.shape
    n_kernels = len(dictionary)
    k_max_len = np.max([k.shape[0] for k in dictionary])
    k_min_len = np.min([k.shape[0] for k in dictionary])

    residual = signal.copy()
    # signal decomposition is [amplitude, offset, kernel]*n_nonzero_coefs
    decomposition = np.zeros((n_nonzero_coefs, 3)) - 1
    if verbose >= 4:
        print("[M-OMP # 0 ] signal is")
        print(residual)
    Ainv = np.zeros((n_kernels, n_kernels), np.float)
    Ainv[0, 0] = 1.0

    # First iteration
    correlation_score = np.zeros((n_kernels, n_features))
    for i in range(n_kernels):
        corr = 0
        for j in range(n_dims):
            corr += np.correlate(residual[:, j], dictionary[i][:, j], "valid")
        correlation_score[i, : len(corr)] = corr

    if verbose >= 4:
        print("[M-OMP # 0 ] correlation is", correlation_score)

    (k_selected, k_off) = np.unravel_index(
        np.argmax(np.abs(correlation_score)), correlation_score.shape
    )
    k_amplitude = correlation_score[k_selected, k_off]

    # Put the selected kernel into an atom
    selected_atom = _shift_and_extend(dictionary[k_selected], n_features, k_off)

    if verbose >= 3:
        print(
            "[M-OMP # 0 ] kernel", k_selected, "is selected with amplitude", k_amplitude
        )
    if verbose >= 4:
        print(selected_atom)

    # List of selected atoms is flatten
    selected_list = selected_atom.flatten()
    estimated_signal = k_amplitude * selected_atom
    residual = signal - estimated_signal

    if verbose >= 4:
        print("[M-OMP # 0 ] residual is now")
        print(residual)

    signal_energy = (signal ** 2).sum(1).mean()
    residual_energy = (residual ** 2).sum(1).mean()

    if verbose >= 3:
        print(
            "[M-OMP # 0 ] signal energy is",
            signal_energy,
            "and residual energy is",
            residual_energy,
        )

    decomposition[0, :] = np.array([k_amplitude, k_off, k_selected])

    # Main loop
    atoms_in_estimate = 1
    while atoms_in_estimate < n_nonzero_coefs:
        correlation_score = np.zeros(
            (n_kernels, max(n_features, k_max_len) - min(n_features, k_min_len) + 1)
        )
        # TODO: compute correlation only if kernel has not been selected
        for i in range(n_kernels):
            corr = 0
            for j in range(n_dims):
                corr += np.correlate(residual[:, j], dictionary[i][:, j], "valid")
            correlation_score[i, : len(corr)] = corr
        if verbose >= 4:
            print("[M-OMP #", atoms_in_estimate, "] correlation is", correlation_score)
        (k_selected, k_off) = np.unravel_index(
            np.argmax(np.abs(correlation_score)), correlation_score.shape
        )
        k_amplitude = correlation_score[k_selected, k_off]

        # Verify that the atom is not already selected
        if np.any((k_off == decomposition[:, 1]) & (k_selected == decomposition[:, 2])):
            if verbose >= 4:
                print(
                    "kernel",
                    k_selected,
                    "already selected from",
                    "a previous iteration. Exiting loop.",
                )
            break
        selected_atom = _shift_and_extend(dictionary[k_selected], n_features, k_off)
        if verbose >= 3:
            print(
                "[M-OMP #",
                atoms_in_estimate,
                "] kernel",
                k_selected,
                "at position",
                k_off,
            )

        # Update decomposition coefficients
        v = np.array(selected_list.dot(selected_atom.flatten()), ndmin=2)
        b = np.array(Ainv[0:atoms_in_estimate, 0:atoms_in_estimate].dot(v.T), ndmin=2)
        vb = v.dot(b)
        if np.allclose(vb, 1.0):
            beta = 0.0
        else:
            beta = 1.0 / (1.0 - vb)
        alpha = correlation_score[k_selected, k_off] * beta
        decomposition[0:atoms_in_estimate, 0:1] -= alpha * b
        Ainv[0:atoms_in_estimate, 0:atoms_in_estimate] += beta * b.dot(b.T)
        Ainv[atoms_in_estimate : atoms_in_estimate + 1, 0:atoms_in_estimate] = -beta * b.T
        Ainv[0:atoms_in_estimate, atoms_in_estimate : atoms_in_estimate + 1] = -beta * b
        Ainv[atoms_in_estimate, atoms_in_estimate] = beta
        decomposition[atoms_in_estimate] = np.array(
            [alpha, k_off, k_selected], dtype=np.float64
        )
        atoms_in_estimate += 1
        selected_list = np.vstack((selected_list, selected_atom.flatten()))

        # Update the estimated signal and residual
        estimated_signal = np.zeros((n_features, n_dims))
        for i in range(atoms_in_estimate):
            k_amp = decomposition[i, 0]
            k_off = int(decomposition[i, 1])
            k_kernel = int(decomposition[i, 2])
            k_len = dictionary[k_kernel].shape[0]

            estimated_signal[k_off : k_off + k_len, :] += k_amp * dictionary[k_kernel]
        residual = signal - estimated_signal

        if verbose >= 3:
            residual_energy = (residual ** 2).sum(1).mean()
            print(
                "[M-OMP #",
                atoms_in_estimate - 1,
                "] signal energy is",
                signal_energy,
                "and residual energy is",
                residual_energy,
            )
        if verbose >= 4:
            print(
                "[M-OMP #",
                atoms_in_estimate - 1,
                "]: partial decomposition",
                "is",
                decomposition[:atoms_in_estimate, :],
            )

    # End big loop
    decomposition = decomposition[0:atoms_in_estimate, :]
    if verbose >= 4:
        print(
            "[M-OMP # end ]: decomposition matrix is: ",
            "(amplitude, offset, kernel_id)",
            decomposition,
        )
        print("")

    return residual, decomposition


def _multivariate_sparse_encode(X, kernels, n_nonzero_coefs=None, verbose=False):
    """Sparse coding multivariate signal

    Each columns of the results is the OMP approximation on a
    multivariate dictionary.

    Parameters
    ----------
    X: array of shape (n_samples, n_features, n_dims)
        Data matrices, where n_samples in the number of samples
        Each sample is a matrix of shape (n_features, n_dims) with
        n_features >= n_dims.

    kernels: list of arrays
        The dictionary against which to solve the sparse coding of
        the data. The dictionary learned is a list of n_kernels
        elements. Each element is a convolution kernel, i.e. an
        array of shape (k, n_dims), where k <= n_features and is
        kernel specific. The algorithm normalizes the kernels.

    n_nonzero_coefs : int
        Sparsity controller parameter for multivariate variant
        of OMP

    verbose:
        Degree of output the procedure will print.

    Returns
    -------

    residual: list of arrays (n_features, n_dims)
        The sparse decomposition residuals

    decomposition: list of arrays (n_nonzero_coefs, 3)
        The sparse code decomposition : (amplitude, offset, kernel)
        for all n_nonzero_coefs. The list concatenates all the
        decomposition of the n_samples of X.

    See also
    --------
    SparseCoder
    """
    n_samples, n_features, n_dims = X.shape
    n_kernels = len(kernels)

    if n_nonzero_coefs > n_kernels:
        raise ValueError("The sparsity should be less than the " "number of atoms")
    if n_nonzero_coefs <= 0:
        raise ValueError("The sparsity should be positive")

    decomposition = list()
    residual = list()
    for k in range(n_samples):
        r, d = _multivariate_OMP(X[k, :, :], kernels, n_nonzero_coefs, verbose)
        decomposition.append(d)
        residual.append(r)
    return residual, decomposition


def multivariate_sparse_encode(
    X, dictionary, n_nonzero_coefs=None, n_jobs=1, verbose=False
):
    """Sparse coding

    Each row of the result is the solution to a sparse coding problem.
    The goal is to find a sparse array `code` such that::

        X ~= code * dictionary

    Parameters
    ----------
    X: array of shape (n_samples, n_features, n_dims)
        Data matrices, where n_samples in the number of samples
        Each sample is a matrix of shape (n_features, n_dims) with
        n_features >= n_dims.

    dictionary: list of arrays or MultivariateDictLearning instance
        The dictionary against which to solve the sparse coding of
        the data. The dictionary learned is a list of n_kernels
        elements. Each element is a convolution kernel, i.e. an
        array of shape (k, n_dims), where k <= n_features and is
        kernel specific. The algorithm normalizes the kernels.

    n_nonzero_coefs: int, 0.1 * n_features by default
        Number of nonzero coefficients to target in each column of the
        solution.

    n_jobs: int, optional
        Number of parallel jobs to run.

    verbose:
        Degree of output the procedure will print.

    Returns
    -------
    residual: list of array(n_features, n_dims)
        Decomposition residual

    code: list of arrays (n_nonzero_coefs, 3)
        The sparse code decomposition: (amplitude, offset, kernel)
        for all n_nonzero_coefs. The list concatenates all the
        decomposition of the n_samples of X

    See also
    --------
    sklearn.linear_model.lars_path
    sklearn.linear_model.orthogonal_mp
    sklearn.linear_model.Lasso
    SparseCoder
    """
    if verbose >= 2:
        tstart = time()

    X = array3d(X)
    n_samples, n_features, n_dims = X.shape
    if isinstance(dictionary, MultivariateDictLearning) or isinstance(
        dictionary, MiniBatchMultivariateDictLearning
    ):
        kernels = dictionary.kernels_
    else:
        kernels = dictionary

    if n_nonzero_coefs is None:
        n_nonzero_coefs = max(int(np.round(n_features / 10)), 1)

    if n_jobs == -1:
        n_jobs = cpu_count()

    if n_jobs == 1:
        r, d = _multivariate_sparse_encode(X, kernels, n_nonzero_coefs, verbose)
        return r, d

    # Enter parallel code block
    residuals = list()
    decompositions = list()
    slices = list(gen_even_slices(n_samples, n_jobs))

    if verbose >= 3:
        print(
            "[Debug-MOMP] starting parallel %d jobs for %d samples" % (n_jobs, n_samples)
        )

    views = Parallel(n_jobs=n_jobs)(
        delayed(_multivariate_sparse_encode)(
            X[this_slice], kernels, n_nonzero_coefs, verbose
        )
        for this_slice in slices
    )
    # for this_slice, this_res, this_code in zip(slices, res_views, code_views):
    for this_res, this_code in views:
        residuals.extend(this_res)
        decompositions.extend(this_code)

    if verbose >= 3:
        print("sparse decomposition: ", time() - tstart, "s")

    return residuals, decompositions


def reconstruct_from_code(code, dictionary, n_features):
    """Reconstruct input from multivariate dictionary decomposition

    Parameters
    ----------
    code: list of arrays (n_nonzero_coefs, 3)
        The sparse code decomposition: (amplitude, offset, kernel)
        for all n_nonzero_coefs. The list concatenates all the
        decomposition of the n_samples of input X

    dictionary: list of arrays
        The dictionary against which to solve the sparse coding of
        the data. The dictionary learned is a list of n_kernels
        elements. Each element is a convolution kernel, i.e. an
        array of shape (k, n_dims), where k <= n_features and is
        kernel specific. The algorithm normalizes the kernels.

    n_features: int
        A signal is an array of shape (n_features, n_dims)

    Returns
    -------
    signal: array of shape (n_samples, n_features, n_dims)
        Data matrices of the reconstructed signal, where n_samples
        in the number of samples. Each sample is a matrix of shape
        (n_features, n_dims) with n_features >= n_dims.
    """
    n_dims = dictionary[0].shape[1]
    n_samples = len(code)
    signal = list()
    for i in range(n_samples):
        decomposition = code[i]
        s = np.zeros(shape=(n_features, n_dims))
        for k_amplitude, k_offset, k_selected in decomposition:
            s += k_amplitude * _shift_and_extend(
                dictionary[int(k_selected)], int(n_features), int(k_offset)
            )
        signal.append(s)
    return np.array(signal)


def _compute_gradient(
    dictionary,
    decomposition,
    residual,
    learning_rate=None,
    random_state=None,
    verbose=False,
):
    """Compute the gradient to apply on the dictionary.

    Parameters
    ----------
    dictionary: list of arrays of
        Value of the dictionary at the previous iteration.
        The dictionary is a list of n_kernels
        elements. Each element is a convolution kernel, i.e. an
        array of shape (k, n_dims), where k <= n_features and is
        kernel specific. The algorithm normalizes the kernels.

    decomposition: array of shape (n_nonzero_coefs, 3)
        Sparse decomposition of the data against which to optimize
        the dictionary.

    residual: array of shape (n_features, n_dims)
        Residual of the sparse decomposition

    learning_rate: real,
        hyperparameter controling the convergence rate

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

    verbose:
        Degree of output the procedure will print.

    Returns
    -------
    gradient: list of arrays
        list of gradients to apply on the dictionnary.
    """
    n_kernels = len(dictionary)
    random_state = check_random_state(random_state)

    n_active_atoms = decomposition.shape[0]
    signal_len, n_dims = residual.shape
    coefs = decomposition[:, 0]
    offsets = decomposition[:, 1].astype(int)
    index_atoms = decomposition[:, 2].astype(int)
    hessian_sum = 0
    hessian_count = 0

    # Initialization
    gradient = [np.zeros_like(dictionary[i]) for i in range(n_kernels)]

    for i in range(n_active_atoms):
        k_len = dictionary[index_atoms[i]].shape[0]
        if k_len + offsets[i] - 1 <= signal_len:
            # Do not consider oversized atoms
            r = residual[offsets[i] : k_len + offsets[i], :]  # modif
            gradient[index_atoms[i]] += np.conj(coefs[i] * r)
        if verbose >= 5:
            print("[M-DU] Update kernel", i, ", gradient is", gradient[index_atoms[i]])

    # First pass to estimate the step
    step = np.zeros((n_kernels, 1))
    for i in range(n_kernels):
        k_len = dictionary[i].shape[0]
        active_idx = find(index_atoms == i)
        offsets_sorted = np.sort(offsets[active_idx])
        offsets_sorted_idx = np.argsort(offsets[active_idx])
        active_coefs = coefs[active_idx]
        active_coefs = active_coefs[offsets_sorted_idx]
        dOffsets = offsets_sorted[1:] - offsets_sorted[0:-1]
        dOffsets_idx = find(dOffsets < k_len)
        if dOffsets_idx.size == 0:
            # Good separation, use a direct approximation of
            # the Hessian
            hessian_corr = 0
        else:
            # Weak Hessian approximation, crude correction to include
            # the overlapping atoms
            hessian_corr = (
                2.0
                * np.sum(
                    np.abs(active_coefs[dOffsets_idx] * active_coefs[dOffsets_idx + 1])
                    * (k_len - dOffsets[dOffsets_idx])
                )
                / k_len
            )
        hessian_base = np.sum(np.abs(coefs[active_idx]) ** 2)
        # if learning_rate+hessian_corr+hessian_base == 0.:
        if learning_rate == 0.0:
            # Gauss-Newton method if mu = 0
            step[i] = 0
        else:
            step[i] = 1.0 / (learning_rate + hessian_corr + hessian_base)
        if (hessian_corr + hessian_base) != 0:
            hessian_sum += hessian_corr + hessian_base
            hessian_count += 1

    if verbose >= 5:
        print("[M-DU]: step is:")
        print(step)
        print("[M-DU]: gradient is:")
        print(gradient)
    gradient = [gradient[i] * step[i] for i in range(n_kernels)]

    # TODO: add forget factor?

    if verbose >= 5:
        print("[M-DU]: diff is:")
        print(gradient)

    return gradient


def _update_dict(
    dictionary,
    decomposition,
    residual,
    learning_rate=None,
    random_state=None,
    verbose=False,
):
    """Update the dense dictionary in place

    Update the dictionary based on sparse codes and residuals

    Parameters
    ----------
    dictionary: list of arrays
        The dictionary against which to solve the sparse coding of
        the data. The dictionary learned is a list of n_kernels
        elements. Each element is a convolution kernel, i.e. an
        array of shape (k, n_dims), where k <= n_features and is
        kernel specific. The algorithm normalizes the kernels.

    decomposition: list of array of shape (n_features, n_dims)
        Each sample is a matrix of shape (n_features, n_dims) with
        n_features >= n_dims.

    residual: list of array of shape (n_features, n_dims)
        Residual of the sparse decomposition

    learning_rate: real,
        hyperparameter controling the convergence rate

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

    verbose:
        Degree of output the procedure will print.

    Returns
    -------
    dictionary: list of arrays
        Updated dictionary.

    """
    if verbose >= 2:
        tstart = time()

    gradients = list()
    for c, r in zip(decomposition, residual):
        g = _compute_gradient(dictionary, c, r, learning_rate, random_state, verbose)
        gradients.append(g)

    _g = [np.zeros(k.shape) for k in dictionary]
    for g in gradients:
        for i in range(len(dictionary)):
            _g[i] = _g[i] + g[i]

    if verbose >= 3:
        print("[M-DU] energy change ratio is ")
        print(
            [
                (_g[i] ** 2).sum(0).mean() / (dictionary[i] ** 2).sum(0).mean()
                for i in range(len(dictionary))
            ]
        )
        print("learning_rate is ", learning_rate)

    for i in range(len(dictionary)):
        dictionary[i] = dictionary[i] + _g[i]
    dictionary = _normalize(dictionary)

    if verbose >= 3:
        print("dict update: ", time() - tstart, "s")

    # if verbose >= 1:
    #     diff = grad = 0.
    #     for i in range(len(dictionary)):
    #         grad += (_g[i]**2).sum(0).mean()
    #         diff += ((dictionary[i]-initial[i])**2).sum(0).mean()
    #     print ('[MDLA-DU] Gradient energy is', grad, 'and energy change is', diff)
    #     print ([((dictionary[i]-initial[i])**2).sum(0).mean() for i in range(len(dictionary))])
    #     print (dictionary[7])

    return dictionary


# def _update_dict_online(signal, dictionary, n_nonzero_coefs=None,
#                         learning_rate=None, random_state=None, verbose=False,
#                         return_r2=False):
#     """Update the dense dictionary in place

#     First OMP to obtain a sparse code, then update the dictionary.

#     Parameters
#     ----------
#     signal: array of shape (n_features, n_dims)
#         Data matrix.
#         Each sample is a matrix of shape (n_features, n_dims) with
#         n_features >= n_dims.

#     dictionary: list of arrays
#         The dictionary against which to solve the sparse coding of
#         the data. The dictionary learned is a list of n_kernels
#         elements. Each element is a convolution kernel, i.e. an
#         array of shape (k, n_dims), where k <= n_features and is
#         kernel specific. The algorithm normalizes the kernels.

#     n_nonzero_coefs : int
#         Sparsity controller parameter for multivariate variant
#         of OMP

#     learning_rate: real,
#         hyperparameter controling the convergence rate

#     random_state: int or RandomState
#         Pseudo number generator state used for random sampling.

#     verbose:
#         Degree of output the procedure will print.

#     return_r2: bool
#         Whether to compute and return the residual sum of squares corresponding
#         to the computed solution.

#     Returns
#     -------
#     dictionary: list of arrays
#         Updated dictionary.

#     """
#     n_kernels = len(dictionary)
#     residual, decomposition = _multivariate_OMP(signal, dictionary, n_nonzero_coefs, verbose)
#     gradient = _compute_gradient(dictionary, decomposition, residual, random_state, verbose)
#     for i in n_kernels:
#         dictionary[i] = dictionary[i] + gradient[i]
#     dictionary = _normalize(dictionary)

#     if return_r2:
#         r2 = np.linalg.norm(residual, 'fro')
#         return dictionary, r2
#     return dictionary


def multivariate_dict_learning(
    X,
    n_kernels,
    n_nonzero_coefs=1,
    max_iter=100,
    tol=1e-8,
    n_jobs=1,
    learning_rate=None,
    dict_init=None,
    callback=None,
    verbose=False,
    kernel_init_len=None,
    random_state=None,
    dict_obj=None,
):
    """Solves a dictionary learning matrix factorization problem.

    Finds the best dictionary and the corresponding sparse code for
    approximating the data matrix X by solving::

        (U^*, V^*) = argmin 0.5 || X - U V ||_F^2 s.t. || U ||_0 < K
                     (U,V)
                    with || V_i ||_F = 1 for all  0 <= i < n_kernels

    where V is the dictionary and U is the sparse code.

    Parameters
    ----------
    X: array of shape (n_samples, n_features, n_dims)
        Data matrices, where n_samples in the number of samples
        Each sample is a matrix of shape (n_features, n_dims) with
        n_features >= n_dims.

    n_kernels: int,
        Number of dictionary atoms to extract.

    n_nonzero_coefs: int,
        Sparsity controlling parameter.

    max_iter: int,
        Maximum number of iterations to perform.

    tol: float,
        Tolerance for the stopping condition.

    n_jobs: int,
        Number of parallel jobs to run, or -1 to autodetect.

    learning_rate: real,
        hyperparameter controling the convergence rate

    dict_init: list of arrays
        Initial value for the dictionary for warm restart scenarios.
        List of n_kernels elements, each one is an array of shape
        (k, n_dims), where k <= n_features and is kernel specific

    callback:
        Callable that gets invoked every iterations.

    verbose:
        Degree of output the procedure will print.

    kernel_init_len: int,
        Initial length for all the dictionary kernel

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

    Returns
    -------
    code: list of arrays (n_nonzero_coefs, 3)
        The sparse code decomposition: (amplitude, offset, kernel)
        for all n_nonzero_coefs. The list concatenates all the
        decomposition of the n_samples of X

    dictionary: list of arrays
        The dictionary learned is a list of n_kernels elements.
        Each element is a convolution kernel, i.e. an array
        of shape (k, n_dims), where k <= n_features and is kernel
        specific

    errors: array
        matrix of errors at each iteration.

    See also
    --------
    multivariate_dict_learning_online
    DictionaryLearning
    MiniBatchDictionaryLearning
    SparsePCA
    MiniBatchSparsePCA
    """
    t0 = time()
    n_samples, n_features, n_dims = X.shape
    random_state = check_random_state(random_state)

    if n_jobs == -1:
        n_jobs = cpu_count()

    if dict_init is not None:
        dictionary = list(dict_init)
        n_kernels = len(dictionary)
        if verbose >= 2:
            print("\n[MDL] Warm restart with dictionary of", len(dictionary), "kernels")
    else:
        # Init the dictionary with random samples of X
        k_len = kernel_init_len
        max_offset = n_features - k_len

        if verbose >= 2:
            print("[MDL] Initializing dictionary from samples")
        offset = random_state.randint(0, max_offset + 1, n_kernels)
        ind_kernels = random_state.permutation(n_samples)[:n_kernels]
        dictionary = [X[p[0], p[1] : p[1] + k_len, :] for p in zip(ind_kernels, offset)]
        dictionary = _normalize(dictionary)

    errors = []
    current_cost = np.nan

    if verbose == 1:
        print("\n[dict_learning]", end=" ")

    for ii in range(max_iter):
        dt = time() - t0
        if verbose >= 2:
            print(
                "[MDL] Iteration % 3i "
                "(elapsed time: % 3is, % 4.1fmn, current cost % 7.3f)"
                % (ii, dt, dt / 60, current_cost / n_samples)
            )
        elif verbose == 1:
            sys.stdout.write(".")
            sys.stdout.flush()

        try:
            # Update code
            r, code = multivariate_sparse_encode(
                X,
                dictionary,
                n_nonzero_coefs=n_nonzero_coefs,
                n_jobs=n_jobs,
                verbose=verbose,
            )
            # Update dictionary
            mu = _get_learning_rate(ii, max_iter, learning_rate)
            dictionary = _update_dict(
                dictionary,
                decomposition=code,
                residual=r,
                verbose=verbose,
                learning_rate=mu,
                random_state=random_state,
            )
            if verbose >= 2:
                print("[MDL] Dictionary updated, iteration", ii, "with learning rate", mu)

            # Cost function
            current_cost = 0.0
            for i in range(len(r)):
                current_cost += np.linalg.norm(r[i], "fro") + len(code[i])
            # current_cost = 0.5 * residuals + np.sum(np.abs(code))
            errors.append(current_cost / len(r))
        except KeyboardInterrupt:
            break

        if ii > 0:
            dE = abs(errors[-2] - errors[-1])
            # assert(dE >= -tol * errors[-1])
            if ii == 1 and verbose == 1:
                print(
                    "Expecting this learning experiment to finish in %.2f m"
                    % ((time() - t0) * max_iter / 60.0)
                )
            if dE < tol * errors[-1]:
                if verbose >= 1:
                    # A line return
                    print("")
                elif verbose:
                    print("--- Convergence reached after %d iterations" % ii)
                break
        if callback is not None:
            callback(locals())
    # reformating the error
    # print ("errors=", len(errors), ", reshape into", (max_iter,))
    if len(errors) < max_iter:
        errors = np.array(errors).reshape((len(errors),))
    else:
        errors = np.array(errors).reshape((max_iter,))
    return code, dictionary, errors


def multivariate_dict_learning_online(
    X,
    n_kernels=2,
    n_nonzero_coefs=1,
    n_iter=100,
    iter_offset=0,
    dict_init=None,
    callback=None,
    batch_size=None,
    verbose=False,
    shuffle=True,
    n_jobs=1,
    kernel_init_len=None,
    learning_rate=None,
    random_state=None,
    dict_obj=None,
):
    """Solves an online multivariate dictionary learning factorization problem

    Finds the best dictionary and the corresponding sparse code for
    approximating the data matrices X by solving::

        (U^*, V^*) = argmin 0.5 || X - U V ||_F^2 s.t. || U ||_0 < k
                     (U,V)
                     with || V_i ||_F = 1 for all  0 <= i < n_kernels

    where V is the dictionary and U is the sparse code. This is
    accomplished by repeatedly iterating over mini-batches by slicing
    the input data.

    Parameters
    ----------
    X: array of shape (n_samples, n_features, n_dims)
        Set of n_samples data matrix.

    n_kernels : int,
        Number of dictionary atoms to extract.

    n_nonzero_coefs : int,
        Sparsity controlling parameter.

    n_iter : int,
        Number of iterations to perform.

    iter_offset : int, default 0
        Number of previous iterations completed on the dictionary used for
        initialization.

    dict_init : list of arrays
        Initial value for the dictionary for warm restart scenarios.
        The dictionary is a list of n_kernels elements. Each element
        is a convolution kernel, i.e. an array of shape (k, n_dims),
        where k <= n_features and is kernel specific.

    callback :
        Callable that gets invoked every iterations.

    batch_size : int,
        The number of samples to take in each batch. If None, initialised
        as 5 * n_jobs

    verbose :
        Degree of output the procedure will print.

    shuffle : boolean,
        Whether to shuffle the data before splitting it in batches.

    n_jobs : int,
        Number of parallel jobs to run, or -1 to autodetect.

    learning_rate: real,
        hyperparameter controling the convergence rate

    random_state : int or RandomState
        Pseudo number generator state used for random sampling.

    Returns
    -------
    dictionary : list of arrays
        the solutions to the dictionary learning problem

    errors: array
        matrix of errors at each iteration.

    See also
    --------
    multivariate_dict_learning
    DictionaryLearning
    MiniBatchDictionaryLearning
    SparsePCA
    MiniBatchSparsePCA
    """
    t0 = time()

    n_samples, n_features, n_dims = X.shape
    # if n_samples < n_kernels:
    #     print ('Too few examples, reducing the number of kernel to', n_samples)
    #     n_kernels = n_samples
    random_state = check_random_state(random_state)

    # if n_jobs == -1 and cpu_count() != 0:
    #     n_jobs = cpu_count()
    # else: n_jobs = 1
    if n_jobs == -1:
        n_jobs = cpu_count()

    if batch_size is None:
        batch_size = 5 * n_jobs

    if dict_init is not None:
        dictionary = list(dict_init)
        n_kernels = len(dictionary)
        if verbose >= 2:
            print("\n[MDL] Warm restart with dictionary of", len(dictionary), "kernels")
    else:
        # Init dictionary with random X samples
        k_len = kernel_init_len
        max_offset = n_features - k_len

        if verbose >= 2:
            print("[MDL] Initializing dictionary from samples")
        offset = random_state.randint(0, max_offset + 1, n_kernels)
        ind_kernels = random_state.permutation(n_samples)[:n_kernels]
        dictionary = [X[p[0], p[1] : p[1] + k_len, :] for p in zip(ind_kernels, offset)]
        dictionary = _normalize(dictionary)

    errors = []
    current_cost = np.nan

    if verbose == 1:
        print("\n[dict_learning]", end=" ")

    n_batches = int(floor(float(len(X)) / batch_size))
    if n_batches == 0:
        n_batches = 1
    if shuffle:
        X_train = X.copy()
        random_state.shuffle(X_train)
    else:
        X_train = X
    batches = np.array_split(X_train, n_batches)
    batches = itertools.cycle(batches)

    if verbose >= 2:
        print(
            "[MDL] Using %d jobs and %d batch of %d examples"
            % (n_jobs, n_batches, batch_size)
        )

    for ii, this_X in zip(
        range(iter_offset * n_batches, (iter_offset + n_iter) * n_batches), batches
    ):
        dt = time() - t0

        try:
            r, code = multivariate_sparse_encode(
                this_X,
                dictionary,
                n_nonzero_coefs=n_nonzero_coefs,
                n_jobs=n_jobs,
                verbose=verbose,
            )
            # Update dictionary
            mu = _get_learning_rate(
                ii / n_batches + 1, iter_offset + n_iter + 1, learning_rate
            )
            dictionary = _update_dict(
                dictionary,
                decomposition=code,
                residual=r,
                verbose=verbose,
                learning_rate=mu,
                random_state=random_state,
            )

            # Cost function
            current_cost = 0.0
            for i in range(len(r)):
                current_cost += np.linalg.norm(r[i], "fro") + len(code[i])
            errors.append(current_cost / len(r))

            if np.mod((ii - iter_offset), n_batches) == 0:
                if verbose >= 2:
                    print(
                        "[MDL] Dictionary updated, iteration %d "
                        "with learning rate %.2f (elapsed time: "
                        "% 3is, % 4.1fmn)"
                        % ((ii - iter_offset) / n_batches, mu, dt, dt / 60)
                    )
                elif verbose == 1:
                    sys.stdout.write(".")
                    sys.stdout.flush()
                if callback is not None:
                    callback(locals())

            if ii == (iter_offset + 1) * n_batches and verbose >= 1:
                print(
                    "Expecting this learning iterations to finish in %.2f m"
                    % ((time() - t0) * n_iter / 60.0)
                )
                # if verbose == 1:
                # print ('Time from begining is',time()-t0,'s, with n_iter=',
                #         n_iter, ', iter_offset=', iter_offset,
                #         ', i.e.', n_iter, 'iterations to go.')
        except KeyboardInterrupt:
            break

    if verbose >= 2:
        dt = time() - t0
        print("[MDL] learning done (total time: % 3is, % 4.1fmn)" % (dt, dt / 60))
    # reformating the error
    errors = np.array(errors).reshape((n_iter, n_batches))
    return dictionary, errors


class MultivariateDictMixin(TransformerMixin):
    """Multivariate sparse coding mixin"""

    # TODO update doc.
    def _set_mdla_params(
        self,
        n_kernels,
        n_nonzero_coefs=1,
        kernel_init_len=None,
        n_jobs=1,
        learning_rate=None,
    ):
        # TODO: add kernel_init_len=None, kernel_max_len=None,
        # kernel_adapt_thres=None, kernel_adapt_inc=None
        self.n_kernels = n_kernels
        self.n_nonzero_coefs = n_nonzero_coefs
        self.n_jobs = n_jobs
        self.kernel_init_len = kernel_init_len
        if learning_rate is not None:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = 1.5
        # split_sign=False ?
        # self.kernel_max_len = kernel_max_len
        # self.kernel_adapt_thres = kernel_adapt_thres
        # self.kernel_adapt_inc = kernel_adapt_inc

    def transform(self, X, y=None):
        """Encode the data as a sparse combination of the dictionary atoms.

        The coding method is the multivariate OMP, whose parameters are
        n_kernels: the number of dictionary kernels
        n_nonzero_coefs: sparsity term
        n_jobs: for parallel jobs

        Parameters
        ----------
        X : array of shape (n_samples, n_features, n_dims)
            Test data to be transformed, must have the same number of
            features as the data used to train the model.

        Returns
        -------
        X_new : array, shape (n_samples, n_kernels, n_dims)
            Transformed data

        """
        X = array3d(X)
        n_samples, n_features, n_dims = X.shape

        _, code = multivariate_sparse_encode(
            X, self.kernels_, n_nonzero_coefs=self.n_nonzero_coefs, n_jobs=self.n_jobs
        )

        return code


class SparseMultivariateCoder(BaseEstimator, MultivariateDictMixin):
    """Sparse coding of multivariate data

    Finds a sparse representation of data against a fixed, precomputed
    dictionary.

    Each row of the result is the solution to a sparse coding problem.
    The goal is to find a sparse array `code` such that::

        X ~= code * dictionary

    Parameters
    ----------
    dictionary : list of arrays array
        The dictionary atoms used for sparse coding. The dictionary
        learned is a list of n_kernels elements. Each element is a
        convolution kernel, i.e. an array of shape (k, n_dims), where
        k <= n_features and is kernel specific.

    n_nonzero_coefs : int, ``0.1 * n_features`` by default
        Number of nonzero coefficients to target in each column of the
        solution.

    n_jobs : int,
        number of parallel jobs to run

    Attributes
    ----------
    `kernels_` : list of arrays
        The unchanged dictionary atoms.

    See also
    --------
    DictionaryLearning
    MiniBatchDictionaryLearning
    SparsePCA
    MiniBatchSparsePCA
    sparse_encode
    """

    def __init__(
        self,
        dictionary,
        n_nonzero_coefs=None,
        kernel_init_len=None,
        n_jobs=1,
        learning_rate=None,
    ):
        self._set_mdla_params(
            len(dictionary), n_nonzero_coefs, kernel_init_len, n_jobs, learning_rate
        )
        self.kernels_ = list(dictionary)

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged

        This method is just there to implement the usual API and hence
        work in pipelines.
        """
        return self


class MultivariateDictLearning(BaseEstimator, MultivariateDictMixin):
    """Multivariate dictionary learning

    Finds a dictionary (a set of atoms represented as matrices) that can
    best be used to represent data using a sparse code.

    Solves the optimization problem:

        (U^*,V^*) = argmin 0.5 || Y - U V ||_F^2 s.t. || U ||_0 < K
                    (U,V)
                    with || V_i ||_F = 1 for all  0 <= i < n_kernels

    Parameters
    ----------
    n_kernels : int,
        number of dictionary elements to extract

    max_iter : int,
        maximum number of iterations to perform

    tol : float,
        tolerance for numerical error

    n_nonzero_coefs : int, ``0.1 * n_features`` by default
        Number of nonzero coefficients to target in each column of the
        solution.

    n_jobs : int,
        number of parallel jobs to run

    dict_init : list of arrays
        Initial values for the dictionary, for warm restart
        The dictionary is a list of n_kernels elements. Each element
        is a convolution kernel, i.e. an array of shape (k, n_dims),
        where k <= n_features and is kernel specific.

    verbose :
        degree of verbosity of the printed output

    random_state : int or RandomState
        Pseudo number generator state used for random sampling.

    learning_rate : float
        Value for the learning rate exponent, of the form
        i**learning_rate, where i is the iteration.

    callback: function
        Function called during learning process, the only parameter
        is a dictionary containing all local variables

    Attributes
    ----------
    `kernels_` : list of arrays
        dictionary atoms extracted from the data

    `error_` : array
        vector of errors at each iteration

    Notes
    -----
    **References:**

    Multivariate Temporal Dictionary Learning for EEG,
    J. Neurosci Methods, 2013

    Shift & 2D Rotation Invariant Sparse Coding for
    Multivariate Signals, IEEE TSP, 2012.


    See also
    --------
    SparseCoder
    MiniBatchDictionaryLearning
    SparsePCA
    MiniBatchSparsePCA
    """

    def __init__(
        self,
        n_kernels=None,
        max_iter=1000,
        tol=1e-8,
        n_nonzero_coefs=None,
        n_jobs=1,
        kernel_init_len=None,
        dict_init=None,
        verbose=False,
        learning_rate=None,
        random_state=None,
        callback=None,
    ):

        self._set_mdla_params(
            n_kernels, n_nonzero_coefs, kernel_init_len, n_jobs, learning_rate
        )
        self.max_iter = max_iter
        self.tol = tol
        if dict_init is not None:
            self.dict_init = _normalize(list(dict_init))
        else:
            self.dict_init = None
        self.verbose = verbose
        self.random_state = random_state
        self.callback = callback

    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features, n_dims)
            Training matrices, where n_samples in the number of samples.
            Each sample is a matrix of shape (n_features, n_dims) with
            n_features >= n_dims.

        Returns
        -------
        self: object
            Returns the object itself
        """
        random_state = check_random_state(self.random_state)
        X = array3d(X)
        n_samples, n_features, n_dims = X.shape
        if hasattr(self, "kernels_"):
            self.dict_init = _normalize(list(self.kernels_))
            if self.verbose >= 1:
                print("\nWarm restart with existing kernels")
                # print (self.kernels_[7])
                # print ('')
        if self.dict_init is not None:
            self.n_kernels = len(self.dict_init)
        elif self.n_kernels is None:
            self.n_kernels = 2 * n_features
        if self.kernel_init_len is None:
            self.kernel_init_len = n_features

        if n_dims > self.kernel_init_len:
            raise ValueError("X should have more n_dims than n_features")
        if self.n_kernels < self.kernel_init_len:
            print("Warning: X has more features than dictionary kernels")
            # raise ValueError('X has more features than dictionary kernels')

        code, dictionary, err = multivariate_dict_learning(
            X,
            self.n_kernels,
            n_nonzero_coefs=self.n_nonzero_coefs,
            tol=self.tol,
            learning_rate=self.learning_rate,
            n_jobs=self.n_jobs,
            dict_init=self.dict_init,
            verbose=self.verbose,
            kernel_init_len=self.kernel_init_len,
            random_state=random_state,
            max_iter=self.max_iter,
            callback=self.callback,
            dict_obj=self,
        )
        self.kernels_ = list(dictionary)
        self.error_ = err

        # if self.verbose >= 1:
        #     print ('\nEnd of fit')
        #     print (self.kernels_[7])
        #     print ()

        return self


class MiniBatchMultivariateDictLearning(BaseEstimator, MultivariateDictMixin):
    """Mini-batch multivariate dictionary learning

    Finds a dictionary (a set of atoms represented as matrices) that can
    best be used to represent data using a sparse code.

    Solves the optimization problem::

       (U^*,V^*) = argmin 0.5 || Y - U V ||_F^2 s.t. || U ||_0 < k
                    (U,V)
                    with || V_i ||_F = 1 for all  0 <= i < n_kernels

    Parameters
    ----------
    n_kernels : int,
        number of dictionary elements to extract

    n_iter : int,
        total number of iterations to perform

    n_nonzero_coefs : int, ``0.1 * n_features`` by default
        Number of nonzero coefficients to target in each column of the
        solution.

    n_jobs : int,
        number of parallel jobs to run

    dict_init : list of arrays
        Initial value of the dictionary for warm restart scenarios
        The dictionary is a list of n_kernels elements. Each element
        is a convolution kernel, i.e. an array of shape (k, n_dims),
        where k <= n_features and is kernel specific.

    verbose :
        degree of verbosity of the printed output

    learning_rate : float
        Value for the learning rate exponent, of the form
        i**learning_rate, where i is the iteration.

    batch_size : int,
        number of samples in each mini-batch

    shuffle : bool,
        whether to shuffle the samples before forming batches

    random_state : int or RandomState
        Pseudo number generator state used for random sampling.

    callback: function
        Function called during learning process, the only parameter
        is a dictionary containing all local variables

    Attributes
    ----------
    `kernels_` : list of arrays
        Kernels extracted from the data


    Notes
    -----
    **References:**

    Multivariate Temporal Dictionary Learning for EEG,
    J. Neurosci Methods, 2013

    Shift & 2D Rotation Invariant Sparse Coding for
    Multivariate Signals, IEEE TSP, 2012.

    See also
    --------
    SparseCoder
    DictionaryLearning
    SparsePCA
    MiniBatchSparsePCA
    """

    def __init__(
        self,
        n_kernels=None,
        n_iter=10,
        n_jobs=1,
        batch_size=None,
        shuffle=True,
        dict_init=None,
        n_nonzero_coefs=None,
        verbose=False,
        kernel_init_len=None,
        random_state=None,
        learning_rate=None,
        callback=None,
    ):

        self._set_mdla_params(
            n_kernels, n_nonzero_coefs, kernel_init_len, n_jobs, learning_rate
        )
        self.n_iter = n_iter
        if dict_init is not None:
            self.dict_init = _normalize(list(dict_init))
        else:
            self.dict_init = None
        self.verbose = verbose
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.random_state = random_state
        self.callback = callback

    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features, n_dims)
            Training matrices, where n_samples in the number of samples
            Each sample is a matrix of shape (n_features, n_dims) with
            n_features >= n_dims.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        random_state = check_random_state(self.random_state)
        X = array3d(X)
        n_samples, n_features, n_dims = X.shape
        if hasattr(self, "kernels_"):
            self.dict_init = _normalize(list(self.kernels_))
            if self.verbose:
                print("\nWarm restart with existing kernels_")
                # print (self.kernels_[7])
                # print ('')
        if self.dict_init is not None:
            self.n_kernels = len(self.dict_init)
        elif self.n_kernels is None:
            self.n_kernels = 2 * n_features
        if self.kernel_init_len is None:
            self.kernel_init_len = n_features

        if n_dims > self.kernel_init_len:
            raise ValueError("X should have more n_dims than n_features")
        if self.n_kernels < self.kernel_init_len:
            print("Warning: X has more features than dictionary kernels")
            # raise ValueError('X has more features than dictionary kernels')
        # if n_samples < self.n_kernels:
        #     raise ValueError('There is more kernel (%d) than samples (%d)' % (self.n_kernels, n_samples))

        dictionary, e = multivariate_dict_learning_online(
            X,
            self.n_kernels,
            n_nonzero_coefs=self.n_nonzero_coefs,
            n_iter=self.n_iter,
            n_jobs=self.n_jobs,
            dict_init=self.dict_init,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            verbose=self.verbose,
            kernel_init_len=self.kernel_init_len,
            random_state=random_state,
            learning_rate=self.learning_rate,
            callback=self.callback,
            dict_obj=self,
        )
        self.kernels_ = list(dictionary)
        self.iter_offset_ = self.n_iter
        self.error_ = e

        return self

    def partial_fit(self, X, y=None, iter_offset=None):
        """Updates the model using the data in X as a mini-batch.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features, n_dims)
            Training matrices, where n_samples in the number of samples
            Each sample is a matrix of shape (n_features, n_dims) with
            n_features >= n_dims.

        iter_offset: integer, optional
            The number of iteration on data batches that has been
            performed before this call to partial_fit. This is optional:
            if no number is passed, the memory of the object is
            used.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if not hasattr(self, "random_state_"):
            self.random_state_ = check_random_state(self.random_state)
        X = array3d(X)
        n_samples, n_features, n_dims = X.shape
        if hasattr(self, "kernels_"):
            self.dict_init = _normalize(list(self.kernels_))
        if self.dict_init is not None:
            self.n_kernels = len(self.dict_init)
        elif self.n_kernels is None:
            self.n_kernels = 2 * n_features
        if self.kernel_init_len is None:
            self.kernel_init_len = n_features

        if n_dims > self.kernel_init_len:
            raise ValueError("X should have more n_dims than n_features")
        if self.n_kernels < self.kernel_init_len:
            print("Warning: X has more features than dictionary kernels")
            # raise ValueError('X has more features than dictionary kernels')
        # if n_samples < self.n_kernels:
        #     raise ValueError('There is more kernel than samples')

        if iter_offset is None:
            iter_offset = getattr(self, "iter_offset_", 0)

        dictionary, e = multivariate_dict_learning_online(
            X,
            self.n_kernels,
            n_nonzero_coefs=self.n_nonzero_coefs,
            n_iter=self.n_iter,
            n_jobs=self.n_jobs,
            dict_init=self.dict_init,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            verbose=self.verbose,
            iter_offset=iter_offset,
            kernel_init_len=self.kernel_init_len,
            learning_rate=self.learning_rate,
            random_state=self.random_state_,
            callback=self.callback,
            dict_obj=self,
        )
        self.kernels_ = list(dictionary)
        self.iter_offset_ = iter_offset + self.n_iter
        self.error_ = e

        return self


def array3d(X, dtype=None, order=None, copy=False, force_all_finite=True):
    """Returns at least 3-d array with data from X"""
    if sp.issparse(X):
        raise TypeError(
            "A sparse matrix was passed, but dense data "
            "is required. Use X.toarray() to convert to dense."
        )
    X_3d = np.array(np.atleast_3d(X), dtype=dtype, order=order, copy=copy)
    if type(X) is np.ndarray and X.ndim == 2:
        X_3d = X_3d.swapaxes(0, 2)
        X_3d = X_3d.swapaxes(1, 2)
    if force_all_finite:
        assert_all_finite(X_3d)
    return X_3d
