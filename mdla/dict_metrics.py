# -*- coding: utf-8 -*-
"""
The `dict_metrics` module implements utilities to compare
frames and dictionaries.

This module implements several criteria and metrics to compare different sets
of atoms. This module is primarily focused on multivariate kernels and
atoms.
"""

# Authors: Sylvain Chevallier <sylvain.chevallier@uvsq.fr>
# License: GPL v3

# TODO: add docstring to criteria fonction
#       verify Fubini-Study scale parameter
#       verify beta dist behavior, seems like 1-bd
#       change scale behavior, replace 1-d with d !

import cvxopt as co
import cvxopt.solvers as solv
import numpy as np
import scipy.linalg as sl
from numpy import (
    NaN,
    abs,
    all,
    arccos,
    arcsin,
    argmax,
    array,
    atleast_2d,
    concatenate,
    infty,
    max,
    min,
    ones,
    ones_like,
    sqrt,
    trace,
    unravel_index,
    zeros,
    zeros_like,
)
from numpy.linalg import det, norm, svd


def _kernel_registration(this_kernel, dictionary, g):
    k_len = this_kernel.shape[0]
    n_kernels = len(dictionary)
    k_max_len = array([i.shape[0] for i in dictionary]).max()

    m_dist = ones((n_kernels, k_max_len - k_len + 1)) * infty
    m_corr = zeros((n_kernels, k_max_len - k_len + 1))
    for i, kernel in enumerate(dictionary):  # kernel loop
        ks = kernel.shape[0]
        # for t in range(k_max_len-k_len+1): # convolution loop
        for t in range(ks - k_len + 1):  # convolution loop
            # print ("t = ", t, "and l =", l)
            # print ("kernel = ", kernel.shape,
            #        "and kernel[t:t+l,:] = ", kernel[t:t+k_len,:].shape)
            m_dist[i, t] = g(this_kernel, kernel[t : t + k_len, :])
            m_corr[i, t] = trace(this_kernel.T.dot(kernel[t : t + k_len, :])) / (
                norm(this_kernel, "fro") * norm(kernel[t : t + k_len, :], "fro")
            )
    return m_dist, m_corr


def principal_angles(A, B):
    """Compute the principal angles between subspaces A and B.

    The algorithm for computing the principal angles is described in :
    A. V. Knyazev and M. E. Argentati,
    Principal Angles between Subspaces in an A-Based Scalar Product:
    Algorithms and Perturbation Estimates. SIAM Journal on Scientific Computing,
    23 (2002), no. 6, 2009-2041.
    http://epubs.siam.org/sam-bin/dbq/article/37733
    """
    # eps = np.finfo(np.float64).eps**.981
    # for i in range(A.shape[1]):
    #     normi = la.norm(A[:,i],np.inf)
    #     if normi > eps: A[:,i] = A[:,i]/normi
    # for i in range(B.shape[1]):
    #     normi = la.norm(B[:,i],np.inf)
    #     if normi > eps: B[:,i] = B[:,i]/normi
    QA = sl.orth(A)
    QB = sl.orth(B)
    _, s, Zs = svd(QA.T.dot(QB), full_matrices=False)
    s = np.minimum(s, ones_like(s))
    theta = np.maximum(np.arccos(s), np.zeros_like(s))
    V = QB.dot(Zs)
    idxSmall = s > np.sqrt(2.0) / 2.0
    if np.any(idxSmall):
        RB = V[:, idxSmall]
        _, x, _ = svd(RB - QA.dot(QA.T.dot(RB)), full_matrices=False)
        thetaSmall = np.flipud(
            np.maximum(arcsin(np.minimum(x, ones_like(x))), zeros_like(x))
        )
        theta[idxSmall] = thetaSmall
    return theta


def chordal_principal_angles(A, B):
    """
    chordal_principal_angles(A, B) Compute the chordal distance based on
    principal angles.
    Compute the chordal distance based on principal angles between A and B
    as :math:`d=\sqrt{ \sum_i \sin^2 \theta_i}`
    """
    return sqrt(np.sum(np.sin(principal_angles(A, B)) ** 2))


def chordal(A, B):
    """
    chordal(A, B) Compute the chordal distance
    Compute the chordal distance between A and B
    as d=\sqrt{K - ||\bar{A}^T\bar{B}||_F^2}
    where K is the rank of A and B, || . ||_F is the Frobenius norm,
    \bar{A} is the orthogonal basis associated with A and the same goes for B.
    """
    if A.shape != B.shape:
        raise ValueError(
            f"Atoms have not the same dimension ({A.shape} and {B.shape}). Error raised"
            f"in chordal(A, B)",
        )

    if np.allclose(A, B):
        return 0.0
    else:
        d2 = A.shape[1] - norm(sl.orth(A).T.dot(sl.orth(B)), "fro") ** 2
        if d2 < 0.0:
            return sqrt(abs(d2))
        else:
            return sqrt(d2)


def fubini_study(A, B):
    """
    fubini_study(A, B) Compute the Fubini-Study distance
    Compute the Fubini-Study distance based on principal angles between A and B
    as d=\acos{ \prod_i \theta_i}
    """
    if A.shape != B.shape:
        raise ValueError(
            f"Atoms have different dim ({A.shape} and {B.shape}). Error raised in"
            f"fubini_study(A, B)",
        )
    if np.allclose(A, B):
        return 0.0
    return arccos(det(sl.orth(A).T.dot(sl.orth(B))))


def binet_cauchy(A, B):
    """Compute the Binet-Cauchy distance
    Compute the Binet-Cauchy distance based on principal angles between A
    and B with d=\sqrt{ 1 - \prod_i \cos^2 \theta_i}
    """
    theta = principal_angles(A, B)
    return sqrt(1.0 - np.prod(np.cos(theta) ** 2))


def geodesic(A, B):
    """
    geodesic (A, B) Compute the arc length or geodesic distance
    Compute the arc length or geodesic distance based on principal angles between A
    and B with d=\sqrt{ \sum_i \theta_i^2}
    """
    theta = principal_angles(A, B)
    return norm(theta)


def frobenius(A, B):
    if A.shape != B.shape:
        raise ValueError(
            f"Atoms have different dim ({A.shape} and {B.shape}). Error raised in"
            f"frobenius(A, B)",
        )
    return norm(A - B, "fro")


def abs_euclidean(A, B):
    if (A.ndim != 1 and A.shape[1] != 1) or (B.ndim != 1 and B.shape[1] != 1):
        raise ValueError(
            f"Atoms are not univariate ({A.shape} and {B.shape}). Error raised"
            f"in abs_euclidean(A, B)",
        )
    if np.allclose(A, B):
        return 0.0
    else:
        return sqrt(2.0 * (1.0 - np.abs(A.T.dot(B))))


def euclidean(A, B):
    if (A.ndim != 1 and A.shape[1] != 1) or (B.ndim != 1 and B.shape[1] != 1):
        raise ValueError(
            f"Atoms are not univariate ({A.shape} and {B.shape}). Error raised in"
            f"euclidean(A, B)",
        )
    if np.allclose(A, B):
        return 0.0
    else:
        return sqrt(2.0 * (1.0 - A.T.dot(B)))


def _valid_atom_metric(gdist):
    """Verify that atom metric exist and return the correct function"""
    if gdist == "chordal":
        return chordal
    elif gdist == "chordal_principal_angles":
        return chordal_principal_angles
    elif gdist == "fubinistudy":
        return fubini_study
    elif gdist == "binetcauchy":
        return binet_cauchy
    elif gdist == "geodesic":
        return geodesic
    elif gdist == "frobenius":
        return frobenius
    elif gdist == "abs_euclidean":
        return abs_euclidean
    elif gdist == "euclidean":
        return euclidean
    else:
        return None


def _scale_metric(gdist, d, D1):
    if (
        gdist == "chordal"
        or gdist == "chordal_principal_angles"
        or gdist == "fubinistudy"
        or gdist == "binetcauchy"
        or gdist == "geodesic"
    ):
        # TODO: scale with max n_features
        return d / sqrt(D1[0].shape[0])
    elif gdist == "frobenius":
        return d / sqrt(2.0)
    else:
        return d


def _compute_gdm(D1, D2, g):
    """Compute ground distance matrix from dictionaries D1 and D2

    Distance g acts as ground distance.
    A kernel registration is applied if dictionary atoms do not have
    the same size.
    """
    # Do we need a registration? If kernel do not have the same shape, yes
    if not all(array([i.shape[0] for i in D1 + D2]) == D1[0].shape[0]):
        # compute correlation and distance matrices
        k_dim = D1[0].shape[1]
        # minl = np.array([i.shape[1] for i in D1+D2]).min()
        max_l1 = array([i.shape[0] for i in D1]).max()
        max_l2 = array([i.shape[0] for i in D2]).max()
        if max_l2 > max_l1:
            Da = D1
            Db = D2
            max_l = max_l2
        else:
            Da = D2
            Db = D1
            max_l = max_l1
        # Set all Db atom to largest value
        Dbe = []
        for i in range(len(Db)):
            k_l = Db[i].shape[0]
            Dbe.append(concatenate((zeros((max_l - k_l, k_dim)), Db[i]), axis=0))
        gdm = zeros((len(Da), len(Db)))
        for i in range(len(Da)):
            m_dist, m_corr = _kernel_registration(Da[i], Dbe, g)
            k_l = Da[i].shape[0]
            # m_dist, m_corr = _kernel_registration(np.concatenate((zeros((np.int(np.floor((max_l-k_l)/2.)), k_dim)), Da[i], zeros((np.int(np.ceil((max_l-k_l)/2.)), k_dim))), axis=0), Dbe, g)
            for j in range(len(Dbe)):
                gdm[i, j] = m_dist[
                    j, unravel_index(abs(m_corr[j, :]).argmax(), m_corr[j, :].shape)
                ]
    else:
        # all atoms have the same length, no registration
        gdm = zeros((len(D1), len(D2)))
        for i in range(len(D1)):
            for j in range(len(D2)):
                gdm[i, j] = g(D1[i], D2[j])
    return gdm


def hausdorff(D1, D2, gdist, scale=False):
    """
    Compute the Hausdorff distance between two sets of elements, here
    dictionary atoms, using a ground distance.
    Possible choice are "chordal", "fubinistudy", "binetcauchy", "geodesic",
    "frobenius", "abs_euclidean" or "euclidean".
    The scale parameter changes the return value to be between 0 and 1.
    """
    g = _valid_atom_metric(gdist)
    if g is None:
        print("Unknown ground distance, exiting.")
        return NaN
    gdm = _compute_gdm(D1, D2, g)
    d = max([max(min(gdm, axis=0)), max(min(gdm, axis=1))])
    if not scale:
        return d
    else:
        return _scale_metric(gdist, d, D1)


def emd(D1, D2, gdist, scale=False):
    """
    Compute the Earth Mover's Distance (EMD) between two sets of elements,
    here dictionary atoms, using a ground distance.
    Possible choice are "chordal", "fubinistudy", "binetcauchy", "geodesic",
    "frobenius", "abs_euclidean" or "euclidean".
    The scale parameter changes the return value to be between 0 and 1.
    """
    g = _valid_atom_metric(gdist)
    if g is None:
        print("Unknown ground distance, exiting.")
        return NaN
    # if gdist == "chordal":
    #     g = chordal
    # elif gdist == "chordal_principal_angles":
    #     g = chordal_principal_angles
    # elif gdist == "fubinistudy":
    #     g = fubini_study
    # elif gdist == "binetcauchy":
    #     g = binet_cauchy
    # elif gdist == "geodesic":
    #     g = geodesic
    # elif gdist == "frobenius":
    #     g = frobenius
    # elif gdist == "abs_euclidean":
    #     g = abs_euclidean
    # elif gdist == "euclidean":
    #     g = euclidean
    # else:
    #     print 'Unknown ground distance, exiting.'
    #     return NaN

    # # Do we need a registration? If kernel do not have the same shape, yes
    # if not np.all(np.array([i.shape[0] for i in D1+D2]) == D1[0].shape[0]):
    #     # compute correlation and distance matrices
    #     k_dim = D1[0].shape[1]
    #     # minl = np.array([i.shape[1] for i in D1+D2]).min()
    #     max_l1 = np.array([i.shape[0] for i in D1]).max()
    #     max_l2 = np.array([i.shape[0] for i in D2]).max()
    #     if max_l2 > max_l1:
    #         Da = D1
    #         Db = D2
    #         max_l = max_l2
    #     else:
    #         Da = D2
    #         Db = D1
    #         max_l = max_l1
    #     Dbe = []
    #     for i in range(len(Db)):
    #         k_l = Db[i].shape[0]
    #         Dbe.append(np.concatenate((zeros((max_l-k_l, k_dim)), Db[i]), axis=0))
    #     gdm = zeros((len(Da), len(Db)))
    #     for i in range(len(Da)):
    #         k_l = Da[i].shape[0]
    #         m_dist, m_corr = _kernel_registration(np.concatenate((zeros(( np.int(np.floor((max_l-k_l)/2.)), k_dim)), Da[i], zeros((np.int(np.ceil((max_l-k_l)/2.)), k_dim))), axis=0), Dbe, g)
    #         for j in range(len(Dbe)):
    #             gdm[i,j] = m_dist[j, np.unravel_index(np.abs(m_corr[j,:]).argmax(), m_corr[j,:].shape)]
    # else:
    #     # all atoms have the same length, no registration
    #     gdm = np.zeros((len(D1), len(D2)))
    #     for i in range(len(D1)):
    #         for j in range(len(D2)):
    #             gdm[i,j] = g(D1[i], D2[j])
    gdm = _compute_gdm(D1, D2, g)

    c = co.matrix(gdm.flatten(order="F"))
    G1 = co.spmatrix([], [], [], (len(D1), len(D1) * len(D2)))
    G2 = co.spmatrix([], [], [], (len(D2), len(D1) * len(D2)))
    G3 = co.spmatrix(-1.0, range(len(D1) * len(D2)), range(len(D1) * len(D2)))
    for i in range(len(D1)):
        for j in range(len(D2)):
            k = j + (i * len(D2))
            G1[i, k] = 1.0
            G2[j, k] = 1.0
    G = co.sparse([G1, G2, G3])
    h1 = co.matrix(1.0 / len(D1), (len(D1), 1))
    h2 = co.matrix(1.0 / len(D2), (len(D2), 1))
    h3 = co.spmatrix([], [], [], (len(D1) * len(D2), 1))
    h = co.matrix([h1, h2, h3])
    A = co.matrix(1.0, (1, len(D1) * len(D2)))
    b = co.matrix([1.0])

    co.solvers.options["show_progress"] = False
    sol = solv.lp(c, G, h, A, b)
    d = sol["primal objective"]

    if not scale:
        return d
    else:
        return _scale_metric(gdist, d, D1)
    # if (gdist == "chordal" or gdist == "chordal_principal_angles" or
    #     gdist == "fubinistudy" or gdist == "binetcauchy" or
    #     gdist == "geodesic"):
    #     return d/sqrt(D1[0].shape[0])
    # elif gdist == "frobenius":
    #     return d/sqrt(2.)
    # else:
    #     return d


def _multivariate_correlation(s, D):
    """Compute correlation between multivariate atoms

    Compute the correlation between a multivariate atome s and dictionary D
    as the sum of the correlation in each n_dims dimensions.
    """
    n_features = s.shape[0]
    n_dims = s.shape[1]
    n_kernels = len(D)
    corr = np.zeros((n_kernels, n_features))
    for k in range(n_kernels):  # for all atoms
        corrTmp = 0
        for j in range(n_dims):  # for all dimensions
            corrTmp += np.correlate(s[:, j], D[k][:, j])
        corr[k, : len(corrTmp)] = corrTmp
    return corr


def detection_rate(ref, recov, threshold):
    """Compute the detection rate between reference and recovered dictionaries

    The reference ref and the recovered recov are univariate or multivariate
    dictionaries. An atom a of the ref dictionary is considered as recovered if
    $c < threshold$ with $c = argmax_{r \in R} |<a, r>|$, that is the absolute
    value of the maximum correlation between a and any atom r of the recovered
    dictionary R is above a given threshold.
    The process is iterative and an atom r could be matched only once with an
    atom a of the reference dictionary. In other word, each atom a is matched
    with a different atom r.
    """
    n_kernels_ref, n_kernels_recov = len(ref), len(recov)
    n_features = ref[0].shape[0]
    if ref[0].ndim == 1:
        n_dims = 1
        for k in range(n_kernels_ref):
            ref[k] = atleast_2d(ref[k]).T
    else:
        n_dims = ref[0].shape[1]
    if recov[0].ndim == 1:
        for k in range(n_kernels_recov):
            recov[k] = atleast_2d(recov[k]).T
    dr = 0
    corr = zeros((n_kernels_ref, n_kernels_recov))
    for k in range(n_kernels_ref):
        c_tmp = _multivariate_correlation(
            concatenate(
                (zeros((n_features, n_dims)), ref[k], zeros((n_features, n_dims))), axis=0
            ),
            recov,
        )
        for j in range(n_kernels_recov):
            idx_max = argmax(abs(c_tmp[j, :]))
            corr[k, j] = c_tmp[j, idx_max]
    c_local = np.abs(corr.copy())
    for _ in range(n_kernels_ref):
        max_corr = c_local.max()
        if max_corr >= threshold:
            dr += 1
        idx_max = np.unravel_index(c_local.argmax(), c_local.shape)
        c_local[:, idx_max[1]] = zeros(n_kernels_ref)
        c_local[idx_max[0], :] = zeros(n_kernels_recov)
    return float(dr) / n_kernels_recov * 100.0


def _convert_array(ref, recov):
    if ref[0].ndim == 1:
        for k in range(len(ref)):
            ref[k] = atleast_2d(ref[k]).T
    if recov[0].ndim == 1:
        for k in range(len(recov)):
            recov[k] = atleast_2d(recov[k]).T
    D1 = np.array(ref)
    D2 = np.array(recov)
    M = D1.shape[0]
    N = D1.shape[1]
    D1 = D1.reshape((M, N))
    D2 = D2.reshape((M, N))
    return D1, D2, M


def precision_recall(ref, recov, threshold):
    """Compute precision and recall for recovery experiment"""
    D1, D2, M = _convert_array(ref, recov)
    corr = D1.dot(D2.T)
    precision = float((np.max(corr, axis=0) > threshold).sum()) / float(M)
    recall = float((np.max(corr, axis=1) > threshold).sum()) / float(M)
    return precision * 100.0, recall * 100.0


def precision_recall_points(ref, recov):
    """Compute the precision and recall for each atom in a recovery experiment"""
    # if ref[0].ndim == 1:
    #     for k in range(len(ref)):
    #         ref[k] = atleast_2d(ref[k]).T
    # if recov[0].ndim == 1:
    #     for k in range(len(recov)):
    #         recov[k] = atleast_2d(recov[k]).T
    # D1 = np.array(ref)
    # D2 = np.array(recov)
    # M = D1.shape[0]
    # N = D1.shape[1]
    # D1 = D1.reshape((M, N))
    # D2 = D2.reshape((M, N))
    D1, D2, _ = _convert_array(ref, recov)
    corr = D1.dot(D2.T)
    precision = np.max(corr, axis=0)
    recall = np.max(corr, axis=1)
    return precision, recall


def beta_dist(D1, D2):
    """Compute the Beta-distance proposed by Skretting and Engan

    The beta-distance is:
    $\beta(D1, D2)=1/(M1+M2)(\sum_j \beta(D1, d^2_j)+\sum_j \beta(D2, d^1_j))$
    with $\beta(D, x) = arccos(\max_i |d^T_i x|/||x||)$
    as proposed in:
    Karl Skretting and Kjersti Engan,
    Learned dictionaries for sparse image representation: properties and results,
    SPIE, 2011.
    """
    if D1[0].shape != D2[0].shape:
        raise ValueError(
            f"Dictionaries have different dim : {D1[0].shape} and {D2[0].shape}."
        )
    D1 = np.array(D1)
    M1 = D1.shape[0]
    N = D1.shape[1]
    D1 = D1.reshape((M1, N))
    D2 = np.array(D2)
    M2 = D2.shape[0]
    D2 = D2.reshape((M2, N))
    corr = D1.dot(D2.T)
    if np.allclose(np.max(corr, axis=0), ones(M2)) and np.allclose(
        np.max(corr, axis=1), ones(M1)
    ):
        return 0.0
    return (
        np.sum(np.arccos(np.max(corr, axis=0))) + np.sum(np.arccos(np.max(corr, axis=1)))
    ) / (M1 + M2)
