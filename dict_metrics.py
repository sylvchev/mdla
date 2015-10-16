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

import numpy as np
import numpy.linalg as la
import scipy.linalg as sl
import cvxopt as co
import cvxopt.solvers as solv

from numpy import infty, ones, zeros, ones_like, zeros_like, NaN
from numpy import arccos, arcsin, sqrt, array

def _kernel_registration(this_kernel, dictionary, g):
    k_len = this_kernel.shape[0]
    n_kernels = len(dictionary)
    k_max_len = array([i.shape[0] for i in dictionary]).max()
    
    m_dist = ones((n_kernels, k_len))*infty
    m_corr = zeros((n_kernels, k_len))
    for i, kernel in enumerate(dictionary): # kernel loop
        l = kernel.shape[0]
        # mdistTmp = np.ones((1, kl))*np.infty
        # mcorrTmp = np.zeros((1, kl))
        for t in range(k_max_len-k_len+1): # convolution loop
            m_dist[i, t] = g(this_kernel[:,t:t+l], kernel)
            m_corr[i, t] = np.trace(this_kernel[t:t+l,:].T.dot(kernel)) / (la.norm(this_kernel[t:t+l,:], 'fro') * la.norm(kernel, 'fro'))
    return m_dist, m_corr

def chordalPA(A, B):
    '''
    chordalPA(A, B) Compute the chordal distance based on principal angles
    Compute the chordal distance based on principal angles between A and B
    as d=\sqrt{ \sum_i \sin^2 \theta_i}
    '''
    return sqrt(np.sum(np.sin(principalAngles(A,B))**2))

def chordal(A, B):
    '''
    chordal(A, B) Compute the chordal distance
    Compute the chordal distance between A and B
    as d=\sqrt{K - ||\bar{A}^T\bar{B}||_F^2}
    where K is the rank of A and B, || . ||_F is the Frobenius norm,
    \bar{A} is the orthogonal basis associated with A and the same goes for B.
    '''
    if (A.shape != B.shape):
        raise AtomSizeError ('Atoms have not the same dimension (', A.shape, ' and ', B.shape,'). Error raised in chordal(A, B)')
    # Atoms should have more lines than columns
    # if (A.shape[1] > A.shape[0]):
    #     A = A.T
    #     B = B.T
    if np.allclose(A, B): return 0.
    # else: return np.sqrt(A.T.shape[1] - la.norm(sl.orth(A.T).T.dot(sl.orth(B.T)), 'fro')**2)
    else: 
        d2 = A.shape[1] - la.norm(sl.orth(A).T.dot(sl.orth(B)), 'fro')**2
        if d2 < 0.: return sqrt(abs(d2))
        else: return sqrt(d2)
    # return np.sqrt(A.shape[0] - la.norm(sl.orth(A).T.dot(sl.orth(B)), 'fro')**2)
        
def fubiniStudy(A, B):
    '''
    fubiniStudy(A, B) Compute the Fubini-Study distance
    Compute the Fubini-Study distance based on principal angles between A and B
    as d=\acos{ \prod_i \theta_i}
    '''
    # if (A.shape[1] > A.shape[0]):
    #     A = A.T
    #     B = B.T
    return arccos(la.det(sl.orth(A).T.dot(sl.orth(B))))

def principalAngles(A, B):
    '''Compute the principal angles between subspaces A and B.

    The algorithm for computing the principal angles is described in :
    A. V. Knyazev and M. E. Argentati,
    Principal Angles between Subspaces in an A-Based Scalar Product: 
    Algorithms and Perturbation Estimates. SIAM Journal on Scientific Computing, 
    23 (2002), no. 6, 2009-2041.
    http://epubs.siam.org/sam-bin/dbq/article/37733
    '''    
    # eps = np.finfo(np.float64).eps**.981
    # for i in range(A.shape[1]):
    #     normi = la.norm(A[:,i],np.inf)
    #     if normi > eps: A[:,i] = A[:,i]/normi
    # for i in range(B.shape[1]):
    #     normi = la.norm(B[:,i],np.inf)
    #     if normi > eps: B[:,i] = B[:,i]/normi
    QA = sl.orth(A)
    QB = sl.orth(B)
    _, s, Zs = la.svd(QA.T.dot(QB), full_matrices=False)
    s = np.minimum(s, ones_like(s))
    theta = np.maximum(np.arccos(s), np.zeros_like(s))
    V = QB.dot(Zs)
    idxSmall = s > np.sqrt(2.)/2.
    if np.any(idxSmall):
        RB = V[:,idxSmall]
        _, x, _ = la.svd(RB-QA.dot(QA.T.dot(RB)),full_matrices=False)
        thetaSmall = np.flipud(np.maximum(arcsin(np.minimum(x, ones_like(x))), zeros_like(x)))
        theta[idxSmall] = thetaSmall
    return theta

def binetCauchy(A, B):
    '''Compute the Binet-Cauchy distance
    Compute the Binet-Cauchy distance based on principal angles between A
    and B with d=\sqrt{ 1 - \prod_i \cos^2 \theta_i}
    '''    
    theta = principalAngles(A, B)
    return sqrt(1. - np.prod(np.cos(theta)**2))

def geodesic(A, B):
    '''
    geodesic (A, B) Compute the arc length or geodesic distance
    Compute the arc length or geodesic distance based on principal angles between A
    and B with d=\sqrt{ \sum_i \theta_i^2}
    '''
    theta = principalAngles(A, B)
    return la.norm(theta)

def frobeniusBased(A, B):
    return la.norm(np.abs(A)-np.abs(B), 'fro')

def absEuclidean(A, B):
    return 2.*(1.-np.abs(A.T.dot(B)))

def euclidean (A, B):
    return 2.*(1.-A.T.dot(B))

def hausdorff(D1, D2, gdist, scale=False):
    '''
    Compute the Hausdorff distance between two sets of elements, here
    dictionary atoms, using a ground distance.
    Possible choice are "chordal", "fubinistudy", "binetcauchy", "geodesic"
    or "frobeniusBased".
    The scale parameter changes the return value to be between 0 and 100.
    '''
    if   gdist == "chordal":
        g = chordal
    elif   gdist == "chordalPA":
        g = chordalPA
    elif gdist == "fubinistudy":
        g = fubiniStudy
    elif gdist == "binetcauchy":
        g = binetCauchy
    elif gdist == "geodesic":
        g = geodesic
    elif gdist == "frobeniusBased":
        g = frobeniusBased
    elif gdist == "absEuclidean":
        g = absEuclidean
    elif gdist == "euclidean":
        g = euclidean
    else:
        print 'Unknown ground distance, exiting.'
        return NaN

    # Do we need a registration? If kernel do not have the same shape, yes
    if not np.all(array([i.shape[0] for i in D1+D2]) == D1[0].shape[0]):
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
        # maxl = np.array([i.shape[1] for i in D1+D2]).max()
        gdm = zeros((len(Da), len(Db)))
        for i in range(len(Da)):
            k_l = Da[i].shape[0]
            m_dist, m_corr = _kernel_registration(np.concatenate((zeros(( np.int(np.floor((max_l-k_l)/2.)), k_dim)), Da[i], zeros(( np.int(np.ceil((max_l-k_l)/2.)), k_dim))), axis=0), Db, g)
            for j in range(len(Db)):
                gdm[i,j] = m_dist[j, np.unravel_index(np.abs(m_corr[j,:]).argmax(), m_corr[j,:].shape)]
    else:    
        # all atoms have the same length, no registration
        gdm = np.zeros((len(D1), len(D2)))
        for i in range(len(D1)):
            for j in range(len(D2)):
                gdm[i,j] = g(D1[i], D2[j])

    d =  max(np.max(np.min(gdm, axis=0)), np.max(np.min(gdm, axis=1)))
    if not scale: return d
    else:
        if gdist == "chordal" or gdist == "chordalPA":
            return (sqrt(D1[0].shape[0])-d)/sqrt(D1[0].shape[0])*100.
        elif gdist == "fubinistudy":
            return (sqrt(D1[0].shape[0])-d)/sqrt(D1[0].shape[0])*100.
        elif gdist == "binetcauchy":
            return (sqrt(D1[0].shape[0])-d)/sqrt(D1[0].shape[0])*100.
        elif gdist == "geodesic":
            return (sqrt(D1[0].shape[0])-d)/sqrt(D1[0].shape[0])*100.
        elif gdist == "frobeniusBased":
            return (sqrt(2.)-d)/sqrt(2.)*100.
        else:
            return d*100.
        
def emd(D1, D2, gdist, scale=False):
    '''
    Compute the Earth Mover's Distance (EMD) between two sets of elements,
    here dictionary atoms, using a ground distance.
    Possible choice are "chordal", "fubinistudy", "binetcauchy", "geodesic"
    or "frobeniusBased".
    The scale parameter changes the return value to be between 0 and 100.
    '''
    if gdist == "chordal":
        g = chordal
    elif gdist == "chordalPA":
        g = chordalPA
    elif gdist == "fubinistudy":
        g = fubiniStudy
    elif gdist == "binetcauchy":
        g = binetCauchy
    elif gdist == "geodesic":
        g = geodesic
    elif gdist == "frobeniusBased":
        g = frobeniusBased
    elif gdist == "absEuclidean":
        g = absEuclidean
    elif gdist == "euclidean":
        g = euclidean
    else:
        print 'Unknown ground distance, exiting.'
        return NaN

    # Do we need a registration? If kernel do not have the same shape, yes
    if not np.all(np.array([i.shape[0] for i in D1+D2]) == D1[0].shape[0]):
        # compute correlation and distance matrices
        k_dim = D1[0].shape[1]
        # minl = np.array([i.shape[1] for i in D1+D2]).min()
        max_l1 = np.array([i.shape[0] for i in D1]).max()
        max_l2 = np.array([i.shape[0] for i in D2]).max()
        if max_l2 > max_l1:
            Da = D1
            Db = D2
            max_l = max_l2
        else:
            Da = D2
            Db = D1
            max_l = max_l1
        # maxl = np.array([i.shape[1] for i in D1+D2]).max()
        gdm = zeros((len(Da), len(Db)))
        for i in range(len(Da)):
            k_l = Da[i].shape[0]
            m_dist, m_corr = _kernel_registration(np.concatenate((zeros(( np.int(np.floor((max_l-k_l)/2.)), k_dim)), Da[i], zeros((np.int(np.ceil((max_l-k_l)/2.)), k_dim))), axis=0), Db, g)
            for j in range(len(Db)):
                gdm[i,j] = m_dist[j, np.unravel_index(np.abs(m_corr[j,:]).argmax(), m_corr[j,:].shape)]
    else:    
        # all atoms have the same length, no registration
        gdm = np.zeros((len(D1), len(D2)))
        for i in range(len(D1)):
            for j in range(len(D2)):
                gdm[i,j] = g(D1[i], D2[j])
            
    c = co.matrix(gdm.flatten(order='F'))
    G1 = co.spmatrix([], [], [], (len(D1), len(D1)*len(D2)))
    G2 = co.spmatrix([], [], [], (len(D2), len(D1)*len(D2)))
    G3 = co.spmatrix(-1., range(len(D1)*len(D2)), range(len(D1)*len(D2)))
    for i in range(len(D1)):
        for j in range(len(D2)):
            k = j+(i*len(D2))
            G1[i,k] = 1.
            G2[j,k] = 1.
    G = co.sparse([G1,G2,G3])
    h1 = co.matrix(1./len(D1), (len(D1), 1))
    h2 = co.matrix(1./len(D2), (len(D2), 1))
    h3 = co.spmatrix([], [], [], (len(D1)*len(D2), 1))
    h = co.matrix([h1, h2, h3])
    A = co.matrix(1., (1, len(D1)*len(D2)))
    b = co.matrix([1.])

    co.solvers.options['show_progress'] = False
    sol = solv.lp(c, G, h, A, b)
    d = sol['primal objective']

    if not scale: return d
    else:
        if gdist == "chordal" or gdist == "chordalPA":
            return (sqrt(D1[0].shape[1])-d)/sqrt(D1[0].shape[1])*100.
        elif gdist == "fubinistudy":
            return (sqrt(D1[0].shape[1])-d)/sqrt(D1[0].shape[1])*100.
        elif gdist == "binetcauchy":
            return (sqrt(D1[0].shape[1])-d)/sqrt(D1[0].shape[1])*100.
        elif gdist == "geodesic":
            return (sqrt(D1[0].shape[1])-d)/sqrt(D1[0].shape[1])*100.
        elif gdist == "frobeniusBased":
            return (sqrt(2.)-d)/sqrt(2.)*100.
        else:
            return d*100.

def computeCorrelation(s, D):
    corr = np.zeros((len(D), s.shape[1]))
    for i in range(len(D)): # for all atoms
        corrTmp = 0
        for j in range(s.shape[0]): # for all dimensions
            corrTmp += np.correlate(s[j,:],D[i][j,:])
        corr[i,:len(corrTmp)] = corrTmp
    return corr

def detectionRate(dictRef, dictLearn, threshold):
    nbDR = 0
    corr = np.zeros((len(dictRef), len(dictLearn)))
    for i in range(len(dictRef)):
        corrTmp = computeCorrelation(np.hstack((np.zeros((dictRef[0].shape[0],5)), dictRef[i], np.zeros((dictRef[0].shape[0],5)))), dictLearn)
        for j in range(len(dictLearn)):
            idxMax = np.argmax(np.abs(corrTmp[j,:]))
            corr[i,j] = corrTmp[j,idxMax]
    corrLocal = np.abs(corr.copy())
    for i in range(len(dictRef)):
        maxCorr = corrLocal.max()
        if maxCorr >= threshold: nbDR+=1
        idxMax = np.unravel_index(corrLocal.argmax(), corrLocal.shape)
        corrLocal[:,idxMax[1]] = np.zeros(len(dictRef))
        corrLocal[idxMax[0],:] = np.zeros(len(dictLearn))
    return float(nbDR)/len(dictLearn)*100.

def precisionRecall(dictRef, dictLearn, threshold):
    dr = 0
    D1 = np.array(dictRef)
    M  = D1.shape[0]
    N  = D1.shape[2]
    D1 = D1.reshape((M, N))
    D2 = np.array(dictLearn)
    D2 = D2.reshape((M, N))
    corr = D1.dot(D2.T)
    precision = float((np.max(corr, axis=0) > threshold).sum()) / float(M)
    recall    = float((np.max(corr, axis=1) > threshold).sum()) / float(M)
    return precision*100., recall*100.

def precisionRecallPoints(dictRef, dictLearn):
    dr = 0
    D1 = np.array(dictRef)
    M  = D1.shape[0]
    N  = D1.shape[2]
    D1 = D1.reshape((M, N))
    D2 = np.array(dictLearn)
    D2 = D2.reshape((M, N))
    corr = D1.dot(D2.T)
    precision = np.max(corr, axis=0)
    recall    = np.max(corr, axis=1)
    return precision, recall

def betaDist(dictRef, dictLearn):
    dr = 0
    D1 = np.array(dictRef)
    M  = D1.shape[0]
    N  = D1.shape[1]
    D1 = D1.reshape((M, N))
    D2 = np.array(dictLearn)
    D2 = D2.reshape((M, N))
    corr = D1.dot(D2.T)
    return (np.sum(np.arccos(np.max(corr, axis=0))) + np.sum(np.arccos(np.max(corr, axis=1)))) / (2*M)


