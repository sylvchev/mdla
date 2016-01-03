from __future__ import print_function
import numpy as np

from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_almost_equal
# from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal, assert_not_equal
from sklearn.utils.testing import assert_true
# from sklearn.utils.testing import assert_less
from sklearn.utils.testing import assert_raises
from numpy.linalg import norm
from numpy.random import randn
from numpy import arange, NaN, concatenate, zeros, ones, allclose
from dict_metrics import hausdorff, emd, _multivariate_correlation, \
     detection_rate, precision_recall, precision_recall_points, beta_dist

# TODO: verif scale, dist (and non existant dist), n_kernels,
#       univariate, multivariate

n_kernels, n_features, n_dims = 10, 5, 3
dm = [randn(n_features, n_dims) for i in range(n_kernels)]
for i in range(len(dm)):
    dm[i] /= norm(dm[i], 'fro')
du = [randn(n_features,) for i in range(n_kernels)]
for i in range(len(du)):
    du[i] /= norm(du[i])
    
gdm = ["chordal", "chordal_principal_angles", "fubinistudy", "binetcauchy",
       "geodesic", "frobenius"]
gdu = ["abs_euclidean", "euclidean"]
dist = [hausdorff, emd, detection_rate, precision_recall,
        precision_recall_points, beta_dist]

def test_scale():
    for m in [hausdorff, emd]:
        for g in gdm:
            print("for", g, ':')
            assert_almost_equal (0., m(dm, dm, g, scale=True))
        for g in gdu:
            print("for", g, ':')
            assert_almost_equal (0., m(du, du, g, scale=True))

def test_kernel_registration():
    dm2 = [randn(n_features+i/2, n_dims) for i in range(n_kernels)]
    for i in range(len(dm2)):
        dm2[i] /= norm(dm2[i], 'fro')

    for m in [hausdorff, emd]:
        assert_not_equal(0., m(dm, dm2, 'chordal'))
        assert_not_equal(0., m(dm2, dm, 'chordal'))

    dm3 = []
    for i in range(len(dm)):
        k_l = dm[i].shape[0]
        dm3.append(concatenate((zeros((4, 3)), dm[i]), axis=0))

    for m in [hausdorff, emd]:
        assert_almost_equal(0., m(dm, dm3, 'chordal'))
        assert_almost_equal(0., m(dm3, dm, 'chordal'))

    # max(dm3) > max(dm4), min(dm4) > min(dm3)
    dm4 = []
    for i in range(len(dm)):
        k_l = dm[i].shape[0]
        dm4.append(concatenate((zeros((i/2+1, 3)), dm[i]), axis=0))
    dm5 = []
    for i in range(len(dm)):
        k_l = dm[i].shape[0]
        dm5.append(concatenate((zeros((3, 3)), dm[i]), axis=0))
        
    for m in [hausdorff, emd]:
        assert_almost_equal(0., m(dm4, dm5, 'chordal'))
        assert_almost_equal(0., m(dm5, dm4, 'chordal'))

def test_unknown_metric():
    for m in [hausdorff, emd]:
        assert_true((m(dm, dm, 'inexistant_metric') is NaN))
                
def test_inhomogeneous_dims():
    idx = arange(n_dims)
    for g in ['chordal_principal_angles', 'binetcauchy', 'geodesic']:
        for i in range(n_dims, 0, -1):
            assert_almost_equal (0., emd(dm, [a[:,idx[:i]] for a in dm],
                                         g, scale=True))
            assert_almost_equal(0., hausdorff(dm, [a[:,idx[:i]] for a in dm],
                                              g, scale=True))
    for g in ["chordal", "fubinistudy", "frobenius"]:
        assert_raises(ValueError, emd, dm,
                      [a[:,:-1] for a in dm], g)
        assert_raises(ValueError, hausdorff, dm,
                      [a[:,:-1] for a in dm], g)

def test_univariate():
    for m in [hausdorff, emd]:
        for g in gdu:
            assert_raises (ValueError, emd, dm, dm, g)

def test_correlation():
    du2 = [randn(n_features,) for i in range(n_kernels)]
    for i in range(len(du2)):
        du2[i] /= norm(du2[i])
    dm2 = [randn(n_features, n_dims) for i in range(n_kernels)]
    for i in range(len(dm2)):
        dm2[i] /= norm(dm2[i])

    assert_equal(100., detection_rate(du, du, 0.97))
    assert_not_equal(100., detection_rate(du, du2, 0.99))
    assert_equal(100., detection_rate(dm, dm, 0.97))
    assert_not_equal(100., detection_rate(dm, dm2, 0.99))
    assert_equal((100., 100.), precision_recall(du, du, 0.97))
    assert_equal((0., 0.), precision_recall(du, du2, 0.99))
    assert_true(allclose(precision_recall_points(du, du),
                            (ones(len(du)), ones(len(du)))))
    assert_true(not allclose(precision_recall_points(du, du2),
                                (ones(len(du)), ones(len(du2)))))
        
def test_beta_dist():
    du2 = [randn(n_features, 1) for i in range(n_kernels)]
    for i in range(len(du2)):
        du2[i] /= norm(du2[i])

    assert_equal(0., beta_dist(du, du))
    assert_not_equal(0., beta_dist(du, du2))

    du2 = [randn(n_features+2, 1) for i in range(n_kernels)]
    for i in range(len(du2)):
        du2[i] /= norm(du2[i])
    assert_raises(ValueError, beta_dist, du, du2)

def test_beta_dict_length():
    du2 = [randn(n_features, 1) for i in range(n_kernels+2)]
    for i in range(len(du2)):
        du2[i] /= norm(du2[i])
        
    assert_not_equal(0., beta_dist(du, du2))
    
