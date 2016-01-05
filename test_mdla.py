from __future__ import print_function
import numpy as np

from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_almost_equal
# from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_true
# from sklearn.utils.testing import assert_less
from sklearn.utils.testing import assert_raises

from mdla import MultivariateDictLearning
from mdla import MiniBatchMultivariateDictLearning
from mdla import SparseMultivariateCoder
from mdla import reconstruct_from_code
from mdla import multivariate_sparse_encode

rng_global = np.random.RandomState(0)
n_samples, n_features, n_dims = 10, 5, 3
X = [rng_global.randn(n_features, n_dims) for i in range(n_samples)]

def test_mdla_shapes():
    n_kernels = 8
    dico = MultivariateDictLearning(n_kernels=n_kernels, random_state=0,
                                    max_iter=10, verbose=5).fit(X)
    for i in range(n_kernels):
        assert_true(dico.kernels_[i].shape == (n_features, n_dims))

    dico = MiniBatchMultivariateDictLearning(n_kernels=n_kernels,
                random_state=0, verbose=5, n_iter=10).fit(X)
    for i in range(n_kernels):
        assert_true(dico.kernels_[i].shape == (n_features, n_dims))
        
def test_multivariate_input_shape():
    n_kernels = 4
    dico = MultivariateDictLearning(n_kernels=n_kernels)
    assert_raises(ValueError, dico.fit, X)
    
    n_dims = 6
    Xw = [rng_global.randn(n_features, n_dims) for i in range(n_samples)]
    dico = MultivariateDictLearning(n_kernels=n_kernels)
    assert_raises(ValueError, dico.fit, Xw)

    n_kernels = 4
    dico = MiniBatchMultivariateDictLearning(n_kernels=n_kernels)
    assert_raises(ValueError, dico.fit, X)
    
    n_dims = 6
    Xw = [rng_global.randn(n_features, n_dims) for i in range(n_samples)]
    dico = MiniBatchMultivariateDictLearning(n_kernels=n_kernels)
    assert_raises(ValueError, dico.fit, Xw)
    
    n_kernels = 4
    dico = MiniBatchMultivariateDictLearning(n_kernels=n_kernels)
    assert_raises(ValueError, dico.partial_fit, X)
    
    n_dims = 6
    Xw = [rng_global.randn(n_features, n_dims) for i in range(n_samples)]
    dico = MiniBatchMultivariateDictLearning(n_kernels=n_kernels)
    assert_raises(ValueError, dico.partial_fit, Xw)

def test_mdla_normalization():
    n_kernels = 8
    dico = MultivariateDictLearning(n_kernels=n_kernels, random_state=0,
                                    max_iter=2, verbose=1).fit(X)
    for k in dico.kernels_:
        assert_almost_equal(np.linalg.norm(k, 'fro'), 1.)

    dico = MiniBatchMultivariateDictLearning(n_kernels=n_kernels,
                random_state=0, n_iter=2, verbose=1).fit(X)
    for k in dico.kernels_:
        assert_almost_equal(np.linalg.norm(k, 'fro'), 1.)

def test_callback():
    n_kernels = 8
    def my_callback(loc):
        d = loc['dict_obj']
        
    dico = MultivariateDictLearning(n_kernels=n_kernels, random_state=0,
                                    max_iter=2, n_nonzero_coefs=1,
                                    callback=my_callback)
    code = dico.fit(X).transform(X[0])
    assert_true(len(code[0]) <= 1)
    dico = MiniBatchMultivariateDictLearning(n_kernels=n_kernels,
                random_state=0, n_iter=2, n_nonzero_coefs=1,
                callback=my_callback)
    code = dico.fit(X).transform(X[0])
    assert_true(len(code[0]) <= 1)

def test_mdla_nonzero_coefs():
    n_kernels = 8
    dico = MultivariateDictLearning(n_kernels=n_kernels, random_state=0,
                                max_iter=3, n_nonzero_coefs=3, verbose=5)
    code = dico.fit(X).transform(X[0])
    assert_true(len(code[0]) <= 3)

    dico = MiniBatchMultivariateDictLearning(n_kernels=n_kernels,
                    random_state=0, n_iter=3, n_nonzero_coefs=3, verbose=5)
    code = dico.fit(X).transform(X[0])
    assert_true(len(code[0]) <= 3)

def test_X_array():
    n_kernels = 8
    X = rng_global.randn(n_samples, n_features, n_dims)
    dico = MultivariateDictLearning(n_kernels=n_kernels, random_state=0,
                                max_iter=3, n_nonzero_coefs=3, verbose=5)
    code = dico.fit(X).transform(X[0])
    assert_true(len(code[0]) <= 3)
    
    dico = MiniBatchMultivariateDictLearning(n_kernels=n_kernels,
                    random_state=0, n_iter=3, n_nonzero_coefs=3, verbose=5)
    code = dico.fit(X).transform(X[0])
    assert_true(len(code[0]) <= 3)

def test_mdla_shuffle():
    n_kernels = 8
    dico = MiniBatchMultivariateDictLearning(n_kernels=n_kernels,
                    random_state=0, n_iter=3, n_nonzero_coefs=1,
                    verbose=5, shuffle=False)
    code = dico.fit(X).transform(X[0])
    assert_true(len(code[0]) <= 1)

def test_n_kernels():
    dico = MultivariateDictLearning(random_state=0, max_iter=2,
                                    n_nonzero_coefs=1, verbose=5).fit(X)
    assert_true(len(dico.kernels_) == 2*n_features)
    
    dico = MiniBatchMultivariateDictLearning(random_state=0,
                    n_iter=2, n_nonzero_coefs=1, verbose=5).fit(X)
    assert_true(len(dico.kernels_) == 2*n_features)
    
    dico = MiniBatchMultivariateDictLearning(random_state=0,
                    n_iter=2, n_nonzero_coefs=1, verbose=5).partial_fit(X)
    assert_true(len(dico.kernels_) == 2*n_features)
    
def test_mdla_nonzero_coef_errors():
    n_kernels = 8
    dico = MultivariateDictLearning(n_kernels=n_kernels, random_state=0,
                                max_iter=2, n_nonzero_coefs=0)
    assert_raises(ValueError, dico.fit, X)
    
    dico = MiniBatchMultivariateDictLearning(n_kernels=n_kernels, random_state=0,
                                n_iter=2, n_nonzero_coefs=n_kernels+1)
    assert_raises(ValueError, dico.fit, X)

def test_sparse_encode():
    n_kernels = 8
    dico = MultivariateDictLearning(n_kernels=n_kernels, random_state=0,
                                    max_iter=2, n_nonzero_coefs=1)
    dico = dico.fit(X)
    _, code = multivariate_sparse_encode(X, dico, n_nonzero_coefs=1,
                                        n_jobs=-1, verbose=3)
    assert_true(len(code[0]) <= 1)

def test_dict_init():
    n_kernels = 8
    d = [rng_global.randn(n_features, n_dims) for i in range(n_kernels)]
    for i in range(len(d)):
        d[i] /= np.linalg.norm(d[i], 'fro')
    dico = MultivariateDictLearning(n_kernels=n_kernels, random_state=0,
                                    max_iter=1, n_nonzero_coefs=1, learning_rate=0.,
                                    dict_init=d, verbose=5).fit(X)
    dico = dico.fit(X)
    for i in range(n_kernels):
        assert_array_almost_equal(dico.kernels_[i], d[i])
    # code = dico.fit(X).transform(X[0])
    # assert_true(len(code[0]) > 1)
    
    dico = MiniBatchMultivariateDictLearning(n_kernels=n_kernels,
                random_state=0, n_iter=1, n_nonzero_coefs=1,
                dict_init=d, verbose=1, learning_rate=0.).fit(X)
    dico = dico.fit(X)
    for i in range(n_kernels):
        assert_array_almost_equal(dico.kernels_[i], d[i])
    # code = dico.fit(X).transform(X[0])
    # assert_true(len(code[0]) <= 1)

def test_mdla_dict_init():
    n_kernels = 10
    n_samples, n_features, n_dims = 20, 5, 3
    X = [rng_global.randn(n_features, n_dims) for i in range(n_samples)]
    dict_init = [np.random.randn(n_features, n_dims) for i in range(n_kernels)]
    dico = MultivariateDictLearning(n_kernels=n_kernels, random_state=0,
                                    max_iter=10, dict_init=dict_init).fit(X)
    diff = 0.
    for i in range(n_kernels):
        diff = diff + (dico.kernels_[i]-dict_init[i]).sum()
    assert_true(diff !=0)
                
def test_mdla_dict_update():
    n_kernels = 10
    # n_samples, n_features, n_dims = 100, 5, 3
    n_samples, n_features, n_dims = 80, 5, 3
    X = [rng_global.randn(n_features, n_dims) for i in range(n_samples)]
    dico = MultivariateDictLearning(n_kernels=n_kernels, random_state=0,
                                    max_iter=10, n_jobs=-1).fit(X)
    first_epoch = list(dico.kernels_)
    dico = dico.fit(X)
    second_epoch = list(dico.kernels_)
    for k, c in zip(first_epoch, second_epoch):
        assert_true((k-c).sum() != 0.)

    dico = MiniBatchMultivariateDictLearning(n_kernels=n_kernels,
                random_state=0, n_iter=10, n_jobs=-1).fit(X)
    first_epoch = list(dico.kernels_)
    dico = dico.fit(X)
    second_epoch = list(dico.kernels_)
    for k, c in zip(first_epoch, second_epoch):
        assert_true((k-c).sum() != 0.)

    dico = MiniBatchMultivariateDictLearning(n_kernels=n_kernels,
                random_state=0, n_iter=10, n_jobs=-1).partial_fit(X)
    first_epoch = list(dico.kernels_)
    dico = dico.partial_fit(X)
    second_epoch = list(dico.kernels_)
    for k, c in zip(first_epoch, second_epoch):
        assert_true((k-c).sum() != 0.)
            
def test_sparse_multivariate_coder():
    n_kernels = 8
    d = [np.random.randn(n_features, n_dims) for i in range(n_kernels)]
    coder = SparseMultivariateCoder(dictionary=d, n_nonzero_coefs=1, n_jobs=-1)
    coder.fit(X)
    for i in range(n_kernels):    
        assert_array_almost_equal(d[i], coder.kernels_[i])

def TODO_test_shift_invariant_input():
    n_kernels = 8
    dico = list()
    dico.append(np.array([1, 2, 3, 2, 1]))

def _generate_testbed(kernel_init_len, n_nonzero_coefs, n_kernels,
                     n_samples=10, n_features=5, n_dims=3):
    dico = [np.random.randn(kernel_init_len, n_dims) for i in range(n_kernels)]
    for i in range(len(dico)):
        dico[i] /= np.linalg.norm(dico[i], 'fro')
    
    signals = list()
    decomposition = list()
    for i in range(n_samples):
        s = np.zeros(shape=(n_features, n_dims))
        d = np.zeros(shape=(n_nonzero_coefs, 3))
        rk = np.random.permutation(range(n_kernels))
        for j in range(n_nonzero_coefs):
            k_idx = rk[j]
            k_amplitude = 3. * np.random.rand() + 1.
            k_offset = np.random.randint(n_features - kernel_init_len + 1)
            s[k_offset:k_offset+kernel_init_len, :] += (k_amplitude *
                                                        dico[k_idx])
            d[j, :] = np.array([k_amplitude, k_offset, k_idx])
        decomposition.append(d)
        signals.append(s)
    signals = np.array(signals)

    return dico, signals, decomposition
        
def test_mdla_reconstruction():
    n_kernels = 8
    n_nonzero_coefs = 3
    kernel_init_len = n_features 
    dico, signals, decomposition = _generate_testbed(kernel_init_len,
                                                     n_nonzero_coefs,
                                                     n_kernels)
    
    assert_array_almost_equal(reconstruct_from_code(decomposition,
                                                    dico, n_features),
                              signals)

def test_multivariate_OMP():
    n_samples = 10
    n_features = 100
    n_dims = 90
    n_kernels = 8
    n_nonzero_coefs = 3
    kernel_init_len = n_features
    verbose = False

    dico, signals, decomposition = _generate_testbed(kernel_init_len,
                                                     n_nonzero_coefs,
                                                     n_kernels,
                                                     n_samples, n_features,
                                                     n_dims)
    r, d = multivariate_sparse_encode(signals, dico, n_nonzero_coefs,
                                      n_jobs = 1)
    if verbose == True:
        for i in range(n_samples):
            # original signal decomposition, sorted by amplitude
            sorted_decomposition = np.zeros_like(decomposition[i]).view('float, int, int')
            for j in range(decomposition[i].shape[0]):
                sorted_decomposition[j] = tuple(decomposition[i][j,:].tolist())
            sorted_decomposition.sort(order=['f0'], axis=0)
            for j in reversed(sorted_decomposition): print (j)
        
            # decomposition found by OMP, also sorted
            sorted_d = np.zeros_like(d[i]).view('float, int, int')
            for j in range(d[i].shape[0]):
                sorted_d[j] = tuple(d[i][j,:].tolist())
            sorted_d.sort(order=['f0'], axis=0)
            for j in reversed(sorted_d): print (j)
            
    assert_array_almost_equal(reconstruct_from_code(d, dico, n_features),
                             signals, decimal=3)

def _test_with_pydico():
    import pickle, shutil
    n_kernels = 8
    n_nonzero_coefs = 3
    kernel_init_len = n_features
    dico, signals, decomposition = _generate_testbed(kernel_init_len,
                                                    n_nonzero_coefs, n_kernels)
    o = {'signals':signals, 'dico':dico, 'decomposition':decomposition}
    with open('skmdla.pck', 'w') as f:
        pickle.dump(o, f)
    f.close()
    shutil.copy('skmdla.pck', '../RC/skmdla.pck')

    print (signals)
    print (dico)
    
    r, d = multivariate_sparse_encode(signals, dico, n_nonzero_coefs,
                                      n_jobs = 1, verbose=4)

def _test_with_pydico_reload():
    import pickle
    n_kernels = 8
    n_nonzero_coefs = 3
    kernel_init_len = n_features
    with open('skmdla.pck', 'w') as f:
        o = pickle.load(f)
    f.close()
    dico = o['dico']
    signals = o['signals']
    decomposition = o['decomposition']
    
    r, d = multivariate_sparse_encode(signals, dico, n_nonzero_coefs,
                                      n_jobs = 1, verbose=4)
    
def _verif_OMP():
    n_samples = 1000
    n_nonzero_coefs = 3

    for n_features in range (5, 50, 5):
        kernel_init_len = n_features - n_features/2
        n_dims = n_features/2
        n_kernels = n_features*5
        dico, signals, decomposition = _generate_testbed(kernel_init_len,
                                                        n_nonzero_coefs,
                                                        n_kernels,
                                                        n_samples, n_features,
                                                        n_dims)
        r, d = multivariate_sparse_encode(signals, dico, n_nonzero_coefs,
                                          n_jobs = 1)
        reconstructed = reconstruct_from_code(d, dico, n_features)

        residual_energy = 0.
        for sig, rec in zip(signals, reconstructed):
            residual_energy += ((sig-rec)**2).sum(1).mean()

        print ('Mean energy of the', n_samples, 'residuals for', (n_features, n_dims), 'features and', n_kernels, 'kernels of', (kernel_init_len, n_dims),' is', residual_energy/n_samples)
