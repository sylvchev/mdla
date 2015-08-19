# MDLA - Multivariate Dictionary Learning Algorithm

<!---
[![Coverage Status](https://coveralls.io/repos/sylvchev/mdla/badge.svg?branch=master&service=github)](https://coveralls.io/github/sylvchev/mdla?branch=master)
[![Travis CI](https://travis-ci.org/sylvchev/mdla.svg?branch=master)](https://travis-ci.org/sylvchev/mdla)
[![Code Climate](https://codeclimate.com/github/sylvchev/mdla/badges/gpa.svg)](https://codeclimate.com/github/sylvchev/mdla)
-->

## Dictionary Learning for the multivariate dataset

This dictionary learning variant is tailored for dealing with
multivariate datasets and especially timeseries, where samples are
matrices and the dataset is seen as a tensor.
Dictionary Learning Algorithm (DLA) decompose input vector on
a dictionary matrix with a sparse coefficient vector, see (a) on
figure below. To handle
multivariate data, a first approach called **multichannel DLA**, see (b) on figure below, is to
decompose the matrix vector on a dictionary matrix but with sparse
coefficient matrices, assuming that a multivariate sample could be
seen as a collection of channels explained by the same dictionary.
Nonetheless, multichannel DLA breaks the "spatial" coherence of
multivariate samples, discarding the column-wise relationship
existing in the samples. **Multivariate DLA**, (c), on figure below, decompose the matrix
input on a tensor dictionary, where each atom is a matrix, with sparse
coefficient vectors. In this case, the spatial relationship are
directly encoded in the dictionary, as each atoms has the same
dimension than an input samples.

![dictionaries](https://github.com/sylvchev/mdla/raw/master/img/multidico.png)

(figure from [Chevallier et al., 2014][CHE14] )

To handle timeseries, two major modifications are brought to DLA:
1. extension to **multivariate** samples
2. **shift-invariant** approach, 
The first point is explained above.  To implement the second one,
there is two possibility, either slicing the input timeseries into
small overlapping samples or to have atoms smaller than input samples,
leading to a decomposition with sparse coefficients and offsets. In
the latter case, the decomposition could be seen as sequence of
kernels occuring at different time steps.

![shift invariance](https://github.com/sylvchev/mdla/raw/master/img/audio4spikegram.png)

(figure from [Smith & Lewicki, 2005][LEW05])

The proposed implementation is an adaptation of the work of the
following authors:
- Q. Barthélemy, A. Larue, A. Mayoue, D. Mercier, and
  J.I. Mars. *Shift & 2D rotation invariant sparse coding for multi-
  variate signal*. IEEE Trans. Signal Processing, 60:1597–1611, 2012.
- Q. Barthélemy, A. Larue, and J.I. Mars. *Decomposition and
  dictionary learning for 3D trajectories*. Signal Process.,
  98:423–437, 2014.
- Q. Barthélemy, C. Gouy-Pailler, Y. Isaac, A. Souloumiac, A. Larue,
  and J.I. Mars. *Multivariate temporal dictionary learning for
  EEG*. Journal of Neuroscience Methods, 215:19–28, 2013.

## Dependencies

The only dependencies are scikit-learn, matplotlib, numpy and scipy.

No installation is required.

## Example

A straightforward example is:

```python
from mdla import MultivariateDictLearning
from mdla import multivariate_sparse_encode
from np.linalg import norm

rng_global = np.random.RandomState(0)
n_samples, n_features, n_dims = 10, 5, 3
X = rng_global.randn(n_samples, n_features, n_dims)

n_kernels = 8
dico = MultivariateDictLearning(n_kernels=n_kernels, max_iter=10).fit(X)
residual, code = multivariate_sparse_encode(X, dico)
print ('Objective error for each samples is:')
for i in range(len(r)):
    print ('Sample', i, ':', norm(r[i], 'fro') + len(code[i]))
```

## Bibliography

[CHE14]: http://dx.doi.org/10.1109/ICASSP.2014.6854993 "Chevallier, S., Barthelemy, Q., & Atif, J. (2014, May). *Subspace metrics for multivariate dictionaries and application to EEG*. In Acoustics, Speech and Signal Processing (ICASSP), 2014 IEEE International Conference on (pp. 7178-7182). IEEE."

[LEW05]: http://dl.acm.org/citation.cfm?id=1119614 "Smith, E., & Lewicki, M. S. (2005). *Efficient coding of time-relative structure using spikes*. Neural Computation, 17(1), 19-45."
