# MDLA - Multivariate Dictionary Learning Algorithm

[![Coverage Status](https://coveralls.io/repos/sylvchev/mdla/badge.svg?branch=master&service=github)](https://coveralls.io/github/sylvchev/mdla?branch=master)
[![Travis CI](https://travis-ci.org/sylvchev/mdla.svg?branch=master)](https://travis-ci.org/sylvchev/mdla)
[![Code Climate](https://codeclimate.com/github/sylvchev/mdla/badges/gpa.svg)](https://codeclimate.com/github/sylvchev/mdla)

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

(figure from [Chevallier et al., 2014](#biblio) )

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

(figure from [Smith & Lewicki, 2005](#biblio))

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
import numpy as np
from mdla import MultivariateDictLearning
from mdla import multivariate_sparse_encode
from numpy.linalg import norm

rng_global = np.random.RandomState(0)
n_samples, n_features, n_dims = 10, 5, 3
X = rng_global.randn(n_samples, n_features, n_dims)

n_kernels = 8
dico = MultivariateDictLearning(n_kernels=n_kernels, max_iter=10).fit(X)
residual, code = multivariate_sparse_encode(X, dico)
print ('Objective error for each samples is:')
for i in range(len(residual)):
    print ('Sample', i, ':', norm(residual[i], 'fro') + len(code[i]))
```

## <a id="biblio"></a>Bibliography

- Chevallier, S., Barthelemy, Q., & Atif, J. (2014). [*Subspace metrics for multivariate dictionaries and application to EEG*][1]. In Acoustics, Speech and Signal Processing (ICASSP), IEEE International Conference on (pp. 7178-7182).
- Smith, E., & Lewicki, M. S. (2005). [*Efficient coding of time-relative structure using spikes*][2]. Neural Computation, 17(1), 19-45
- Chevallier, S., Barthélemy, Q., & Atif, J. (2014). [*On the need for metrics in dictionary learning assessment*][3]. In European Signal Processing Conference (EUSIPCO), pp. 1427-1431. 

[1]: http://dx.doi.org/10.1109/ICASSP.2014.6854993 "Chevallier et al., 2014"

[2]: http://dl.acm.org/citation.cfm?id=1119614 "Smith and Lewicki, 2005"

[3]: https://hal-uvsq.archives-ouvertes.fr/hal-01352054/document

