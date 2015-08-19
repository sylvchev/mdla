# MDLA - Multivariate Dictionary Learning Algorithm

[![Coverage Status](https://coveralls.io/repos/sylvchev/mdla/badge.svg?branch=master&service=github)](https://coveralls.io/github/sylvchev/mdla?branch=master)
[![Travis CI](https://travis-ci.org/sylvchev/mdla.svg?branch=master)](https://travis-ci.org/sylvchev/mdla)
[![Code Climate](https://codeclimate.com/github/sylvchev/mdla/badges/gpa.svg)](https://codeclimate.com/github/sylvchev/mdla)

## Dictionary Learning for the multivariate dataset

This dictionary learning variant is tailored for dealing with
multivariate datasets and especially timeseries, where samples are
matrices and the dataset is seen as a tensor.
Dictionary Learning Algorithm (DLA) decompose input vector on
a dictionary matrix with a sparse coefficient vector. To handle
multivariate data, a first approach called **multichannel DLA** is to
decompose the matrix vector on a dictionary matrix but with sparse
coefficient matrices, assuming that a multivariate sample could be
seen as a collection of channels explained by the same dictionary.
Nonetheless, multichannel DLA breaks the "spatial" coherence of
multivariate samples, discarding the column-wise relationship
existing in the samples. **Multivariate DLA** decompose the matrix
input on a tensor dictionary, where each atom is a matrix, with sparse
coefficient vectors. In this case, the spatial relationship are
directly encoded in the dictionary, as each atoms has the same
dimension than an input samples.

figure DLA vs multi-channel vs mutivariate

To handle timeseries, two major modifications are brought to DLA:
1. extension to **multivariate** samples
2. **shift-invariant** approach, 
The first point is explained above.  To implement the second one,
there is two possibility, either slicing the input timeseries into
small overlapping samples or to have atoms smaller than input samples,
leading to a decomposition with sparse coefficients and offsets. In
the latter case, the decomposition could be seen as sequence of
kernels occuring at different time steps.

figure Lewicki

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



## Useful links

-   Code from Mantas Lukoševičius: http://organic.elis.ugent.be/software/minimal
-   Code from Mantas Lukoševičius: http://minds.jacobs-university.de/mantas/code
-   More serious reservoir computing softwares: http://organic.elis.ugent.be/software
-   Scikit-learn, indeed: http://scikit-learn.org/

## Dependencies

The only dependencies are scikit-learn, numpy and scipy.

No installation is required.

## Example

Using the SimpleESN class is easy as:

```python
from simple_esn import SimpleESN
import numpy as np
n_samples, n_features = 10, 5
np.random.seed(0)
X = np.random.randn(n_samples, n_features)
esn = SimpleESN(n_readout = 2)
echoes = esn.fit_transform(X)
```

It could also be part of a Pipeline:

```python
from simple_esn import SimpleESN
# Pick your classifier
pipeline = Pipeline([('esn', SimpleESN(n_readout=1000)),
                     ('svr', svm.SVR())])
parameters = {
    'esn__weight_scaling': [0.5, 1.0],
    'svr__C': [1, 10]
}
grid_search = GridSearchCV(pipeline, parameters)
grid_search.fit(X_train, y_train)
```
