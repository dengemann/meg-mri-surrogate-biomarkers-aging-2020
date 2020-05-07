from nilearn.connectome import sym_matrix_to_vec
from nilearn.connectome.connectivity_matrices import (_geometric_mean,
                                                      _map_eigenvalues)
import numpy as np


def map_tangent(data, diag=False):
    """Transform to tangent space.

    Parameters
    ----------
    data: list of numpy.ndarray of shape(n_features, n_features)
        List of semi-positive definite matrices.
    diag: bool
        Whether to discard the diagonal elements before vectorizing. Default is
        False.

    Returns
    -------
    tangent: numpy.ndarray, shape(n_features * (n_features - 1) / 2)
    """
    mean_ = _geometric_mean(data, max_iter=30, tol=1e-7)
    whitening_ = _map_eigenvalues(lambda x: 1. / np.sqrt(x),
                                  mean_)
    tangent = [_map_eigenvalues(np.log, whitening_.dot(c).dot(whitening_))
               for c in data]
    tangent = np.array(tangent)

    return sym_matrix_to_vec(tangent, discard_diagonal=diag)
