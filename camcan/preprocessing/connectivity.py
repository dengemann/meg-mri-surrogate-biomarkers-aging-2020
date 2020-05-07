"""Extract connectivity from time series."""

from nilearn.connectome import ConnectivityMeasure, sym_to_vec


def extract_connectivity(subjects_time_series, kind='tangent'):
    """Extract functional connectivity from time series.

    Parameters
    ----------
    subjects_time_series: list of ndarray of shape (n_samples, n_features)
        List of the time series for all the subjects.

    kind : str, (default='tangent')
        The type of matrix to be computed.

    Returns
    -------
    connectivity : ndarray, shape (n_samples, n_features * (n_features+1) / 2)
        The flattened lower triangular part of the connectivity matrix.

    """
    fc = ConnectivityMeasure(kind=kind)
    X = fc.fit_transform(subjects_time_series)
    diag = True if kind == 'correlation' else False

    return sym_to_vec(X, discard_diagonal=diag)
