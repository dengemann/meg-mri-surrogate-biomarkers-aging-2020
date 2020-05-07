"""Extract temporal series from Nifti images."""

import numpy as np

import nibabel as nib
from nilearn.datasets import fetch_atlas_basc_multiscale_2015

from ..utils import make_masker_from_atlas


def extract_timeseries(func,
                       atlas=fetch_atlas_basc_multiscale_2015().scale064,
                       confounds=None,
                       memory=None,
                       memory_level=1,
                       duration=None):
    """Extract time series for a list of functional volume.

    Parameters
    ----------
    func : str,
        Path of Nifti volumes.

    atlas : str or 3D/4D Niimg-like object, (default=BASC64)
        The atlas to use to create the masker. If string, it should corresponds
        to the path of a Nifti image.

    confounds : str,
        Path containing the confounds.

    memory : instance of joblib.Memory or string, (default=None)
        Used to cache the masking process. By default, no caching is done. If a
        string is given, it is the path to the caching directory.

    memory_level : integer, optional (default=1)
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    duration : float
        The duration of timeseries to be extracted, measured in seconds.

    """
    TR = 1.992  # s
    masker = make_masker_from_atlas(atlas, memory=memory,
                                    memory_level=memory_level)
    masker.fit()

    if confounds is not None:
        confounds_ = np.loadtxt(confounds)
    else:
        confounds_ = None

    func_img = nib.load(func)

    if duration is not None:
        # how many slices to keep
        n_slices = int(duration / TR)
        func_img = func_img.slicer[:, :, :, :n_slices]

        if confounds_ is not None:
            confounds_ = confounds_[:n_slices, :]

    return masker.transform(func_img, confounds=confounds_)
