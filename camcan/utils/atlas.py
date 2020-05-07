"""Utilities to manage mask."""

from nilearn._utils import check_niimg
from nilearn.input_data import NiftiLabelsMasker, NiftiMapsMasker


def make_masker_from_atlas(atlas, memory=None, memory_level=1):
    """Construct a maker from a given atlas.

    Parameters
    ----------
    atlas : str or 3D/4D Niimg-like object,
        The atlas to use to create the masker. If string, it should corresponds
        to the path of a Nifti image.

    memory : instance of joblib.Memory or string, (default=None)
        Used to cache the masking process. By default, no caching is done. If a
        string is given, it is the path to the caching directory.

    memory_level : integer, optional (default=1)
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    Returns
    -------
    masker : Nilearn Masker
        Nilearn Masker.

    """
    atlas_ = check_niimg(atlas)
    atlas_dim = len(atlas_.shape)
    if atlas_dim == 3:
        masker = NiftiLabelsMasker(atlas_,
                                   memory=memory,
                                   memory_level=memory_level,
                                   smoothing_fwhm=6,
                                   detrend=True,
                                   verbose=1)
    elif atlas_dim == 4:
        masker = NiftiMapsMasker(atlas_,
                                 memory=memory,
                                 memory_level=memory_level,
                                 smoothing_fwhm=6,
                                 detrend=True,
                                 verbose=1)

    return masker
