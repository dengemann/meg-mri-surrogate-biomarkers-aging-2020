from nilearn.datasets import (fetch_atlas_basc_multiscale_2015,
                              fetch_atlas_msdl)
from nilearn.input_data import NiftiLabelsMasker, NiftiMapsMasker

from sklearn.utils.testing import assert_true, assert_equal

from camcan.utils import make_masker_from_atlas


def test_make_masker_from_atlas():
    atlas = fetch_atlas_basc_multiscale_2015().scale007
    masker = make_masker_from_atlas(atlas)
    assert_true(isinstance(masker, NiftiLabelsMasker))
    assert_equal(masker.labels_img.shape, (53, 64, 52))

    atlas = fetch_atlas_msdl().maps
    masker = make_masker_from_atlas(atlas)
    assert_true(isinstance(masker, NiftiMapsMasker))
    assert_equal(masker.maps_img.shape, (40, 48, 35, 39))
