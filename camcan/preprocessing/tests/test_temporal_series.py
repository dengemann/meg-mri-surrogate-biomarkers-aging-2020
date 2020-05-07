from os.path import join, split

from numpy import mean, std

from nilearn.datasets import fetch_atlas_msdl

from sklearn.utils.testing import assert_equal, assert_almost_equal

from camcan.preprocessing import extract_timeseries


def test_extract_timeseries():
    current_dir = split(__file__)[0]
    filename_func = join(current_dir, 'data', 'func.nii.gz')
    filename_confounds = join(current_dir, 'data', 'confounds.txt')

    time_serie = extract_timeseries(filename_func,
                                    atlas=fetch_atlas_msdl().maps)
    assert_equal(time_serie.shape, (5, 39))
    assert_almost_equal(mean(time_serie), -1.05343771823e-13)
    assert_almost_equal(std(time_serie), 5.253714931447222)

    time_serie = extract_timeseries(filename_func,
                                    atlas=fetch_atlas_msdl().maps,
                                    confounds=filename_confounds)
    assert_equal(time_serie.shape, (5, 39))
    assert_almost_equal(mean(time_serie), -1.05567147082e-13)
    assert_almost_equal(std(time_serie), 1.8468688363637491e-13)
