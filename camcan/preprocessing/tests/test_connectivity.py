from os.path import join, split

import joblib

from numpy import mean, std

from sklearn.utils.testing import assert_equal, assert_almost_equal

from camcan.preprocessing import extract_connectivity


def test_extract_connectivity():
    current_dir = split(__file__)[0]
    filename_timeseries = join(current_dir, 'data', 'timeseries.pkl')

    X = joblib.load(filename_timeseries)
    connectivity = extract_connectivity([X])
    assert_equal(connectivity.shape, (1, 780))
    assert_almost_equal(mean(connectivity), -2.63516237674e-16)
    assert_almost_equal(std(connectivity), 5.23176746128e-15)
