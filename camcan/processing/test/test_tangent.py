from pathlib import Path

import mne
import numpy as np
from sklearn.covariance import LedoitWolf

from camcan.preprocessing import extract_connectivity
from camcan.processing import map_tangent


sample_data = Path(mne.datasets.sample.data_path())
fname = sample_data / Path('MEG/sample/sample_audvis_raw.fif')

raw = mne.io.read_raw_fif(str(fname), preload=True)

tmin = 0
tmax = 2
baseline = None

events = mne.find_events(raw)[:10]
raw.pick_types(meg='mag', eeg=False)
epochs = mne.Epochs(raw=raw, tmin=tmin, tmax=tmax, events=events, decim=5)
timeseries = epochs.get_data()

connectivity_tangent = extract_connectivity(timeseries, kind='tangent')

cov_estimator = LedoitWolf(store_precision=False)
connectivities = [cov_estimator.fit(x).covariance_ for x in timeseries]

connectivity_tangent2 = map_tangent(connectivities, diag=False)

np.testing.assert_array_equal(connectivity_tangent, connectivity_tangent2)
