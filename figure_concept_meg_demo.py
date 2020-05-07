import mne

import numpy as np
import mne
from mne import io, read_proj, read_selection
from mne.datasets import sample
from mne.time_frequency import psd_welch

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
proj_fname = data_path + '/MEG/sample/sample_audvis_eog-proj.fif'

tmin, tmax = 0, 240  # use the first 60s of data

# Setup for reading the raw data (to save memory, crop before loading)
raw = io.read_raw_fif(raw_fname).crop(tmin, tmax).load_data()
raw.info['bads'] += ['MEG 2443', 'EEG 053']  # bads + 2 more

# Add SSP projection vectors to reduce EOG and ECG artifacts
projs = read_proj(proj_fname)
raw.add_proj(projs, remove_existing=True)
raw.apply_proj()
raw.filter(1, 80)

fmin, fmax = 1, 50 # look at frequencies between 2 and 300Hz
n_fft = 8196  # the FFT size (n_fft). Ideally a power of 2

picks = mne.pick_types(raw.info, meg='mag')

psds, freqs = psd_welch(raw, tmin=tmin, tmax=tmax,
                        fmin=fmin, fmax=fmax, proj=True, picks=picks,
                        n_fft=n_fft, n_jobs=1)
psds = 10 * np.log10(psds)

import pandas as pd

dfs = list()
for ii, psd in enumerate(psds):
    data = pd.DataFrame(
        dict(psd=psd,
            freqs=freqs)
    )
    data['channel'] = raw.ch_names[picks[ii]]
    dfs.append(data)
dfs = pd.concat(dfs, axis=0)
dfs.to_csv("./outputs/demo_meg_psd.csv")
