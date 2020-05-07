# Author: Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import mne

# power spectra

proc_log = pd.read_csv('data/log_compute_rest_spectra.csv')
subject = proc_log.query('error == "None"')['subject'].values

X = np.load('data/features_rest_allch_psd_all.npy')
X = X[proc_log.query('error == "None"')['subject'].isin(subject)]
freqs = np.linspace(.1, 120, 491)
info = mne.io.read_info('data/neuromag-info.fif')

df_info = pd.read_csv('data/participant_data.tsv', sep='\t')
df_info = df_info[df_info.Observations.isin(subject)]

X = np.log10(X)

gamma_low = (freqs >= 35.) & (freqs < 50.)
gamma_mid = (freqs >= 50.) & (freqs < 76.)
gamma_high = (freqs >= 76.) & (freqs <= 120.)

low_freq = (freqs >= 0.1) & (freqs <= 1.5)
y_low = X[..., low_freq]
x_low = np.log10(freqs[low_freq])[:, np.newaxis]

gamma_low_low = (freqs >= 35) & (freqs <= 49.8)
y_gamma_low = X[..., gamma_low_low]  # dB = 20 log10 Hz
x_gamma_low = np.log10(freqs[gamma_low_low])[:, np.newaxis]

low_freq_slope = []
gamma_low_slope = []


def get_slope(x, y):
    lm = LinearRegression()
    lm.fit(x, y)
    return lm.coef_

picks_mag = mne.pick_types(info, meg="mag")

X_1f_low = np.array([get_slope(x_low, yy[picks_mag].T)[:, 0]
                     for yy in y_low])
X_1f_gamma = np.array([get_slope(x_gamma_low, yy[picks_mag].T)[:, 0]
                       for yy in y_gamma_low])

log_freq = np.log10(freqs[:, None])

poly_freqs = PolynomialFeatures(degree=15).fit_transform(
    log_freq)

lm = LinearRegression()

chs = sum([mne.selection.read_selection(sel, info=info) for sel in
           ('Left-parietal', 'Right-parietal')], [])

mag = [info['ch_names'][ii] for ii in picks_mag]
picks = [info['ch_names'].index(ch) for ch in chs if ch not in mag]

X_fit = X[:, picks].mean(1).T
lm.fit(poly_freqs, X_fit)

plt.figure()
plt.plot(log_freq, X_fit, color='black', alpha=0.2)
plt.plot(log_freq, lm.predict(poly_freqs), color='red', alpha=0.2)
plt.xticks(np.log10([0.1, 1, 10, 100]), [0.1, 1, 10, 100])
plt.show()

resid = X_fit - lm.predict(poly_freqs)

plt.figure()
plt.plot(log_freq,
         resid, color='black', alpha=0.2)
plt.xticks(np.log10([0.1, 1, 10, 100]), [0.1, 1, 10, 100])
plt.show()

filt = ((freqs >= 6) &
        (freqs < 15))

idx = resid[filt].argmax(0)

peaks = freqs[filt][idx]
plt.figure()
plt.hist(peaks, bins=100)

columns = [f'1f_low{ii + 1}' for ii in range(X_1f_low.shape[1])]
columns += [f'1f_gamma{ii + 1}' for ii in range(X_1f_low.shape[1])]

df = pd.DataFrame(
    np.concatenate([X_1f_low, X_1f_gamma], axis=1),
    columns=columns)

df['subject'] = df_info['Observations'].values
df['alpha_peak'] = peaks

df.to_hdf('./data/meg_extra_data.h5', key="MEG_rest_extra")