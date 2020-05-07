# Author: Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np
import pandas as pd
from mne.externals import h5io

DRAGO_PATH = '/storage/inria/agramfor/camcan_derivatives'

FREQ_BANDS = ('alpha',
              'beta_high',
              'beta_low',
              'delta',
              'gamma_high',
              'gamma_lo',
              'gamma_mid',
              'low',
              'theta')

n_labels = 448

meg_source_power = h5io.read_hdf5(
    op.join(DRAGO_PATH, 'all_mne_source_power.h5'))

subjects = list(meg_source_power)

c_index = np.eye(n_labels, dtype=np.bool)
c_index = np.invert(c_index[np.triu_indices(n_labels)])
columns_labels = [f'{ii}' for ii in range(n_labels)]
columns_tiu = [f'{ii}' for ii in range(c_index.sum())]


def make_mat(data, n_rows=448, skip_diag=True):
    if skip_diag:
        k = 1
    else:
        k = 0
    C = np.zeros((n_rows, n_rows), dtype=np.float64)
    C[np.triu_indices(n=n_rows, k=k)] = data
    C += C.T
    C.flat[::n_rows + 1] = np.diag(C) / 2.
    return C


for band in FREQ_BANDS:

    power = pd.DataFrame(
        np.log10([meg_source_power[sub][band]['power'][:, 0]
                  for sub in subjects]).reshape(len(subjects), -1),
        columns=columns_labels, index=subjects)
    power.to_hdf(
        op.join(DRAGO_PATH, f'mne_source_power_diag-{band}.h5'),
        'mne_power_diag', mode='w')

    cross_power = pd.DataFrame(
        np.array([meg_source_power[sub][band]['cov'][c_index]
                 for sub in subjects]).reshape(len(subjects), -1),
        columns=columns_tiu, index=subjects)

    cross_power.to_hdf(
        op.join(DRAGO_PATH, f'mne_source_power_cross-{band}.h5'),
        'mne_power_cross', mode='w')

    meg_envelope = h5io.read_hdf5(
        op.join(DRAGO_PATH, f'all_power_envelopes-{band}'))

    envelope_power = pd.DataFrame(
        np.log10([meg_envelope[sub]['cov'].flat[::n_labels + 1]
                  for sub in subjects]).reshape(len(subjects), -1),
        columns=columns_labels, index=subjects)
    envelope_power.to_hdf(
        op.join(DRAGO_PATH, f'mne_envelopes_diag_{band}.h5'),
        'mne_envelope_diag', mode='w')

    # we forgot to take the upper triangle before.
    envelope_cross_power = pd.DataFrame(
        np.array([meg_envelope[sub]['cov'][np.triu_indices(n_labels, 1)]
                  for sub in subjects]).reshape(len(subjects), -1),
        columns=columns_tiu, index=subjects)
    envelope_cross_power.to_hdf(
        op.join(DRAGO_PATH, f'mne_envelopes_cross_{band}.h5'),
        'mne_envelope_cross', mode='w')

    envelope_corr = pd.DataFrame(
        np.array([meg_envelope[sub]['corr'][c_index]
                  for sub in subjects]).reshape(len(subjects), -1),
        columns=columns_tiu, index=subjects)
    envelope_corr.to_hdf(
        op.join(DRAGO_PATH, f'mne_envelopes_corr_{band}.h5'),
        'mne_envelope_corr', mode='w')

    envelope_corr_orth = pd.DataFrame(
        np.array([meg_envelope[sub]['corr_orth'][c_index]
                  for sub in subjects]).reshape(len(subjects), -1),
        columns=columns_tiu, index=subjects)
    envelope_corr_orth.to_hdf(
        op.join(DRAGO_PATH, f'mne_envelopes_corr_orth_{band}.h5'),
        'mne_envelope_corr_orth', mode='w')

# test:
if False:
    df1 = pd.read_hdf(
        op.join(DRAGO_PATH, f'mne_envelopes_corr_low.h5'),
        'mne_envelope_corr')

    df2 = pd.read_hdf(
        op.join(DRAGO_PATH, f'mne_envelopes_corr_alpha.h5'),
        'mne_envelope_corr')
