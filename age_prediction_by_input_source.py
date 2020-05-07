"""Age prediction using MRI, fMRI and MEG data."""
# Author: Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from joblib import Memory, Parallel, delayed

from camcan.utils import run_ridge
from threadpoolctl import threadpool_limits
from camcan.processing import map_tangent

##############################################################################
# Paths

DRAGO_PATH = '/storage/inria/agramfor/camcan_derivatives'
OLEH_PATH = '/storage/tompouce/okozynet/projects/camcan_analysis/data'
PANDAS_OUT_FILE = './data/age_prediction_exp_data_denis_{}-rep.h5'
STRUCTURAL_DATA = f'{OLEH_PATH}/structural/structural_data.h5'
CONNECT_DATA_CORR = f'{OLEH_PATH}/connectivity/connect_data_correlation.h5'
CONNECT_DATA_TAN = f'{OLEH_PATH}/connectivity/connect_data_tangent.h5'
MEG_EXTRA_DATA = './data/meg_extra_data.h5'
MEG_PEAKS = './data/evoked_peaks.csv'
MEG_PEAKS2 = './data/evoked_peaks_task_audvis.csv'

##############################################################################
# Control paramaters

# common subjects 574
N_REPEATS = 10
N_JOBS = 10
N_THREADS = 6
REDUCE_TO_COMMON_SUBJECTS = False

memory = Memory(location=DRAGO_PATH)

##############################################################################
# MEG features
#
# 1. Marginal Power
# 2. Cross-Power
# 3. Envelope Power
# 4. Envelope Cross-Power
# 5. Envelope Connectivity
# 6. Envelope Orthogonalized Connectivity
# 7. 1/f
# 8. Alpha peak
# 9. ERF delay

FREQ_BANDS = ('alpha',
              'beta_high',
              'beta_low',
              'delta',
              'gamma_high',
              'gamma_lo',
              'gamma_mid',
              'low',
              'theta')

meg_source_types = (
    'mne_power_diag',
    'mne_power_cross',
    'mne_envelope_diag',
    'mne_envelope_cross',
    'mne_envelope_corr',
    'mne_envelope_corr_orth'
)


def vec_to_sym(data, n_rows, skip_diag=True):
    """Put vector back in matrix form"""
    if skip_diag:
        k = 1
        # This is usually true as we write explicitly
        # the diag info in asecond step and we only
        # store the upper triangle, hence all files
        # have equal size.
    else:
        k = 0
    C = np.zeros((n_rows, n_rows), dtype=np.float64)
    C[np.triu_indices(n=n_rows, k=k)] = data
    C += C.T
    if not skip_diag:
        C.flat[::n_rows + 1] = np.diag(C) / 2.
    return C


def make_covs(diag, data, n_labels):
    if not np.isscalar(diag):
        assert np.all(diag.index == data.index)
    covs = np.empty(shape=(len(data), n_labels, n_labels))
    for ii, this_cross in enumerate(data.values):
        C = vec_to_sym(this_cross, n_labels)
        if np.isscalar(diag):
            this_diag = diag
        else:
            this_diag = diag.values[ii]
        C.flat[::n_labels + 1] = this_diag
        covs[ii] = C
    return covs


@memory.cache
def read_meg_rest_data(kind, band, n_labels=448):
    """Read the resting state data (600 subjects)

    Read connectivity outptus and do some additional
    preprocessing.

    Parameters
    ----------
    kind : str
        The type of MEG feature.
    band : str
        The frequency band.
    n_label: int
        The number of ROIs in source space.
    """
    if kind == 'mne_power_diag':
        data = pd.read_hdf(
            op.join(DRAGO_PATH, f'mne_source_power_diag-{band}.h5'),
            key=kind)
    elif kind == 'mne_power_cross':
        # We need the diagonal powers to do tangent mapping.
        # but then we will discard it.
        diag = read_meg_rest_data(kind='mne_power_diag', band=band)
        # undp log10
        diag = diag.transform(lambda x: 10 ** x)
        index = diag.index.copy()

        data = pd.read_hdf(
            op.join(DRAGO_PATH, f'mne_source_power_cross-{band}.h5'),
            key=kind)
        covs = make_covs(diag, data, n_labels)
        data = map_tangent(covs, diag=True)
        data = pd.DataFrame(data=data, index=index)
    if kind == 'mne_envelope_diag':
        data = pd.read_hdf(
            op.join(DRAGO_PATH, f'mne_envelopes_diag_{band}.h5'),
            key=kind)
    elif kind == 'mne_envelope_cross':
        # We need the diagonal powers to do tangent mapping.
        # but then we will discard it.
        diag = read_meg_rest_data(kind='mne_envelope_diag', band=band)
        # undp log10
        diag = diag.transform(lambda x: 10 ** x)
        index = diag.index.copy()

        data = pd.read_hdf(
            op.join(DRAGO_PATH, f'mne_envelopes_cross_{band}.h5'),
            key=kind)
        covs = make_covs(diag, data, n_labels)
        data = map_tangent(covs, diag=True)
        data = pd.DataFrame(data=data, index=index)
    elif kind == 'mne_envelope_corr':
        # The diagonal is simply one.
        diag = 1.0
        data = pd.read_hdf(
            op.join(DRAGO_PATH, f'mne_envelopes_corr_{band}.h5'),
            key=kind)
        index = data.index.copy()

        data = map_tangent(make_covs(diag, data, n_labels),
                           diag=True)
        data = pd.DataFrame(data=data, index=index)

    elif kind == 'mne_envelope_corr_orth':
        data = pd.read_hdf(
            op.join(DRAGO_PATH, f'mne_envelopes_corr_orth_{band}.h5'), key=kind)
        # The result here is not an SPD matrix.
        # We do do Fisher's Z-transform instead.
        # https://en.wikipedia.org/wiki/Fisher_transformation
        data = data.transform(np.arctanh)
    return data


meg_power_alpha = read_meg_rest_data(
    kind='mne_power_diag', band='alpha')

meg_power_subjects = set(meg_power_alpha.index)
# source level subjects all the same for resting state

##############################################################################
# MRI features

area_data = pd.read_hdf(STRUCTURAL_DATA, key='area')
thickness_data = pd.read_hdf(STRUCTURAL_DATA, key='thickness')
volume_data = pd.read_hdf(STRUCTURAL_DATA, key='volume')

# read connectivity data
connect_data_tangent_modl = pd.read_hdf(CONNECT_DATA_TAN, key='modl256')
fmri_subjects = set(connect_data_tangent_modl.index)

##############################################################################
# Bundle all data

# Add extra dfeatures
meg_extra = pd.read_hdf(MEG_EXTRA_DATA, key='MEG_rest_extra')
meg_peaks = pd.read_csv(MEG_PEAKS).set_index('subject')[['aud', 'vis']]
meg_peaks2 = pd.read_csv(MEG_PEAKS2).set_index('subject')

meg_common_subjects = (meg_power_subjects.intersection(meg_extra.index)
                                         .intersection(meg_peaks.index)
                                         .intersection(meg_peaks2.index))

meg_union_subjects = (meg_power_subjects.union(meg_extra.index)
                                        .union(meg_peaks.index)
                                        .union(meg_peaks2.index))

print(f"Got {len(meg_union_subjects)} (union) and "
      f"{len(meg_common_subjects)} (intersection) MEG subject")

common_subjects = list(meg_common_subjects.intersection(area_data.index)
                                          .intersection(thickness_data.index)
                                          .intersection(volume_data.index)
                                          .intersection(fmri_subjects))
common_subjects.sort()

union_subjects = list(meg_union_subjects.union(area_data.index)
                                        .union(thickness_data.index)
                                        .union(volume_data.index)
                                        .union(fmri_subjects))
union_subjects.sort()

print(f"Got {len(union_subjects)} (union) and "
      f"{len(common_subjects)} (intersection) subjects")

if REDUCE_TO_COMMON_SUBJECTS:
    union_subjects = common_subjects[:]

print(f"Using {len(union_subjects)} subjects")
# read information about subjects
subjects_data = pd.read_csv('./data/participant_data.csv', index_col=0)
# for storing predictors data

subjects_template = pd.DataFrame(index=union_subjects,
                                 dtype=float)
subjects_predictions = subjects_data.loc[subjects_template.index, ['age']]

print('Data was read successfully.')

data_ref = {
    'MEG 1/f low': meg_extra[
        [cc for cc in meg_extra.columns if '1f_low' in cc]],
    'MEG 1/f gamma': meg_extra[
        [cc for cc in meg_extra.columns if '1f_gamma' in cc]],
    'MEG alpha_peak': meg_extra[['alpha_peak']],
    'MEG aud': meg_peaks[['aud']],
    'MEG vis': meg_peaks[['vis']],
    'MEG audvis': meg_peaks2[['audvis']],
    'Cortical Surface Area': area_data,
    'Cortical Thickness': thickness_data,
    'Subcortical Volumes': volume_data,
    'Connectivity Matrix, MODL 256 tan': connect_data_tangent_modl,
}
for band in FREQ_BANDS:
    for kind in meg_source_types:
        data_ref[f"MEG {kind} {band}"] = dict(kind=kind, band=band)

for kind in ('mne_power_diag', 'mne_envelope_diag'):
    this_data = list()
    for band in FREQ_BANDS:
        band_data = read_meg_rest_data(kind=kind, band=band)
        band_data.columns = [cc + f'_{band}' for cc in band_data.columns]
        this_data.append(band_data)
    this_data = pd.concat(this_data, axis=1)
    key = f'MEG {"power" if "power" in kind else "envelope"} diag'
    data_ref[key] = this_data

##############################################################################
# Main analysis


def run_10_folds(data_ref, repeat, n_splits=10):
    # make more or less arbitrary superstitious seed from repeat
    cv = KFold(n_splits=n_splits, random_state=repeat * 7,
               shuffle=True)
    # store mae, learning curves for summary plots
    subjects_predictions = subjects_data.loc[subjects_template.index, ['age']]
    regression_mae = pd.DataFrame(columns=list(data_ref), dtype=float)
    regression_r2 = pd.DataFrame(columns=list(data_ref), dtype=float)
    learning_curves = {}
    with threadpool_limits(limits=N_THREADS, user_api='blas'):
        for key, data in data_ref.items():
            if isinstance(data, dict):
                data = read_meg_rest_data(**data)

            data = subjects_template.join(data)
            (df_pred, arr_mae, arr_r2, train_sizes, train_scores,
             test_scores) = run_ridge(data, subjects_data, cv=cv, n_jobs=1)

            arr_mae = -arr_mae
            mae = arr_mae.mean()
            std = arr_mae.std()
            print('%s MAE: %.2f, STD %.2f' % (key, mae, std))

            regression_mae[key] = arr_mae
            regression_r2[key] = arr_r2
            subjects_predictions.loc[df_pred.index, key] = df_pred['y_pred']
            subjects_predictions.loc[
                df_pred.index, 'fold_idx'] = df_pred['fold']

            learning_curves[key] = {
                'train_sizes': train_sizes,
                'train_scores': train_scores,
                'test_scores': test_scores
            }
    regression_r2['repeat'] = repeat
    regression_mae['repeat'] = repeat
    subjects_predictions['repeat'] = repeat
    return (regression_mae, regression_r2, subjects_predictions,
            learning_curves)


out = Parallel(n_jobs=40)(delayed(run_10_folds)(data_ref, repeat)
                          for repeat in range(N_REPEATS))
out = zip(*out)
regression_mae = pd.concat(next(out), axis=0)
regression_r2 = pd.concat(next(out), axis=0)
subjects_predictions = pd.concat(next(out), axis=0)
learning_curves = next(out)

# # save results
PANDAS_OUT_FILE = PANDAS_OUT_FILE.format(N_REPEATS)

with open(f'./data/learning_curves_denis_{N_REPEATS}.pkl', 'wb') as handle:
    pickle.dump(learning_curves, handle, protocol=pickle.HIGHEST_PROTOCOL)

if not REDUCE_TO_COMMON_SUBJECTS:
    PANDAS_OUT_FILE = PANDAS_OUT_FILE.replace('exp_data', 'exp_data_na')

subjects_predictions.to_hdf(PANDAS_OUT_FILE, key='predictions', complevel=9)
regression_mae.to_hdf(PANDAS_OUT_FILE, key='regression', complevel=9)
regression_r2.to_hdf(PANDAS_OUT_FILE, key='r2', complevel=9)
