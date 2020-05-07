"""Plot mean absolute error (MAE) figures.

Two types of plots are done:
    - MAE versus the chronological age,
    - MAE of one modality versus MAE of another modality.
"""

# Author: Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import (GridSearchCV, LeaveOneGroupOut)
from sklearn.metrics import mean_absolute_error
from joblib import Parallel, delayed
from camcan.processing import permutation_importance

N_JOBS = 20
N_THREADS = 1
DROPNA = 'global'
N_REPEATS = 10
N_PERMUATIONS = 100
#
IN_PREDICTIONS = f'./data/age_prediction_exp_data_na_denis_{N_REPEATS}-rep.h5'
#
MEG_EXTRA_DATA = './data/meg_extra_data.h5'
MEG_PEAKS = './data/evoked_peaks.csv'
MEG_PEAKS2 = './data/evoked_peaks_task_audvis.csv'
OUT_IMPORTANCE = './data/age_stacked_importance_{}_{}.h5'


data = pd.read_hdf(IN_PREDICTIONS, key='predictions')

# Add extra dfeatures
if False:
    # are included in latest code among predictions.
    meg_extra = pd.read_hdf(MEG_EXTRA_DATA, key='MEG_rest_extra')[['alpha_peak']]
    meg_peaks = pd.read_csv(MEG_PEAKS).set_index('subject')[['aud', 'vis']]
    meg_peaks2 = pd.read_csv(MEG_PEAKS2).set_index('subject')[['audvis']]
    meg_peaks.columns = ['MEG ' + cc for cc in meg_peaks.columns]
    meg_peaks2.columns = ['MEG ' + cc for cc in meg_peaks2.columns]
    meg_extra.columns = ['MEG ' + cc for cc in meg_extra.columns]

    data = data.join(meg_extra).join(meg_peaks).join(meg_peaks2)

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

all_connectivity = [f'MEG {tt} {fb}' for tt in meg_source_types
                    if 'diag' not in tt for fb in FREQ_BANDS]

power_by_freq = [f'MEG {tt} {fb}' for tt in meg_source_types
                 if 'diag' in tt and 'power' in tt for fb in FREQ_BANDS]
envelope_by_freq = [f'MEG {tt} {fb}' for tt in meg_source_types
                    if 'diag' in tt and 'envelope' in tt for fb in FREQ_BANDS]

envelope_cov = [f'MEG {tt} {fb}' for tt in meg_source_types
                if 'cross' in tt and 'envelope' in tt for fb in FREQ_BANDS]

power_cov = [f'MEG {tt} {fb}' for tt in meg_source_types
             if 'cross' in tt and 'power' in tt for fb in FREQ_BANDS]

stacked_keys = {
    # 'MEG-power-envelope-by-freq': power_by_freq + envelope_by_freq,
    # 'connectivity': all_connectivity,
    # 'MEG-power-connectivity': (power_by_freq +
    #                            envelope_by_freq + all_connectivity),
    # 'MEG-all-no-diag': ({cc for cc in data.columns if 'MEG' in cc} -
    #                     {'MEG envelope diag', 'MEG power diag'}),
    'MEG all': [cc for cc in data.columns if 'MEG' in cc]
}

if False:
    MRI = ['Cortical Surface Area', 'Cortical Thickness', 'Subcortical Volumes',
           'Connectivity Matrix, MODL 256 tan']
    stacked_keys['ALL'] = list(stacked_keys['MEG all']) + MRI
    stacked_keys['ALL no fMRI'] = list(stacked_keys['MEG all']) + MRI[:3]
    stacked_keys['ALL MRI'] = MRI


def run_importance(data, stacked_keys, drop_na='global'):
    all_results = dict()
    for key, sel in stacked_keys.items():
        this_data = data[sel]
        if drop_na == 'local':
            mask = this_data.dropna().index
        elif drop_na == 'global':
            mask = data.dropna().index
        else:
            mask = this_data.index
        X = this_data.loc[mask].values
        y = data['age'].loc[mask].values

        if drop_na is False:
            # code missings to make the tress learn from it.
            X_left = X.copy()
            X_left[this_data.isna().values] = -1000
            X_right = X.copy()
            X_right[this_data.isna().values] = 1000
            assert np.sum(np.isnan(X_left)) == 0
            assert np.sum(np.isnan(X_right)) == 0
            assert np.min(X_left) == -1000
            assert np.max(X_right) == 1000
            X = np.concatenate([X_left, X_right], axis=1)

        n_estimators = 1 if DEBUG else 1000

        regs = [
            ('rf_msqrt',
             RandomForestRegressor(n_estimators=n_estimators,
                                   n_jobs=N_JOBS,
                                   max_features='sqrt',
                                   max_depth=5,
                                   random_state=42)),
         ]

        #####
        groups = data.fold_idx.loc[mask].values.astype(int)
        logo = LeaveOneGroupOut()
        importance_result = list()
        n_permuations = 1 if DEBUG else N_PERMUATIONS

        for fold, (train, test) in enumerate(logo.split(X, y, groups)):
            for mod_type, reg in regs:
                print(f'fitting: {mod_type} fold: {fold}')
                # inside fold, inside repeat
                reg.fit(X[train], y[train])

                mdi_importance = dict(zip(sel, reg.feature_importances_))
                mdi_importance.update(
                    {'imp_metric': 'mdi', 'mod_type': mod_type, 'fold_idx': fold})
                importance_result.append(mdi_importance)

                permutation_result = permutation_importance(
                    estimator=reg, X=X[test], y=y[test],
                    n_repeats=n_permuations,
                    n_jobs=N_JOBS,
                    scoring='neg_mean_absolute_error')
                perm_importance = dict(zip(sel,
                                           permutation_result['importances_mean']))
                perm_importance.update(
                    {'imp_metric': 'permutation', 'mod_type': mod_type,
                     'fold_idx': fold})
                importance_result.append(perm_importance)
        #####
        all_results[key] = pd.DataFrame(importance_result)

    return all_results


DEBUG = False
if DEBUG:
    N_JOBS = 1
    data = data.iloc[::6]
    stacked_keys = {k: v for k, v in stacked_keys.items()
                    if k == 'MEG-all-no-diag'}

out = {}
for repeat in range(N_REPEATS):
    this_out = run_importance(data.query(f"repeat == {repeat}"),
                              stacked_keys)
    for key, val in this_out.items():
        val['repeat'] = repeat
        if not key in out:
            out[key] = val
        else:
            out[key] = out[key].append(val)

for key, val in out.items():
    val.to_hdf(OUT_IMPORTANCE.format(len(out), DROPNA), key=key)
