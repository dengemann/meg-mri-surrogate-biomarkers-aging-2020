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
from sklearn.model_selection import (GridSearchCV, LeaveOneGroupOut)
from sklearn.metrics import mean_absolute_error
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

N_REPEATS = 10
N_JOBS = 10
N_THREADS = 5

IN_PREDICTIONS = f'./data/age_prediction_exp_data_na_denis_{N_REPEATS}-rep.h5'
SCORES = './data/age_stacked_scores_{}.csv'
OUT_PREDICTIONS = './data/age_stacked_predictions_{}.csv'


data = pd.read_hdf(IN_PREDICTIONS, key='predictions')

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

meg_handcrafted = [
    'MEG alpha_peak',
    'MEG 1/f low',
    'MEG 1/f gamma',
    'MEG aud',
    'MEG vis',
    'MEG audvis'
]

meg_cat_powers = [
    'MEG power diag',
    'MEG envelope diag'
]

meg_powers = meg_cat_powers + power_by_freq + envelope_by_freq

meg_cross_powers = power_cov + envelope_cov

meg_corr = [f'MEG {tt} {fb}' for tt in meg_source_types
            if 'corr' in tt for fb in FREQ_BANDS]

stacked_keys = {
    'MEG handcrafted': meg_handcrafted,
    'MEG powers': meg_powers,
    'MEG powers + cross powers': meg_powers + meg_cross_powers,
    'MEG powers + cross powers + handrafted': (
        meg_powers + meg_cross_powers + meg_handcrafted),
    'MEG cat powers + cross powers + correlation': (
        meg_cat_powers + meg_cross_powers + meg_corr),
    'MEG cat powers + cross powers + correlation + handcrafted': (
        meg_cat_powers + meg_cross_powers + meg_corr + meg_handcrafted),
    'MEG cross powers + correlation': envelope_cov + power_cov + meg_corr,
    'MEG powers + cross powers + correlation': (
        meg_powers + meg_cross_powers + meg_corr),
    # 'MEG powers + cross powers + correlation + handcrafted':
    'MEG all': meg_powers + meg_cross_powers + meg_corr + meg_handcrafted,
}

MRI = ['Cortical Surface Area', 'Cortical Thickness', 'Subcortical Volumes',
       'Connectivity Matrix, MODL 256 tan']
stacked_keys['ALL'] = list(stacked_keys['MEG all']) + MRI
stacked_keys['ALL no fMRI'] = list(stacked_keys['MEG all']) + MRI[:-1]
stacked_keys['MRI'] = MRI[:-1]
stacked_keys['fMRI'] = MRI[-1:]
stacked_keys['ALL MRI'] = MRI


def get_mae(predictions, key):
    scores = []
    for fold_idx, df in predictions.groupby('fold_idx'):
        scores.append(np.mean(np.abs(df[key] - df['age'])))
    return scores


def fit_predict_score(estimator, X, y, train, test, test_index):
    pred = pd.DataFrame(
        columns=['prediction'], index=test_index)
    with threadpool_limits(limits=N_THREADS, user_api='blas'):
        estimator.fit(X[train], y[train])
        y_pred = estimator.predict(X[test])
        score_mae = mean_absolute_error(y_true=y[test], y_pred=y_pred)
        pred['prediction'] = y_pred
        pred['y'] = y[test]
    return pred, score_mae


def run_stacked(data, stacked_keys, repeat_idx, drop_na):
    out_scores = pd.DataFrame()
    out_predictions = data.copy()
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
        fold_idx = data.loc[mask]['fold_idx'].values

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

        for column in sel:
            score = get_mae(data.loc[mask], column)
            if column not in out_scores:
                out_scores[column] = score
            elif out_scores[column].mean() < np.mean(score):
                out_scores[column] = score

        unstacked = out_scores[sel].values
        idx = unstacked.mean(axis=0).argmin()
        unstacked_mean = unstacked[:, idx].mean()
        unstacked_std = unstacked[:, idx].std()
        print(f'{key} | best unstacked MAE: {unstacked_mean} '
              f'(+/- {unstacked_std}')

        print('n =', len(X))

        param_grid = {'max_depth': [4, 6, 8, None]}
        if X.shape[1] > 10:
            param_grid['max_features'] = (['log2', 'sqrt', None])

        reg = GridSearchCV(
            RandomForestRegressor(n_estimators=1000,
                                  random_state=42),
            param_grid=param_grid,
            scoring='neg_mean_absolute_error',
            iid=False,
            cv=5)
        if DEBUG:
            reg = RandomForestRegressor(n_estimators=1000,
                                        max_features='log2',
                                        max_depth=6,
                                        random_state=42)

        cv = LeaveOneGroupOut()
        out_cv = Parallel(n_jobs=1)(delayed(fit_predict_score)(
            estimator=reg, X=X, y=y, train=train, test=test,
            test_index=this_data.loc[mask].index[test])
            for train, test in cv.split(X, y, fold_idx))

        out_cv = zip(*out_cv)
        predictions = next(out_cv)
        out_predictions[f'stacked_{key}'] = np.nan
        for pred in predictions:
            assert np.all(out_predictions.loc[pred.index]['age'] == pred['y'])
            out_predictions.loc[
                pred.index, f'stacked_{key}'] = pred['prediction'].values
        scores = np.array(next(out_cv))
        print(f'{key} | MAE : %0.3f (+/- %0.3f)' % (
            np.mean(scores), np.std(scores)))

        out_scores[key] = scores
    out_scores['repeat_idx'] = repeat_idx
    out_predictions['repeat_idx'] = repeat_idx
    return out_scores, out_predictions


DEBUG = False
if DEBUG:
    N_JOBS = 1
    stacked_keys = {'MEG all': meg_powers + meg_cross_powers + meg_handcrafted}

drop_na_scenario = (False, 'local', 'global')
for drop_na in drop_na_scenario[:1 if DEBUG else len(drop_na_scenario)]:
    out = Parallel(n_jobs=N_JOBS)(delayed(run_stacked)(
        data.query(f"repeat == {ii}"), stacked_keys, ii, drop_na)
        for ii in range(N_REPEATS))
    out = zip(*out)

    out_scores_meg = next(out)
    out_scores_meg = pd.concat(out_scores_meg, axis=0)
    out_scores_meg.to_csv(
        SCORES.format('meg' + drop_na if drop_na else '_na_coded'),
        index=True)

    out_predictions_meg = next(out)
    out_predictions_meg = pd.concat(out_predictions_meg, axis=0)
    out_predictions_meg.to_csv(
        OUT_PREDICTIONS.format('meg' + drop_na if drop_na else '_na_coded'),
        index=True)
