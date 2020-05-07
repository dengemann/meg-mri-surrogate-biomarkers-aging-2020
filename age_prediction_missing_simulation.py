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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from scipy.special import expit
from scipy.stats import zscore

N_REPEATS = 10
N_JOBS = 10
N_THREADS = 10
N_SPLITS = 10

IN_PREDICTIONS = f'./data/age_prediction_exp_data_na_denis_{N_REPEATS}-rep.h5'
SCORES = './data/age_stacked_scores_{}.csv'
OUT_PREDICTIONS = './data/age_stacked_predictions_{}.csv'

data = pd.read_hdf(IN_PREDICTIONS, key='predictions')
y = data.query('repeat == 0').age.values
X = data.query('repeat == 0').iloc[:, 1:-1]

# prepare missing value coding
X[~X.isna().values] = 0

X_left = X.copy()
X_left[X.isna().values] = -1000
X_right = X.copy()
X_right[X.isna().values] = 1000

assert np.sum(np.isnan(X_left.values)) == 0
assert np.sum(np.isnan(X_right.values)) == 0
assert np.min(X_left.values) == -1000
assert np.max(X_right.values) == 1000
X = np.concatenate([X_left.values, X_right.values], axis=1)

param_grid = {'max_depth': [4, 6, 8, None]}
if X.shape[1] > 10:
    param_grid['max_features'] = (['log2', 'sqrt', None])


def run_cv(X, y):
    reg = GridSearchCV(
        RandomForestRegressor(n_estimators=1000,
                              random_state=42),
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        iid=False,
        cv=5)
     
    scores = list()
    for repeat in range(N_REPEATS):
        # make more or less arbitrary superstitious seed from repeat

        cv = KFold(n_splits=N_SPLITS, random_state=repeat * 7,  # seed fixed
                   shuffle=True)                                # seleative to main
        out = cross_val_score(X=X, y=y, cv=cv, estimator=reg,
                              scoring='neg_mean_absolute_error', n_jobs=N_JOBS)
        scores.extend(out)
    print(round(-np.mean(scores),3))
    return scores


all_scores = list()

df = pd.DataFrame()
df['score'] = run_cv(X, y)
df['prob'] = 0
df['fold'] = range(100)
all_scores.append(df)

# now let's increase the probability of missingness with ages

beta_ranges = (0.1, 0.2, 0.3, 0.4, 0.5)

for beta in beta_ranges:
    p = expit(-3 + zscore(y) * beta)
    size = (p * X.shape[1]).astype(int)
    this_X = X.copy()
    for ii, sz in enumerate(size):
        this_X[ii, :sz] = 1

    this_scores = run_cv(this_X, y)

    df = pd.DataFrame()
    df['score'] = this_scores
    df['beta'] = beta
    df['p_min'] = p.min()
    df['p_mean'] = p.mean()
    df['p_max'] = p.max()
    df['fold'] = range(100)
    all_scores.append(df)

results = pd.concat(all_scores)
results.to_csv("./data/missing_val_simulations.csv")
