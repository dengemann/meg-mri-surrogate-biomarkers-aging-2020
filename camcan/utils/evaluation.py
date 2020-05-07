"""Evaluation performance of models."""
from os import path

import numpy as np
import mne
from mne.datasets import sample
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import (cross_val_score,
                                     cross_val_predict,
                                     learning_curve,
                                     ShuffleSplit, check_cv)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..processing import StackingRegressor, SPoC


def _get_fold_indices(cv, X, y):
    fold = -np.ones(len(y), dtype=int)
    for fold_idx, (_, test) in enumerate(cv.split(X, y)):
        fold[test] = fold_idx
    assert np.all(fold >= 0)
    return fold


def run_ridge(data, subjects_data, cv=10, alphas=None, train_sizes=None,
              n_jobs=None):
    """Run ridge resgression.

    Parameters
    ----------
    data : pandas.DataFrame
        Features to be used for predictions.

    subjects_data : pandas.DataFrame
        Information about subjects from CamCAN dataset.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

    alphas : numpy.ndarray
        Values for parameter alpha to be tested. Default is
        np.logspace(start=-3, stop=1, num=50, base=10.0).

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))

    n_jobs : int or None, optional (default=None)
        The number of CPUs to use to do the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    """
    if alphas is None:
        alphas = np.logspace(-3, 5, 100)
    if train_sizes is None:
        train_sizes = np.linspace(.1, 1.0, 5)

    # prepare data, subjects age
    subjects = data.index.values  # unioin index, can have Nan
    y = subjects_data.loc[subjects]['age']

    reg = make_pipeline(StandardScaler(), RidgeCV(alphas))

    cv = check_cv(cv)
    index = data.index
    mae = list()
    r2 = list()

    df_pred = pd.DataFrame(columns=['fold', 'y_pred'], index=subjects,
                           dtype=float)

    for ii, (train, test) in enumerate(cv.split(data.values)):
        # we must mask on X as y has no nans
        # Nans will bee rowwise only
        train_mask = ~data.loc[index[train]].isna().values[:, 0]
        test_mask = ~data.loc[index[test]].isna().values[:, 0]

        X_train = data.loc[index[train]].values[train_mask]
        X_test = data.loc[index[test]].values[test_mask]

        y_train = y.loc[index[train]].values[train_mask]
        y_test = y.loc[index[test]].values[test_mask]

        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        mae.append(-mean_absolute_error(y_true=y_test, y_pred=y_pred))
        r2.append(r2_score(y_true=y_test, y_pred=y_pred))

        # we need nans on the prediction
        df_pred.loc[index[test[test_mask]], 'y_pred'] = y_pred
        # but not on the fold index
        df_pred.loc[index[test], 'fold'] = ii

    mask = ~data.isna().values[:, 0]
    train_sizes, train_scores, test_scores = \
        learning_curve(reg, data[mask].values, y[mask].values, cv=cv,
                       train_sizes=train_sizes,
                       scoring='neg_mean_absolute_error', n_jobs=n_jobs)
  
    return (df_pred, np.array(mae), np.array(r2), train_sizes, train_scores,
            test_scores)


def run_meg_spoc(data, subjects_data, cv=10, alphas=None,
                 train_sizes=None, fbands=None, n_jobs=None):
    """Run ridge resgression on MEG data with SPoC.

    Parameters
    ----------
    data : pandas.DataFrame
        Features to be used for predictions.

    subjects_data : pandas.DataFrame
        Information about subjects from CamCAN dataset.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

    alphas : numpy.ndarray
        Values for parameter alpha to be tested using RidgeCV. Default is
        np.logspace(start=-3, stop=1, num=50, base=10.0).

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))

    fbands : [(float, float)]
        List of frequency bands to be checked with SPoC.

    n_jobs : int or None, optional (default=None)
        The number of CPUs to use to do the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    """
    if alphas is None:
        alphas = np.logspace(-3, 5, 100)
    if train_sizes is None:
        train_sizes = np.linspace(.1, 1.0, 5)

    # read sample data to prepare picks for epochs
    data_path = sample.data_path()
    raw_fname = path.join(data_path,
                          'MEG/sample/sample_audvis_filt-0-40_raw.fif')
    raw = mne.io.read_raw_fif(raw_fname)
    info = raw.info
    picks = mne.pick_types(info, meg='mag')

    subjects = [d['subject'] for d in data if 'subject' in d]
    covs = np.array(tuple(d['covs'][:, picks][:, :, picks] for d
                          in data if 'subject' in d))

    # prepare data, subjects age
    y = subjects_data.loc[subjects].age.values
    X = np.arange(len(y))

    spoc = SPoC(covs=covs, fbands=fbands, spoc=True,
                n_components=len(picks), alpha=0.01)

    reg = make_pipeline(spoc, StandardScaler(),
                        RidgeCV(alphas))
    # Monte Carlo cross-validation
    cv_ss = ShuffleSplit(n_splits=cv, random_state=42)

    mae = cross_val_score(reg, X, y, scoring='neg_mean_absolute_error',
                          cv=cv_ss, n_jobs=n_jobs)
    r2 = cross_val_score(reg, X, y, scoring='r2', cv=cv_ss, n_jobs=n_jobs)
    y_pred = cross_val_predict(reg, X, y, cv=cv, n_jobs=n_jobs)

    train_sizes, train_scores, test_scores =\
        learning_curve(reg, X, y, cv=cv_ss, train_sizes=train_sizes,
                       scoring='neg_mean_absolute_error', n_jobs=n_jobs)

    df_pred = pd.DataFrame(y_pred, index=subjects, dtype=float)

    return df_pred, mae, r2, train_sizes, train_scores, test_scores


def run_stacking_spoc(named_data, subjects_data, cv=10, alphas=None,
                      train_sizes=None, fbands=None, n_jobs=None):
    """Run stacking.

    Parameters
    ----------
    named_data : list(tuple(str, pandas.DataFrame))
        List of tuples (name, data) with name and corresponding features
        to be used for predictions by linear models.

    subjects_data : pandas.DataFrame
        Information about subjects from CamCAN dataset.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

    alphas : numpy.ndarray
        Values for parameter alpha to be tested. Default is
        np.logspace(start=-3, stop=1, num=50, base=10.0).

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))

    fbands : [(float, float)]
        List of frequency bands to be checked with SPoC.

    n_jobs : int or None, optional (default=None)
        The number of CPUs to use to do the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    """
    if alphas is None:
        alphas = np.logspace(-3, 5, 100)
    if train_sizes is None:
        train_sizes = np.linspace(.1, 1.0, 5)

    rnd_state = 42
    names = []
    combined_data = []
    meg_data = None
    # extract data and estimator names
    for name, data in named_data:
        names.append(name)
        if name == 'meg':
            meg_data = data
            meg_subjects = tuple(d['subject'] for d in data if 'subject' in d)
            pseudo_data = np.arange(len(meg_subjects))
            combined_data.append(pd.DataFrame(pseudo_data, index=meg_subjects))
        else:
            combined_data.append(data)

    data = pd.concat(combined_data, axis=1, join='inner')
    # if we have meg data, we will provide only one column of
    # data for the classifiers
    feature_col_lens = tuple(d.shape[1] for d in combined_data)
    estimators = []
    subjects = data.index.values
    # prepare first-level estimators for stacking
    for i_data, _ in enumerate(named_data):
        feature_transformers = []
        ft_begin = 0
        ft_end = 0
        # prepare input information for ColumnTransformer
        for i_ct, (name, col_len) in enumerate(zip(names, feature_col_lens)):
            trans_name = ('pass_' if i_data == i_ct else 'drop_') + name
            transformer = 'passthrough' if i_data == i_ct else 'drop'
            ft_end = ft_end + col_len
            trans_slice = slice(ft_begin, ft_end)
            ft_begin = ft_begin + col_len
            feature_transformers.append((trans_name, transformer, trans_slice))

        est_name = 'reg_' + named_data[i_data][0]

        if est_name == 'reg_meg':
            if fbands is None:
                raise ValueError('fbands should be given for MEG classifier.')
            # read sample data to prepare picks for epochs
            data_path = sample.data_path()
            raw_fname = path.join(data_path,
                                  'MEG/sample',
                                  'sample_audvis_filt-0-40_raw.fif')
            raw = mne.io.read_raw_fif(raw_fname)
            info = raw.info
            picks = mne.pick_types(info, meg='mag')
            # if there is no subject information than we'll skip that entry
            covs = np.array(tuple(d['covs'][:, picks][:, :, picks] for d
                                  in meg_data if 'subject' in d))
            spoc = SPoC(covs=covs, fbands=fbands, spoc=True,
                        n_components=len(picks), alpha=0.01)

            est_pipeline = make_pipeline(
                ColumnTransformer(feature_transformers),
                spoc, StandardScaler(), RidgeCV(alphas))
        else:
            est_pipeline = make_pipeline(
                ColumnTransformer(feature_transformers),
                StandardScaler(), RidgeCV(alphas))
        estimators.append((est_name, est_pipeline))

    final_estimator = RandomForestRegressor(n_estimators=100,
                                            random_state=rnd_state,
                                            oob_score=True)
    reg = StackingRegressor(estimators=estimators,
                            final_estimator=final_estimator, cv=cv,
                            random_state=rnd_state, n_jobs=n_jobs)

    y = subjects_data.loc[subjects].age.values
    X = data.values

    cv = check_cv(cv)
    mae = cross_val_score(reg, X, y, scoring='neg_mean_absolute_error',
                          cv=cv, n_jobs=n_jobs)

    r2 = cross_val_score(reg, X, y, scoring='r2', cv=cv, n_jobs=n_jobs)
    y_pred = cross_val_predict(reg, X, y, cv=cv, n_jobs=n_jobs)

    train_sizes, train_scores, test_scores = \
        learning_curve(reg, X, y, cv=cv, train_sizes=train_sizes,
                       scoring='neg_mean_absolute_error', n_jobs=n_jobs)

    fold = _get_fold_indices(cv, X, y)
    df_pred = pd.DataFrame(dict(y=y_pred, fold=fold), index=subjects,
                           dtype=float)

    return df_pred, mae, r2, train_sizes, train_scores, test_scores


def run_stacking(named_data, subjects_data, cv=10, alphas=None,
                 train_sizes=None, n_jobs=None):
    """Run stacking.

    Parameters
    ----------
    named_data : list(tuple(str, pandas.DataFrame))
        List of tuples (name, data) with name and corresponding features
        to be used for predictions by linear models.

    subjects_data : pandas.DataFrame
        Information about subjects from CamCAN dataset.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

    alphas : numpy.ndarray
        Values for parameter alpha to be tested. Default is
        np.logspace(start=-3, stop=1, num=50, base=10.0).

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))

    n_jobs : int or None, optional (default=None)
        The number of CPUs to use to do the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    """
    if alphas is None:
        alphas = np.logspace(-3, 5, 100)
    if train_sizes is None:
        train_sizes = np.linspace(.1, 1.0, 5)

    rnd_state = 42
    names = []
    combined_data = []
    # extract data and estimator names
    for name, data in named_data:
        names.append(name)
        combined_data.append(data)

    data = pd.concat(combined_data, axis=1, join='inner')
    feature_col_lens = tuple(d.shape[1] for d in combined_data)
    estimators = []
    subjects = data.index.values
    # prepare first-level estimators for stacking
    for i_data, _ in enumerate(named_data):
        feature_transformers = []
        ft_begin = 0
        ft_end = 0
        # prepare input information for ColumnTransformer
        for i_ct, (name, col_len) in enumerate(zip(names, feature_col_lens)):
            trans_name = ('pass_' if i_data == i_ct else 'drop_') + name
            transformer = 'passthrough' if i_data == i_ct else 'drop'
            ft_end = ft_end + col_len
            trans_slice = slice(ft_begin, ft_end)
            ft_begin = ft_begin + col_len
            feature_transformers.append((trans_name, transformer, trans_slice))

        est_name = 'reg_' + named_data[i_data][0]

        est_pipeline = make_pipeline(
            ColumnTransformer(feature_transformers),
            StandardScaler(), RidgeCV(alphas))
        estimators.append((est_name, est_pipeline))

    final_estimator = RandomForestRegressor(n_estimators=100,
                                            random_state=rnd_state,
                                            oob_score=True)
    reg = StackingRegressor(estimators=estimators,
                            final_estimator=final_estimator, cv=cv,
                            random_state=rnd_state, n_jobs=n_jobs)

    y = subjects_data.loc[subjects].age.values
    X = data.values

    cv = check_cv(cv)
    mae = cross_val_score(reg, X, y, scoring='neg_mean_absolute_error',
                          cv=cv, n_jobs=n_jobs)

    r2 = cross_val_score(reg, X, y, scoring='r2', cv=cv, n_jobs=n_jobs)
    y_pred = cross_val_predict(reg, X, y, cv=cv, n_jobs=n_jobs)

    train_sizes, train_scores, test_scores = \
        learning_curve(reg, X, y, cv=cv, train_sizes=train_sizes,
                       scoring='neg_mean_absolute_error', n_jobs=n_jobs)

    fold = _get_fold_indices(cv, X, y)
    df_pred = pd.DataFrame(dict(y=y_pred, fold=fold), index=subjects,
                           dtype=float)

    return df_pred, mae, r2, train_sizes, train_scores, test_scores


def train_stacked_regressor(named_data, subjects_data, cv=10, alphas=None,
                            train_sizes=None, n_jobs=None):
    """Return stacked classifier trained on provided data.

    For MEG data features estimated in the source space should be provided.

    Parameters
    ----------
    named_data : list(tuple(str, pandas.DataFrame))
        List of tuples (name, data) with name and corresponding features
        to be used for predictions by linear models.

    subjects_data : pandas.DataFrame
        Information about subjects from CamCAN dataset.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

    alphas : numpy.ndarray
        Values for parameter alpha to be tested. Default is
        np.logspace(start=-3, stop=1, num=50, base=10.0).

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))

    n_jobs : int or None, optional (default=None)
        The number of CPUs to use to do the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    """
    if alphas is None:
        alphas = np.logspace(-3, 5, 100)
    if train_sizes is None:
        train_sizes = np.linspace(.1, 1.0, 5)

    rnd_state = 42
    names = []
    combined_data = []
    # extract data and estimator names
    for name, data in named_data:
        names.append(name)
        combined_data.append(data)

    data = pd.concat(combined_data, axis=1, join='inner')
    feature_col_lens = tuple(d.shape[1] for d in combined_data)
    estimators = []
    subjects = data.index.values
    # prepare first-level estimators for stacking
    for i_data, _ in enumerate(named_data):
        feature_transformers = []
        ft_begin = 0
        ft_end = 0
        # prepare input information for ColumnTransformer
        for i_ct, (name, col_len) in enumerate(zip(names, feature_col_lens)):
            trans_name = ('pass_' if i_data == i_ct else 'drop_') + name
            transformer = 'passthrough' if i_data == i_ct else 'drop'
            ft_end = ft_end + col_len
            trans_slice = slice(ft_begin, ft_end)
            ft_begin = ft_begin + col_len
            feature_transformers.append((trans_name, transformer, trans_slice))

        est_name = 'reg_' + named_data[i_data][0]

        est_pipeline = make_pipeline(
            ColumnTransformer(feature_transformers),
            StandardScaler(), RidgeCV(alphas))
        estimators.append((est_name, est_pipeline))

    final_estimator = RandomForestRegressor(n_estimators=100,
                                            random_state=rnd_state,
                                            oob_score=True)
    reg = StackingRegressor(estimators=estimators,
                            final_estimator=final_estimator, cv=cv,
                            random_state=rnd_state, n_jobs=n_jobs)

    y = subjects_data.loc[subjects].age.values
    X = data.values

    reg.fit(X, y)

    return reg, X, y
