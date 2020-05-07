"""Plot mean absolute error (MAE) figures.

Two types of plots are done:
    - MAE versus the chronological age,
    - MAE of one modality versus MAE of another modality.
"""

# Author: Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import partial_dependence
from threadpoolctl import threadpool_limits
from mne.externals import h5io
from joblib import Parallel, delayed

N_JOBS = 10
N_THREADS = 5
DROPNA = "global"
N_REPEATS = 10
IN_PREDICTIONS = "./data/age_prediction_exp_data_na_denis_{}-rep.h5".format(
    N_REPEATS)
MEG_EXTRA_DATA = "./data/meg_extra_data.h5"
MEG_PEAKS = "./data/evoked_peaks.csv"
MEG_PEAKS2 = "./data/evoked_peaks_task_audvis.csv"
OUT_DEPENDENCE = "./data/age_stacked_dependence_{}.h5"


data = pd.read_hdf(IN_PREDICTIONS, key="predictions")

# Add extra dfeatures
meg_extra = pd.read_hdf(MEG_EXTRA_DATA, key="MEG_rest_extra")[["alpha_peak"]]
meg_peaks = pd.read_csv(MEG_PEAKS).set_index("subject")[["aud", "vis"]]
meg_peaks2 = pd.read_csv(MEG_PEAKS2).set_index("subject")[["audvis"]]
meg_peaks.columns = ["MEG " + cc for cc in meg_peaks.columns]
meg_peaks2.columns = ["MEG " + cc for cc in meg_peaks2.columns]
meg_extra.columns = ["MEG " + cc for cc in meg_extra.columns]

data = data.join(meg_extra).join(meg_peaks).join(meg_peaks2)

aMRI = ["Cortical Surface Area", "Cortical Thickness", "Subcortical Volumes"]
MRI = aMRI + ["Connectivity Matrix, MODL 256 tan"]

#  we put in here keys ordered by importance.
dependence_map = {
    # "1d" is the list of features for which we want PDPs.
    "MEG all": {"1d": ["MEG envelope diag",
                       "MEG power diag",
                       "MEG mne_envelope_cross alpha",
                       "MEG mne_envelope_cross beta_low",
                       "MEG mne_power_diag beta_low",
                       "MEG mne_envelope_diag beta_low",
                       "MEG mne_envelope_cross theta",
                       "MEG mne_envelope_corr alpha",
                       "MEG mne_envelope_corr beta_low",
                       "MEG mne_power_cross beta_high"],
                # keys are the columns used to fit the model.
                "keys": [cc for cc in data.columns if "MEG" in cc]},
    "ALL no fMRI": {"1d": ["Cortical Surface Area",
                           "Cortical Thickness",
                           "Subcortical Volumes",
                           "MEG power diag",
                           "MEG envelope diag",
                           "MEG mne_envelope_cross alpha",
                           "MEG mne_envelope_cross beta_low",
                           "MEG mne_power_diag beta_low",
                           "MEG mne_envelope_diag beta_low",
                           "MEG mne_envelope_cross theta",
                           "MEG mne_envelope_corr alpha"],
                    "2d": list(
                        combinations(
                            ["Cortical Thickness",
                             "Subcortical Volumes",
                             "MEG mne_power_diag beta_low",
                             "MEG power diag"], 2)),
                  # keys are the columns used to fit the model.
                  "keys": [cc for cc in data.columns if "MEG" in cc] + aMRI},
    "ALL MRI": {"1d": MRI,
                "2d": list(combinations(MRI, 2)),
                "keys": MRI},
    "ALL": {"1d": ["Cortical Surface Area",
                   "Cortical Thickness",
                   "Subcortical Volumes",
                   "Connectivity Matrix, MODL 256 tan",
                   "MEG power diag",
                   "MEG envelope diag",
                   "MEG mne_envelope_cross alpha",
                   "MEG mne_envelope_cross beta_low",
                   "MEG mne_power_diag beta_low",
                   "MEG mne_envelope_diag beta_low",
                   "MEG mne_envelope_cross theta",
                   "MEG mne_envelope_corr alpha"],
            "2d": list(
                combinations(
                    ["Cortical Thickness",
                     "Subcortical Volumes",
                     "Connectivity Matrix, MODL 256 tan",
                     "MEG power diag"], 2)),
            "keys": [cc for cc in data.columns if "MEG" in cc] + MRI}
}
# now we can add missing combinations for 2D dependene.
dependence_map["MEG all"]["2d"] = list(
    combinations(dependence_map["MEG all"]["1d"][:4], 2))


def compute_pdp(reg, X, columns, vars_2d):
    if isinstance(vars_2d, str):
        vars_2d = [vars_2d]
    feat_idx = [columns.index(vv) for vv in vars_2d]
    return partial_dependence(estimator=reg, X=X,
                              percentiles=(0.05, .95),
                              features=[feat_idx])


def run_dependence(data, dependence_map, n_jobs, drop_na="global"):
    all_results = list()
    for key in dependence_map:
        sel = dependence_map[key]["keys"]
        this_data = data[sel]
        if drop_na == "local":
            mask = this_data.dropna().index
        elif drop_na == "global":
            mask = data.dropna().index
        else:
            mask = this_data.index
        X = this_data.loc[mask].values
        y = data["age"].loc[mask].values

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

        n_estimators = 1000
        regs = [
            ("rf_msqrt",
             RandomForestRegressor(n_estimators=n_estimators,
                                   n_jobs=N_JOBS,
                                   max_features="sqrt",
                                   max_depth=5,
                                   random_state=42)),
            ("rf_m1",
             RandomForestRegressor(n_estimators=n_estimators,
                                   max_features=1,
                                   max_depth=5,
                                   n_jobs=N_JOBS,
                                   random_state=42)),
            ("et_m1",
             ExtraTreesRegressor(n_estimators=n_estimators,
                                 max_features=1,
                                 max_depth=5,
                                 n_jobs=N_JOBS,
                                 random_state=42))]
        if DEBUG:
            regs = regs[:1]
            regs[0][1].n_estimators = 1
        # For each set of stacked modsl,
        # we fit the model variants we used for importance computing.
        with threadpool_limits(limits=N_THREADS, user_api="blas"):
            for mod_type, reg in regs:
                # idea: bootstrap predictions by subsamping tress and
                # hacking fitted objects here the estimator list is
                # overwritten with bootstraps.
                reg.fit(X, y)
                # first we compute the 1d dependence for this configuration

                pdp_output = {"mod_type": mod_type, "stack_model": key}
                out_1d = Parallel(n_jobs=n_jobs)(delayed(compute_pdp)(
                    reg=reg, X=X, columns=list(this_data.columns),
                    vars_2d=vv) for vv in dependence_map[key]["1d"])
                pdp_output["1d"] = dict(zip(dependence_map[key]["1d"], out_1d))

                out_2d = Parallel(n_jobs=n_jobs)(delayed(compute_pdp)(
                    reg=reg, X=X, columns=list(this_data.columns),
                    vars_2d=vv) for vv in dependence_map[key]["2d"])
                labels = ["--".join(k) for k in dependence_map[key]["2d"]]
                pdp_output["2d"] = dict(zip(labels, out_2d))
                all_results.append(pdp_output)
    return all_results


DEBUG = False
if DEBUG:
    N_JOBS = 1
    data = data.iloc[::10]
    dependence_map = {k: v for k, v in dependence_map.items()
                      if k == "MEG all"}
    dependence_map["MEG all"]["1d"] = dependence_map["MEG all"]["1d"][:2]
    dependence_map["MEG all"]["2d"] = [
        tuple(dependence_map["MEG all"]["1d"][:2])]

out = run_dependence(
    data.query("repeat == 0"), dependence_map, n_jobs=N_JOBS)

h5io.write_hdf5(
    OUT_DEPENDENCE.format("model-full"),  out, compression=9,
    overwrite=True)
