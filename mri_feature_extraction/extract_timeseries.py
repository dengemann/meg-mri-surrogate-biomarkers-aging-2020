"""
Extract timeseries from fMRI volumes.

This script aimed at extracting several time series using the Cam-CAN
results with different atlases and dump the results somewhere.
"""

import os
from os.path import join

from camcan.datasets import load_camcan_rest
from camcan.preprocessing import extract_timeseries

from nilearn.datasets import fetch_atlas_basc_multiscale_2015

import joblib
from joblib import Parallel, delayed, Memory


# path to the Cam-CAN data set
STORAGE_HOME = '/storage/tompouce/okozynet/camcan'
CAMCAN_PREPROCESSED = '/storage/data/camcan/camcan_preproc'
CAMCAN_PATIENTS_EXCLUDED = join(STORAGE_HOME, 'excluded_patients.csv')
CAMCAN_TIMESERIES = join(STORAGE_HOME, 'timeseries')
# path to the atlases
# modl atlas from task-fMRI was used
ATLASES = ['/storage/store/derivatives/OpenNeuro/modl/256/0.0001/maps.nii.gz',
           fetch_atlas_basc_multiscale_2015().scale197]
ATLASES_DESCR = ['modl256', 'basc197']
# path for the caching
CACHE_TIMESERIES = join(STORAGE_HOME, 'cache/timeseries')
if not os.path.exists(CACHE_TIMESERIES):
    os.makedirs(CACHE_TIMESERIES)
MEMORY = Memory(CACHE_TIMESERIES)

N_JOBS = 10

dataset = load_camcan_rest(data_dir=CAMCAN_PREPROCESSED,
                           patients_excluded=CAMCAN_PATIENTS_EXCLUDED)
for atlas, atlas_descr in zip(ATLASES, ATLASES_DESCR):

    time_series = Parallel(n_jobs=N_JOBS, verbose=1)(delayed(
        extract_timeseries)(func, atlas=atlas, confounds=confounds,
                            memory=MEMORY, memory_level=0)
        for func, confounds in zip(dataset.func, dataset.motion))

    for ts, subject_id in zip(time_series, dataset.subject_id):
        path_subject = os.path.join(CAMCAN_TIMESERIES, subject_id, atlas_descr)
        if not os.path.exists(path_subject):
            os.makedirs(path_subject)
        filename = os.path.join(path_subject,
                                '%s_task-Rest_confounds.pkl' % subject_id)
        joblib.dump(ts, filename)
