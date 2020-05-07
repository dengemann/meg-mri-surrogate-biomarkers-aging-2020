"""
Extract connectivity from fMRI timeseries.

This script aimed at extracting the connectivity matrices from the time
series using the Cam-CAN for different atalases and dump them somewhere.
"""

import os

from camcan.datasets import load_camcan_timeseries_rest
from camcan.preprocessing import extract_connectivity

import joblib
from joblib import Parallel, delayed

# path to the Cam-CAN data set
CAMCAN_TIMESERIES = '/home/mehdi/data/camcan/camcan_timeseries'
CAMCAN_PATIENTS_EXCLUDED = None
CAMCAN_CONNECTIVITY = '/home/mehdi/data/camcan/camcan_connectivity'
# path to the different atlases
ATLASES = ['msdl', 'basc064', 'basc122', 'basc197']
# path for the different kind of connectivity matrices
CONNECTIVITY_KIND = ['correlation', 'partial correlation', 'tangent']
# path for the caching
CACHE_CONNECTIVITY = '/home/mehdi/data/camcan/cache/connectivity'
if not os.path.exists(CACHE_CONNECTIVITY):
    os.makedirs(CACHE_CONNECTIVITY)
MEMORY = None

N_JOBS = 3

# We need loky to have nested multi-processing
for atlas in ATLASES:

    dataset = load_camcan_timeseries_rest(
        data_dir=CAMCAN_TIMESERIES,
        atlas=atlas,
        patients_excluded=CAMCAN_PATIENTS_EXCLUDED)

    connectivities = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(extract_connectivity)(dataset.timeseries, kind=kind)
        for kind in CONNECTIVITY_KIND)

    for kind_idx, kind in enumerate(CONNECTIVITY_KIND):
        for subject_id_idx, subject_id in enumerate(dataset.subject_id):
            path_subject = os.path.join(CAMCAN_CONNECTIVITY, subject_id,
                                        atlas, kind)
            if not os.path.exists(path_subject):
                os.makedirs(path_subject)

            filename = os.path.join(path_subject,
                                    '%s_task-Rest_confounds.pkl' % subject_id)
            joblib.dump(connectivities[kind_idx][subject_id_idx], filename)
