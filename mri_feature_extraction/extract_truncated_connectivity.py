"""Extract connectivity from fMRI timeseries of different duration."""

import os
from os.path import join

from camcan.datasets import load_camcan_rest
from camcan.preprocessing import extract_timeseries
import joblib
from joblib import Parallel, delayed, Memory
import numpy as np
from nilearn.datasets import fetch_atlas_basc_multiscale_2015
import pandas as pd

from camcan.datasets import load_camcan_timeseries_rest
from camcan.preprocessing import extract_connectivity


# path to the Cam-CAN data set
STORAGE_HOME = '/storage/tompouce/okozynet/camcan'
CAMCAN_PREPROCESSED = '/storage/data/camcan/camcan_preproc'
CAMCAN_PATIENTS_EXCLUDED = join(STORAGE_HOME, 'excluded_patients.csv')
CAMCAN_TIMESERIES = join(STORAGE_HOME, 'timeseries-truncated')
OUT_DIR = join(STORAGE_HOME, 'connectivity')
# path to the atlases
# modl atlas from task-fMRI was used
ATLASES = ['/storage/store/derivatives/OpenNeuro/modl/256/0.0001/maps.nii.gz',
           fetch_atlas_basc_multiscale_2015().scale197]
ATLASES_DESCR = ('modl256', 'basc197')
CONNECTIVITY_KINDS = ('correlation', 'tangent')
# path for the caching
CACHE_TIMESERIES = join(STORAGE_HOME, 'cache/timeseries')
if not os.path.exists(CACHE_TIMESERIES):
    os.makedirs(CACHE_TIMESERIES)
MEMORY = Memory(CACHE_TIMESERIES)

N_JOBS = 10
TIMESERIES_DURATIONS = (2 * 60, 4 * 60, 6 * 60, 8 * 60 + 40)  # seconds
dataset = load_camcan_rest(data_dir=CAMCAN_PREPROCESSED,
                           patients_excluded=CAMCAN_PATIENTS_EXCLUDED)

for dur in TIMESERIES_DURATIONS:
    ts_dir = join(CAMCAN_TIMESERIES, 'ts_%s' % dur)

    for atlas, atlas_descr in zip(ATLASES, ATLASES_DESCR):
        time_series = Parallel(n_jobs=N_JOBS, verbose=1)(delayed(
            extract_timeseries)(func, atlas=atlas, confounds=confounds,
                                memory=MEMORY, memory_level=0, duration=dur)
            for func, confounds in zip(dataset.func, dataset.motion))

        for ts, subject_id in zip(time_series, dataset.subject_id):
            path_subject = join(CAMCAN_TIMESERIES, ts_dir, subject_id,
                                atlas_descr)
            if not os.path.exists(path_subject):
                os.makedirs(path_subject)
            filename = join(path_subject,
                            '%s_task-Rest_confounds.pkl' % subject_id)
            joblib.dump(ts, filename)

# combine truncated connectivities
out_file = join(OUT_DIR, 'connect_data_truncated.h5')
# remove the output file if it exists
if os.path.exists(out_file):
    os.remove(out_file)

for dur in TIMESERIES_DURATIONS:
    str_dur = 'ts_%s' % dur
    ts_dir = join(CAMCAN_TIMESERIES, str_dur)

    for connect_kind in CONNECTIVITY_KINDS:
        for sel_atlas in ATLASES_DESCR:
            print('*******************************************************')
            print(f'Reading timeseries files for {sel_atlas}')

            hdf5_key = f'{str_dur}_{connect_kind}_{sel_atlas}'
            dataset = load_camcan_timeseries_rest(data_dir=ts_dir,
                                                  atlas=sel_atlas)
            connectivities = extract_connectivity(dataset.timeseries,
                                                  kind=connect_kind)

            connect_data = None
            subjects = tuple(s[4:] for s in dataset.subject_id)

            for i, s in enumerate(subjects):
                if connect_data is None:
                    columns = np.arange(start=0, stop=len(connectivities[i]))
                    connect_data = pd.DataFrame(index=subjects,
                                                columns=columns,
                                                dtype=float)
                    if connect_kind == 'correlation':
                        # save and apply Fisher's transform
                        connect_data.loc[s] = np.arctanh(connectivities[i])
                    else:
                        connect_data.loc[s] = connectivities[i]
                else:
                    if connect_kind == 'correlation':
                        # save and apply Fisher's transform
                        connect_data.loc[s] = np.arctanh(connectivities[i])
                    else:
                        connect_data.loc[s] = connectivities[i]

            connect_data.to_hdf(out_file, key=hdf5_key, complevel=9)
