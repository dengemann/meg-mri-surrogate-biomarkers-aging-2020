"""Extract structural data from FreeSurfer outputs."""
import os
from os.path import isdir, join

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from camcan.preprocessing import get_structural_data
from camcan.utils import get_area, get_thickness

# import pdb; pdb.set_trace()
# test functions on MRI data
CAMCAN_CONNECTIVITY = '/storage/data/camcan/camcan_connectivity'
CAMCAN_FREESURFER = '/storage/store/data/camcan-mne/freesurfer'
OUT_DIR = '/storage/tompouce/okozynet/camcan/structural'
VOLUME_FILE = 'aseg.csv'
N_JOBS = 10
N_CORTICAL_FEATURES = 5124
# list of subjects that we have connectivity data for
subjects = [d[4:] for d in os.listdir(CAMCAN_CONNECTIVITY)
            if isdir(join(CAMCAN_CONNECTIVITY, d))]

get_str_data_del = delayed(get_structural_data)
delayed_functions = (get_str_data_del(CAMCAN_FREESURFER, s, OUT_DIR)
                     for s in subjects)
structural_data = Parallel(n_jobs=N_JOBS, verbose=1)(delayed_functions)

# some subjects had problems during intial segmentation,
# fixed files lie in a different folder
CAMCAN_FREESURFER_FIXED = '/storage/tompouce/okozynet/camcan/freesurfer'
subjects_fixed = ['CC410354']

delayed_functions = (get_str_data_del(CAMCAN_FREESURFER_FIXED, s, OUT_DIR)
                     for s in subjects)
structural_data = Parallel(n_jobs=N_JOBS, verbose=1)(delayed_functions)

subjects = tuple(d for d in os.listdir(OUT_DIR) if isdir(join(OUT_DIR, d)))
print(f'Found {len(subjects)} subjects')

area_data = None
thickness_data = None
volume_data = None

area_failed = []
thickness_failed = []
volume_failed = []

for s in subjects:
    subject_dir = join(OUT_DIR, s)

    try:
        t_area = get_area(subject_dir, n_points=N_CORTICAL_FEATURES)

        if area_data is None:
            area_data = pd.DataFrame(
                index=subjects,
                columns=np.arange(start=0, stop=N_CORTICAL_FEATURES),
                dtype=float)
            area_data.loc[s] = t_area
        else:
            area_data.loc[s] = t_area
    except FileNotFoundError:
        print(f'Cannot find area file for subject {s}')
        area_failed.append(s)

    try:
        t_thickness = get_thickness(subject_dir, n_points=N_CORTICAL_FEATURES)

        if thickness_data is None:
            thickness_data = pd.DataFrame(
                index=subjects,
                columns=np.arange(start=0, stop=N_CORTICAL_FEATURES),
                dtype=float)
            thickness_data.loc[s] = t_thickness
        else:
            thickness_data.loc[s] = t_thickness
    except FileNotFoundError:
        print(f'Cannot find thickness file for subject {s}')
        thickness_failed.append(s)

    try:
        volume = pd.read_csv(join(subject_dir, VOLUME_FILE), index_col=0)

        if volume_data is None:
            volume_data = volume
        else:
            volume_data = pd.concat([volume_data, volume])

    except FileNotFoundError:
        print(f'Cannot find volume file for subject {s}')
        volume_failed.append(s)

print('Failed to load area data for\n', area_failed)
print('Failed to load thickness data for\n', thickness_failed)
print('Failed to load volume data for\n', volume_failed)

out_file = join(OUT_DIR, 'structural_data.h5')
area_data.to_hdf(out_file, key='area', complevel=9)
thickness_data.to_hdf(out_file, key='thickness', complevel=9)
volume_data.to_hdf(out_file, key='volume', complevel=9)
