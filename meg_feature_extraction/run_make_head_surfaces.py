# Author: Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import config as cfg
import glob
import library as lib

import mne
from mne.externals import h5io
from joblib import Parallel, delayed

subjects = lib.utils.get_subjects(cfg.camcan_meg_raw_path)

subjects_dir = cfg.mne_camcan_freesurfer_path

mne.utils.set_log_level('warning')

def _make_headmodels(subject):
    # print(subject)
    # return subject
    print('running ', subject)
    error = 'None'
    try:
        mne.bem.make_watershed_bem(
            subject, subjects_dir=subjects_dir, overwrite=True)
    except Exception as ee:
        error = str(ee)
        print(subject, error)
    outputs = glob.glob(op.join(
        cfg.mne_camcan_freesurfer_path, subject, 'bem', '*'))
    return dict(subject=subject, bem_outputs=outputs, error=error)

out = Parallel(n_jobs=11)(
    delayed(_make_headmodels)(subject=subject)
    for subject in subjects)

h5io.write_hdf5('make_head_surfaces_output.h5', out, overwrite=True)
