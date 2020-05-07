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


def _make_conductivty_model(subject):
    error = "None"
    try:
        bem_surfs = mne.bem.make_bem_model(
            subject=subject, ico=4, conductivity=(0.3,),
            subjects_dir=subjects_dir, verbose=None)

        conductivity_model = mne.bem.make_bem_solution(bem_surfs)
        mne.bem.write_bem_solution(
            op.join(subjects_dir, subject, 'bem', '%s-meg-bem.fif' % subject),
            conductivity_model)
    except Exception as ee:
        error = repr(ee)
    outputs = glob.glob(op.join(subjects_dir, subject, 'bem', '*meg-bem.fif'))
    out = dict(subject=subject, outputs=outputs, error=error)
    return out


out = Parallel(n_jobs=44)(
    delayed(_make_conductivty_model)(subject=subject)
    for subject in subjects)

h5io.write_hdf5('make_boundary_element_models.h5', out, overwrite=True)
