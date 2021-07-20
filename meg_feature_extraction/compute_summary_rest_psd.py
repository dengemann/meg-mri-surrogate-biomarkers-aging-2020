import os.path as op

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import mne

import config as cfg
import library as lib


subjects = lib.utils.get_subjects(cfg.camcan_meg_raw_path)


def _summarize_psds(subject, kind, fun='mean'):

    out_path = op.join(
        cfg.derivative_path, subject)

    in_fname = op.join(out_path, 'rest_sensors_psd_welch-epo.h5')
    h5 = mne.externals.h5io.read_hdf5(in_fname)
    my_fun = None
    if fun == 'mean':
        my_fun = np.mean
    else:
        raise NotImplementedError('Not implemented!')

    psd_out = my_fun(h5['psd'], 0)

    out_fname = op.join(out_path, 'rest_sensors_psd_welch-ave.h5')

    mne.externals.h5io.write_hdf5(
        out_fname, {'psd': psd_out, 'freqs': h5['freqs']},
        overwrite=True)
    return {}


def _run_all(subject, kind):
    mne.utils.set_log_level('warning')
    print(subject)
    error = 'None'
    result = dict()
    try:
        result = _summarize_psds(subject, kind)
    except Exception as err:
        error = repr(err)
        print(error)

    out = dict(subject=subject, kind=kind, error=error)
    out.update(result)
    return out


out_df_rest_psd_summary = Parallel(n_jobs=50)(
    delayed(_run_all)(subject=subject, kind='rest')
    for subject in subjects)

out_df_rest_psd_summary = pd.DataFrame(out_df_rest_psd_summary)
out_df_rest_psd_summary.to_csv(
    op.join(
        cfg.derivative_path,
        'log_compute_summary_rest_psd.csv'))
