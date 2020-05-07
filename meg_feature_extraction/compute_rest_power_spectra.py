# Author: Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import pandas as pd
import mne

from joblib import Parallel, delayed
from autoreject import get_rejection_threshold

import config as cfg
import library as lib


subjects = lib.utils.get_subjects(cfg.camcan_meg_raw_path)

max_filter_info_path = op.join(
    cfg.camcan_meg_path,
    "data_nomovecomp/"
    "aamod_meg_maxfilt_00001")


def _parse_bads(subject, kind):
    sss_log = op.join(
        max_filter_info_path, subject,
        kind, "mf2pt2_{kind}_raw.log".format(kind=kind))

    try:
        bads = lib.preprocessing.parse_bad_channels(sss_log)
    except Exception as err:
        print(err)
        bads = []
    # first 100 channels ommit the 0.
    bads = [''.join(['MEG', '0', bb.split('MEG')[-1]])
            if len(bb) < 7 else bb for bb in bads]
    return bads


def _get_global_reject_ssp(raw):
    eog_epochs = mne.preprocessing.create_eog_epochs(raw)
    if len(eog_epochs) >= 5:
        reject_eog = get_rejection_threshold(eog_epochs, decim=8)
        del reject_eog['eog']
    else:
        reject_eog = None

    ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)
    if len(ecg_epochs) >= 5:
        reject_ecg = get_rejection_threshold(ecg_epochs, decim=8)
    else:
        reject_eog = None

    if reject_eog is None:
        reject_eog = reject_ecg
    if reject_ecg is None:
        reject_ecg = reject_eog
    return reject_eog, reject_ecg


def _run_maxfilter(raw, subject, kind):

    bads = _parse_bads(subject, kind)

    raw.info['bads'] = bads

    raw = lib.preprocessing.run_maxfilter(raw, coord_frame='head')
    return raw


def _compute_add_ssp_exg(raw):
    reject_eog, reject_ecg = _get_global_reject_ssp(raw)

    proj_eog = mne.preprocessing.compute_proj_eog(
        raw, average=True, reject=reject_eog, n_mag=1, n_grad=1, n_eeg=1)

    proj_ecg = mne.preprocessing.compute_proj_ecg(
        raw, average=True, reject=reject_ecg, n_mag=1, n_grad=1, n_eeg=1)

    raw.add_proj(proj_eog[0])
    raw.add_proj(proj_ecg[0])


def _get_global_reject_epochs(raw):
    duration = 3.
    events = mne.make_fixed_length_events(
        raw, id=3000, start=0, duration=duration)

    epochs = mne.Epochs(
        raw, events, event_id=3000, tmin=0, tmax=duration, proj=False,
        baseline=None, reject=None)
    epochs.apply_proj()
    epochs.load_data()
    epochs.pick_types(meg=True)
    reject = get_rejection_threshold(epochs, decim=8)
    return reject


def _compute_rest_psd(subject, kind):

    fname = op.join(
        cfg.camcan_meg_raw_path,
        subject, kind, '%s_raw.fif' % kind)

    raw = mne.io.read_raw_fif(fname)
    mne.channels.fix_mag_coil_types(raw.info)
    raw = _run_maxfilter(raw, subject, kind)
    _compute_add_ssp_exg(raw)

    reject = _get_global_reject_epochs(raw)

    stop = raw.times[-1]
    duration = 30.
    overlap = 8.
    stop = raw.times[-1]
    events = mne.make_fixed_length_events(
        raw, id=3000, start=0, duration=overlap,
        stop=stop - duration)

    epochs = mne.Epochs(
        raw, events, event_id=3000, tmin=0, tmax=duration, proj=True,
        baseline=None, reject=reject, preload=True, decim=1)
    #  make sure not to decim it induces power line artefacts!

    picks = mne.pick_types(raw.info, meg=True)
    psd, freqs = mne.time_frequency.psd_welch(
        epochs, fmin=0, fmax=150, n_fft=4096,  # ~12 seconds
        n_overlap=512,
        picks=picks)

    out_path = op.join(
        cfg.derivative_path, subject)

    out_fname = op.join(out_path, 'rest_sensors_psd_welch-epo.h5')

    mne.externals.h5io.write_hdf5(
        out_fname, {'psd': psd, 'freqs': freqs},
        overwrite=True)
    return {'n_events': len(events), 'n_events_good': psd.shape[0]}


def _run_all(subject, kind='rest'):
    mne.utils.set_log_level('warning')
    print(subject)
    error = 'None'
    result = dict()
    try:
        result = _compute_rest_psd(subject, kind)
    except Exception as err:
        error = repr(err)
        print(error)

    out = dict(subject=subject, kind=kind, error=error)
    out.update(result)
    return out


out = Parallel(n_jobs=40)(
    delayed(_run_all)(subject=subject)
    for subject in subjects)

out_df = pd.DataFrame(out)
out_df.to_csv(
    op.join(
        cfg.derivative_path,
        'log_compute_rest_power_spectra.csv'))
