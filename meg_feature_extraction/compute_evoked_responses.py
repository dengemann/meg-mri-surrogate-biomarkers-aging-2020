# Author: Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import os
import os.path as op
from collections import Counter

import numpy as np
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

kinds = ['passive', 'task']

task_info = {
    'passive': {
        'event_id': [{
            'Aud300Hz': 6, 'Aud600Hz': 7, 'Aud1200Hz': 8, 'Vis': 9}],
        'epochs_params': [{
            'tmin': -0.2, 'tmax': 0.7, 'baseline': (-.2, None),
            'decim': 8}],
        'lock': ['stim']
    },
    'task': {
        'event_id': [
            {'AudVis300Hz': 1, 'AudVis600Hz': 2, 'AudVis1200Hz': 3},
            {'resp': 8192}],
        'epochs_params': [
            {'tmin': -0.2, 'tmax': 0.7, 'baseline': (-.2, None),
             'decim': 8},
            {'tmin': -0.5, 'tmax': 1,
             'baseline': (.8, 1.0), 'decim': 8}],
        'lock': ['stim', 'resp'],
    }
}


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
    if 'eog' in raw:
        eog_epochs = mne.preprocessing.create_eog_epochs(raw)
    else:
        eog_epochs = []
    if len(eog_epochs) >= 5:
        reject_eog = get_rejection_threshold(eog_epochs, decim=8)
        del reject_eog['eog']  # we don't want to reject eog based on eog
    else:
        reject_eog = None

    ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)
    # we will always have an ECG as long as there are magnetometers
    if len(ecg_epochs) >= 5:
        reject_ecg = get_rejection_threshold(ecg_epochs, decim=8)
        # here we want the eog
    else:
        reject_ecg = None

    if reject_eog is None and reject_ecg is not None:
        reject_eog = {k: v for k, v in reject_ecg.items() if k != 'eog'}
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


def _get_global_reject_epochs(raw, events, event_id, epochs_params):
    epochs = mne.Epochs(
        raw, events, event_id=event_id, proj=False,
        **epochs_params)
    epochs.load_data()
    epochs.pick_types(meg=True)
    epochs.apply_proj()
    reject = get_rejection_threshold(epochs, decim=1)
    return reject


def _compute_evoked(subject, kind):

    fname = op.join(
        cfg.camcan_meg_raw_path,
        subject, kind, '%s_raw.fif' % kind)

    raw = mne.io.read_raw_fif(fname)
    mne.channels.fix_mag_coil_types(raw.info)
    raw = _run_maxfilter(raw, subject, kind)
    if DEBUG:
        raw.crop(0, 60)
    raw.filter(1, 30)
    _compute_add_ssp_exg(raw)

    out = {}
    for ii, event_id in enumerate(task_info[kind]['event_id']):
        epochs_params = task_info[kind]['epochs_params'][ii]
        lock = task_info[kind]['lock'][ii]
        events = mne.find_events(
            raw, uint_cast=True, min_duration=2. / raw.info['sfreq'])

        if kind == 'task' and lock == 'resp':
            event_map = np.array(
                [(k, v) for k, v in Counter(events[:, 2]).items()])
            button_press = event_map[:, 0][np.argmax(event_map[:, 1])]
            if event_map[:, 1][np.argmax(event_map[:, 1])] >= 50:
                events[events[:, 2] == button_press, 2] = 8192
            else:
                raise RuntimeError('Could not guess button press')

        reject = _get_global_reject_epochs(
            raw,
            events=events,
            event_id=event_id,
            epochs_params=epochs_params)

        epochs = mne.Epochs(
            raw, events=events, event_id=event_id, reject=reject,
            preload=True,
            **epochs_params)

        # noise_cov = mne.compute_covariance(
        #     epochs, tmax=0, method='oas')

        evokeds = list()
        for kk in event_id:
            evoked = epochs[kk].average()
            evoked.comment = kk
            evokeds.append(evoked)

        out_path = op.join(
            cfg.derivative_path, subject)

        if not op.exists(out_path):
            os.makedirs(out_path)

        out_fname = op.join(out_path, '%s_%s_sensors-ave.fif' % (
            kind, lock))

        mne.write_evokeds(out_fname, evokeds)
        out.update({lock: (kind, epochs.average().nave)})

    return out


def _run_all(subject, kind):
    mne.utils.set_log_level('warning')
    print(subject)
    error = 'None'
    result = dict()
    if not DEBUG:
        try:
            result = _compute_evoked(subject, kind)
        except Exception as err:
            error = repr(err)
            print(error)
    else:
        result = _compute_evoked(subject, kind)
    out = dict(subject=subject, kind=kind, error=error)
    out.update(result)
    return out

DEBUG = False
if DEBUG:
    subjects = subjects[:1]

out_passive = Parallel(n_jobs=40)(
    delayed(_run_all)(subject=subject, kind='passive')
    for subject in subjects)

out_df_passive = pd.DataFrame(out_passive)
out_df_passive.to_csv(
    op.join(
        cfg.derivative_path,
        'log_compute_evoked_passive.csv'))


out_task = Parallel(n_jobs=40)(
    delayed(_run_all)(subject=subject, kind='task')
    for subject in subjects)

out_df_task = pd.DataFrame(out_task)
out_df_task.to_csv(
    op.join(
        cfg.derivative_path,
        'log_compute_evoked_task.csv'))
