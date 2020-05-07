# Author: Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import glob
import os.path as op

import numpy as np
import mne
import pandas as pd

import config as cfg

fnames = glob.glob(
    op.join(cfg.derivative_path,
            "*/passive_stim_sensors-ave.fif"))


def _rms_lat(evoked):
    return evoked.times[
        np.sqrt(np.mean(evoked.data ** 2, axis=0)).argmax()]

peaks = list()
for fname in fnames:
    subject = fname.split('/')[-2]
    evokeds = mne.read_evokeds(fname)
    aud = mne.combine_evoked(evokeds[:3], weights='equal')
    vis = evokeds[3]
    peaks.append(
        {"subject": subject,
         "vis": _rms_lat(vis.pick_types(meg='grad')),
         "aud": _rms_lat(aud.pick_types(meg='grad'))})

peaks = pd.DataFrame(peaks)
peaks.to_csv(
    op.join(cfg.derivative_path, 'evoked_peaks.csv'))


# task

fnames = glob.glob(
    op.join(cfg.derivative_path,
            "*/task_stim_sensors-ave.fif"))


peaks_audvis = list()
for fname in fnames:
    subject = fname.split('/')[-2]
    evokeds = mne.read_evokeds(fname)
    evoked = mne.combine_evoked(evokeds, weights='equal')
    peaks_audvis.append(
        {"subject": subject,
         "audvis": _rms_lat(evoked.pick_types(meg='grad'))})
peaks_audvis = pd.DataFrame(peaks_audvis)

peaks_audvis.to_csv(
    op.join(cfg.derivative_path, 'evoked_peaks_task_audvis.csv'))
