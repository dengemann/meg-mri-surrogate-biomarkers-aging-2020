# Author: Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import os
import os.path as op
import random

import numpy as np

from sklearn.covariance import oas
from scipy.stats import pearsonr

import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs

from joblib import Parallel, delayed
from autoreject import get_rejection_threshold

import config as cfg
import library as lib


def _get_subjects(trans_set, shuffle=True):
    trans = 'trans-%s' % trans_set
    found = os.listdir(op.join(cfg.derivative_path, trans))
    if shuffle:
        random.seed(42)
        random.shuffle(found)
    if trans_set == 'halifax':
        subjects = [sub[4:4 + 8] for sub in found]
    elif trans_set == 'krieger':
        subjects = ['CC' + sub[:6] for sub in found]
    print("found", len(subjects), "coregistrations")
    return subjects, [op.join(cfg.derivative_path, trans, ff) for ff in found]


# subjects = lib.utils.get_subjects(cfg.camcan_meg_raw_path)

subjects, trans = _get_subjects(trans_set='krieger')
subject2, trans2 = _get_subjects(trans_set='halifax')
for ii in range(len(subject2)):
    if subject2[ii] not in subjects:
        subjects.append(subject2[ii])
        trans.append(trans2[ii])

trans_map = dict(zip(subjects, trans))

N_JOBS = 40
# subjects = subjects[:40]
# subjects = subjects[:1]
# subjects = subjects[:1]

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
    # first 100 channels omit the 0.
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


def _run_maxfilter(raw, subject, kind, coord_frame="head"):

    bads = _parse_bads(subject, kind)

    raw.info['bads'] = bads

    raw = lib.preprocessing.run_maxfilter(raw, coord_frame=coord_frame)
    return raw


def _compute_add_ssp_exg(raw):
    reject_eog, reject_ecg = _get_global_reject_ssp(raw)

    proj_eog = mne.preprocessing.compute_proj_eog(
        raw, average=True, reject=reject_eog, n_mag=1, n_grad=1, n_eeg=1)

    proj_ecg = mne.preprocessing.compute_proj_ecg(
        raw, average=True, reject=reject_ecg, n_mag=1, n_grad=1, n_eeg=1)

    raw.add_proj(proj_eog[0])
    raw.add_proj(proj_ecg[0])


def _get_global_reject_epochs(raw, decim):
    duration = 3.
    events = mne.make_fixed_length_events(
        raw, id=3000, start=0, duration=duration)

    epochs = mne.Epochs(
        raw, events, event_id=3000, tmin=0, tmax=duration, proj=False,
        baseline=None, reject=None)
    epochs.apply_proj()
    epochs.load_data()
    epochs.pick_types(meg=True)
    reject = get_rejection_threshold(epochs, decim=decim)
    return reject


def _apply_inverse_cov(
        cov, info, nave, inverse_operator, lambda2=1. / 9., method="dSPM",
        pick_ori=None, prepared=False, label=None,
        method_params=None, return_residual=False, verbose=None,
        log=True):
    """Apply inverse operator to evoked data HACKED
    """
    from mne.minimum_norm.inverse import _check_reference
    from mne.minimum_norm.inverse import _check_ori
    from mne.minimum_norm.inverse import _check_ch_names
    from mne.minimum_norm.inverse import _check_or_prepare
    from mne.minimum_norm.inverse import _check_ori
    from mne.minimum_norm.inverse import _pick_channels_inverse_operator
    from mne.minimum_norm.inverse import _assemble_kernel
    from mne.minimum_norm.inverse import _subject_from_inverse
    from mne.minimum_norm.inverse import _get_src_type
    from mne.minimum_norm.inverse import combine_xyz
    from mne.minimum_norm.inverse import _make_stc
    from mne.utils import _check_option
    from mne.utils import logger
    from mne.io.constants import FIFF
    from collections import namedtuple

    INVERSE_METHODS = ['MNE', 'dSPM', 'sLORETA', 'eLORETA']

    fake_evoked = namedtuple('fake', 'info')(info=info)

    _check_reference(fake_evoked, inverse_operator['info']['ch_names'])
    _check_option('method', method, INVERSE_METHODS)
    if method == 'eLORETA' and return_residual:
        raise ValueError('eLORETA does not currently support return_residual')
    _check_ori(pick_ori, inverse_operator['source_ori'])
    #
    #   Set up the inverse according to the parameters
    #

    _check_ch_names(inverse_operator, info)

    inv = _check_or_prepare(inverse_operator, nave, lambda2, method,
                            method_params, prepared)

    #
    #   Pick the correct channels from the data
    #
    sel = _pick_channels_inverse_operator(cov['names'], inv)
    logger.info('Applying inverse operator to cov...')
    logger.info('    Picked %d channels from the data' % len(sel))
    logger.info('    Computing inverse...')

    K, noise_norm, vertno, source_nn = _assemble_kernel(inv, label, method,
                                                        pick_ori)
    
    # apply imaging kernel
    sol = np.einsum('ij,ij->i', K, (cov.data[sel] @ K.T).T)[:, None]

    is_free_ori = (inverse_operator['source_ori'] ==
                   FIFF.FIFFV_MNE_FREE_ORI and pick_ori != 'normal')

    if is_free_ori and pick_ori != 'vector':
        logger.info('    Combining the current components...')
        sol = combine_xyz(sol)

    if noise_norm is not None:
        logger.info('    %s...' % (method,))
        if is_free_ori and pick_ori == 'vector':
            noise_norm = noise_norm.repeat(3, axis=0)
        sol *= noise_norm

    tstep = 1.0 / info['sfreq']
    tmin = 0.0
    subject = _subject_from_inverse(inverse_operator)

    src_type = _get_src_type(inverse_operator['src'], vertno)
    if log:
        sol = np.log10(sol, out=sol)

    stc = _make_stc(sol, vertno, tmin=tmin, tstep=tstep, subject=subject,
                    vector=(pick_ori == 'vector'), source_nn=source_nn,
                    src_type=src_type)
    logger.info('[done]')

    return stc


def _compute_mne_power(subject, kind, freqs):

    ###########################################################################
    # Compute source space
    # -------------------
    src = mne.setup_source_space(subject, spacing='oct6', add_dist=False,
                                 subjects_dir=cfg.mne_camcan_freesurfer_path)
    trans = trans_map[subject]
    bem = cfg.mne_camcan_freesurfer_path + \
        "/%s/bem/%s-meg-bem.fif" % (subject, subject)

    ###########################################################################
    # Compute handle MEG data
    # -----------------------

    fname = op.join(
        cfg.camcan_meg_raw_path,
        subject, kind, '%s_raw.fif' % kind)

    raw = mne.io.read_raw_fif(fname)
    mne.channels.fix_mag_coil_types(raw.info)
    if DEBUG:
        # raw.crop(0, 180)
        raw.crop(0, 120)
    else:
        raw.crop(0, 300)

    raw = _run_maxfilter(raw, subject, kind)
    _compute_add_ssp_exg(raw)

    # get empty room
    fname_er = op.join(
        cfg.camcan_meg_path,
        "emptyroom",
        subject,
        "emptyroom_%s.fif" % subject)

    raw_er = mne.io.read_raw_fif(fname_er)
    mne.channels.fix_mag_coil_types(raw.info)

    raw_er = _run_maxfilter(raw_er, subject, kind, coord_frame="meg")
    raw_er.info["projs"] += raw.info["projs"]

    cov = mne.compute_raw_covariance(raw_er, method='oas')
    # compute before band-pass of interest

    event_length = 5.
    event_overlap = 0.
    raw_length = raw.times[-1]
    events = mne.make_fixed_length_events(
        raw,
        duration=event_length, start=0, stop=raw_length - event_length)

    #######################################################################
    # Compute the forward and inverse
    # -------------------------------

    info = mne.Epochs(raw, events=events, tmin=0, tmax=event_length,
                      baseline=None, reject=None, preload=False,
                      decim=10).info
    fwd = mne.make_forward_solution(info, trans, src, bem)
    inv = make_inverse_operator(info, fwd, cov)
    del fwd

    #######################################################################
    # Compute label time series and do envelope correlation
    # -----------------------------------------------------
    mne_subjects_dir = "/storage/inria/agramfor/MNE-sample-data/subjects"
    labels = mne.read_labels_from_annot('fsaverage', 'aparc_sub',
                                        subjects_dir=mne_subjects_dir)
    labels = mne.morph_labels(
        labels, subject_from='fsaverage', subject_to=subject,
        subjects_dir=cfg.mne_camcan_freesurfer_path)
    labels = [ll for ll in labels if 'unknown' not in ll.name]

    results = dict()
    for fmin, fmax, band in freqs:
        print(f"computing {subject}: {fmin} - {fmax} Hz")
        this_raw = raw.copy()
        this_raw.filter(fmin, fmax, n_jobs=1)
        reject = _get_global_reject_epochs(this_raw, decim=5)
        epochs = mne.Epochs(this_raw, events=events, tmin=0, tmax=event_length,
                            baseline=None, reject=reject, preload=True,
                            decim=5)
        if DEBUG:
            epochs = epochs[:3]

        # MNE cov mapping
        data_cov = mne.compute_covariance(epochs, method='oas')
        stc = _apply_inverse_cov(
            cov=data_cov, info=epochs.info, nave=1,
            inverse_operator=inv, lambda2=1. / 9.,
            pick_ori='normal', method='MNE', log=False)
        # assert np.all(stc.data < 0)

        label_power = mne.extract_label_time_course(
            stc, labels, inv['src'], mode="mean")  # XXX signal should be positive

        # ts source covariance
        stcs = apply_inverse_epochs(
            epochs, inv, lambda2=1. / 9.,
            pick_ori='normal',
            method='MNE',
            return_generator=True)

        label_ts = np.concatenate(mne.extract_label_time_course(
            stcs, labels, inv['src'], mode="pca_flip",
            return_generator=False), axis=-1)

        label_cov, _ = oas(label_ts.T, assume_centered=True)

        if DEBUG:
            print(
                pearsonr(
                    np.log10(np.diag(label_cov)).ravel(),
                    np.log10(label_power.ravel())))

        result = {'cov': label_cov[np.triu_indices(len(label_cov))],
                  'power': label_power, 'subject': subject,
                  'fmin': fmin, 'fmax': fmax, "band": band,
                  'label_names': [ll.name for ll in labels]}
        results[band] = result

        if False:
            out_fname = op.join(
                cfg.derivative_path,
                f'{subject + ("-debug" if DEBUG else "")}_'
                f'cov_mne_{band}.h5')

            mne.externals.h5io.write_hdf5(
                out_fname, result, overwrite=True)
    return results


def _run_all(subject, freqs, kind='rest'):
    mne.utils.set_log_level('warning')
    # mne.utils.set_log_level('info')
    print(subject)
    error = 'None'
    result = dict()
    if not DEBUG:
        try:
            out = _compute_mne_power(subject, kind, freqs)
        except Exception as err:
            error = repr(err)
            print(error)
    else:
        out = _compute_mne_power(subject, kind, freqs)

    if error != 'None':
        out = {band: None for _, _, band in freqs}
        out['error'] = error

    return out

freqs = [(0.1, 1.5, "low"),
         (1.5, 4.0, "delta"),
         (4.0, 8.0, "theta"),
         (8.0, 15.0, "alpha"),
         (15.0, 26.0, "beta_low"),
         (26.0, 35.0, "beta_high"),
         (35.0, 50.0, "gamma_lo"),
         (50.0, 74.0, "gamma_mid"),
         (76.0, 120.0, "gamma_high")]

DEBUG = False
if DEBUG:
    N_JOBS = 1
    subjects = subjects[:1]
    freqs = [freqs[4]]

out = Parallel(n_jobs=N_JOBS)(
    delayed(_run_all)(subject=subject, freqs=freqs, kind='rest')
    for subject in subjects)

out = {sub: dd for sub, dd in zip(subjects, out) if 'error' not in dd}

mne.externals.h5io.write_hdf5(
    op.join(cfg.derivative_path, 'all_mne_source_power.h5'), out,
    overwrite=True)
