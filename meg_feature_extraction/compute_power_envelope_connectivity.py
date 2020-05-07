# Author: Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import  glob
import os
import os.path as op
import random

import numpy as np

import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs

from joblib import Parallel, delayed
from autoreject import get_rejection_threshold

import config as cfg
import library as lib


import numpy as np
from sklearn.covariance import oas

from mne.filter import next_fast_len
from mne.source_estimate import _BaseSourceEstimate
from mne.utils import verbose, _check_combine


@verbose
def envelope_correlation(data, combine='mean', orthogonalize="pairwise",
                         verbose=None):
    """Compute the envelope correlation.
    Parameters
    ----------
    data : array-like, shape=(n_epochs, n_signals, n_times) | generator
        The data from which to compute connectivity.
        The array-like object can also be a list/generator of array,
        each with shape (n_signals, n_times), or a :class:`~mne.SourceEstimate`
        object (and ``stc.data`` will be used). If it's float data,
        the Hilbert transform will be applied; if it's complex data,
        it's assumed the Hilbert has already been applied.
    combine : 'mean' | callable | None
        How to combine correlation estimates across epochs.
        Default is 'mean'. Can be None to return without combining.
        If callable, it must accept one positional input.
        For example::
            combine = lambda data: np.median(data, axis=0)
    orthoganalize : 'pairwise' | False
        Whether to orthogonalize with the pairwise method or not.
        Defaults to 'pairwise'. Note that when False,
        the correlation matrix will not be returned with
        absolute values.

    %(verbose)s
    Returns
    -------
    corr : ndarray, shape ([n_epochs, ]n_nodes, n_nodes)
        The pairwise orthogonal envelope correlations.
        This matrix is symmetric. If combine is None, the array
        with have three dimensions, the first of which is ``n_epochs``.
    Notes
    -----
    This function computes the power envelope correlation between
    orthogonalized signals [1]_ [2]_.
    References
    ----------
    .. [1] Hipp JF, Hawellek DJ, Corbetta M, Siegel M, Engel AK (2012)
           Large-scale cortical correlation structure of spontaneous
           oscillatory activity. Nature Neuroscience 15:884–890
    .. [2] Khan S et al. (2018). Maturation trajectories of cortical
           resting-state networks depend on the mediating frequency band.
           Neuroimage 174:57–68
    """
    from scipy.signal import hilbert
    n_nodes = None
    if combine is not None:
        fun = _check_combine(combine, valid=('mean',))
    else:  # None
        fun = np.array

    corrs = list()
    # Note: This is embarassingly parallel, but the overhead of sending
    # the data to different workers is roughly the same as the gain of
    # using multiple CPUs. And we require too much GIL for prefer='threading'
    # to help.
    for ei, epoch_data in enumerate(data):
        if isinstance(epoch_data, _BaseSourceEstimate):
            epoch_data = epoch_data.data
        if epoch_data.ndim != 2:
            raise ValueError('Each entry in data must be 2D, got shape %s'
                             % (epoch_data.shape,))
        n_nodes, n_times = epoch_data.shape
        if ei > 0 and n_nodes != corrs[0].shape[0]:
            raise ValueError('n_nodes mismatch between data[0] and data[%d], '
                             'got %s and %s'
                             % (ei, n_nodes, corrs[0].shape[0]))
        # Get the complex envelope (allowing complex inputs allows people
        # to do raw.apply_hilbert if they want)
        if epoch_data.dtype in (np.float32, np.float64):
            n_fft = next_fast_len(n_times)
            epoch_data = hilbert(epoch_data, N=n_fft, axis=-1)[..., :n_times]
        
        if orthogonalize is False:
            corrs.append(np.corrcoef(np.abs(epoch_data)))
            continue

        if epoch_data.dtype not in (np.complex64, np.complex128):
            raise ValueError('data.dtype must be float or complex, got %s'
                             % (epoch_data.dtype,))
        data_mag = np.abs(epoch_data)
        data_conj_scaled = epoch_data.conj()
        data_conj_scaled /= data_mag
        # subtract means
        data_mag_nomean = data_mag - np.mean(data_mag, axis=-1, keepdims=True)
        # compute variances using linalg.norm (square, sum, sqrt) since mean=0
        data_mag_std = np.linalg.norm(data_mag_nomean, axis=-1)
        data_mag_std[data_mag_std == 0] = 1
        corr = np.empty((n_nodes, n_nodes))
        for li, label_data in enumerate(epoch_data):
            label_data_orth = (label_data * data_conj_scaled).imag
            label_data_orth -= np.mean(label_data_orth, axis=-1, keepdims=True)
            label_data_orth_std = np.linalg.norm(label_data_orth, axis=-1)
            label_data_orth_std[label_data_orth_std == 0] = 1
            # correlation is dot product divided by variances
            corr[li] = np.dot(label_data_orth, data_mag_nomean[li])
            corr[li] /= data_mag_std[li]
            corr[li] /= label_data_orth_std
        # Make it symmetric (it isn't at this point)
        corr = np.abs(corr)
        corrs.append((corr.T + corr) / 2.)
        del corr

    corr = fun(corrs)
    return corr


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


def _compute_power_envelopes(subject, kind, freqs):

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

        this_raw.apply_hilbert(envelope=False)

        epochs = mne.Epochs(this_raw, events=events, tmin=0, tmax=event_length,
                            baseline=None, reject=reject, preload=True,
                            decim=5)
        if DEBUG:
            epochs = epochs[:3]

        result = {'subject': subject,
                  'fmin': fmin, 'fmax': fmax,
                  'band': band,
                  'label_names': [ll.name for ll in labels]}

        stcs = apply_inverse_epochs(epochs, inv, lambda2=1. / 9.,
                                    pick_ori='normal',
                                    method='MNE',
                                    return_generator=True)

        label_ts = np.concatenate(mne.extract_label_time_course(
            stcs, labels, inv['src'], mode="pca_flip",
            return_generator=False), axis=-1)

        result['cov'], _ = oas(
            np.abs(label_ts).T, assume_centered=False)

        for orth in ("pairwise", False):
            corr = envelope_correlation(
                label_ts[np.newaxis], combine="mean", orthogonalize=orth)
            result[f"corr{'_orth' if orth else ''}"] = corr[np.triu_indices(len(corr))]

        results[band] = result

        if False:  # failsafe mode with intermediate steps written out
            out_fname = op.join(
                cfg.derivative_path,
                f'{subject + ("-debug" if DEBUG else "")}_'
                f'power_envelopes_{band}.h5')

            mne.externals.h5io.write_hdf5(
                out_fname, result, overwrite=True)
    return results


def _run_all(subject, freqs, kind='rest'):
    mne.utils.set_log_level('warning')
    # mne.utils.set_log_level('info')
    print(subject)
    error = 'None'
    if not DEBUG:
        try:
            out = _compute_power_envelopes(subject, kind, freqs)
        except Exception as err:
            error = repr(err)
            print(error)
    else:
        out = _compute_power_envelopes(subject, kind, freqs)

    if error != 'None':
        out = {band: None for _, _, band in freqs}
        out['error'] = error
    return out


N_JOBS = 40
DEBUG = False
freqs = [(0.1, 1.5, "low"),
         (1.5, 4.0, "delta"),
         (4.0, 8.0, "theta"),
         (8.0, 15.0, "alpha"),
         (15.0, 26.0, "beta_low"),
         (26.0, 35.0, "beta_high"),
         (35.0, 50.0, "gamma_lo"),
         (50.0, 74.0, "gamma_mid"),
         (76.0, 120.0, "gamma_high")]

if DEBUG:
    N_JOBS = 1
    subjects = subjects[:1]
    freqs = freqs[:1]

out = Parallel(n_jobs=N_JOBS)(
    delayed(_run_all)(subject=subject, freqs=freqs, kind='rest')
    for subject in subjects)

out = {sub: dd for sub, dd in zip(subjects, out) if 'error' not in dd}

mne.externals.h5io.write_hdf5(
    op.join(cfg.derivative_path, 'all_power_envelopes.h5'), out,
    overwrite=True)
