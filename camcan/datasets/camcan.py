"""Loader for the Cam-CAN data set."""

import re
import glob
import warnings
import itertools
import os

from os.path import join, isdir, relpath, dirname, isfile

import numpy as np
import pandas as pd
import json
import joblib

from sklearn.externals import six
from sklearn.datasets.base import Bunch

from nilearn import input_data

# root path
CAMCAN_DRAGO_STORE = '/storage/data/camcan/camcan_preproc'
# time series rest path
CAMCAN_DRAGO_STORE_TIMESERIES_REST = '/storage/data/camcan/camcan_timeseries'
# connectivity rest path
CAMCAN_DRAGO_STORE_CONNECTIVITY_REST = '/storage/data/camcan/'\
                                       'camcan_connectivity'
# contrast maps path
CAMCAN_DRAGO_STORE_CONTRASTS = '/storage/data/camcan/camcan_smt_maps'

# scores path
CAMCAN_DRAGO_STORE_SCORES = '/storage/data/camcan/cc700-scored/'\
                            'participant_data.csv'

# behavioural path
CAMCAN_DRAGO_STORE_BEHAVIOURAL_EXPERIMENT = \
    "/storage/data/camcan/cc700-scored/behavioural_features.json"

# path for anatomical and functional images - BIDS format
FUNCTIONAL_PATH = 'func'
ANATOMICAL_PATH = 'anat'

# pattern of the different files of interests
FUNC_PATTERN = 'wrsub-*.nii.gz'
MVT_PATTERN = 'rp*.txt'
ANAT_PATTERN = 'wsub-*.nii.gz'
TISSUES_PATTERN = ['mwc1sub-*.nii.gz', 'mwc2sub-*.nii.gz', 'mwc3sub-*.nii.gz']

# column to select from the patients information
COLUMN_SELECT_PATIENTS_INFO = ('Observations', 'age', 'hand', 'gender_text')


def _patients_id_from_csv(csv_file):
    """Private function to extract the patient IDs which need to be excluded.

    Parameters
    ----------
    csv_file : str,
        Filename of the CSV file from which we will extract the information.

    Returns
    -------
    patients_excluded_ : tuple of str,
        The validated list of patient IDs to exclude.

    """
    if isfile(csv_file):
        df = pd.read_csv(csv_file)
        # go line by line and check the validity for each subject
        return tuple([df.iloc[i, 0] for i in range(df.shape[0])
                      if not np.all(df.iloc[i, 1:])])
    else:
        raise ValueError('{}: File not found.'.format(csv_file))


def _validate_patients_excluded(patients_excluded):
    """Private function to validate ``patients_excluded``.

    Parameters
    ----------
    patients_excluded : str, tuple of str or None, optional (default=None)
        - If a string, corresponds to the path of a csv file.
        - If a tuple of strings, contains the ID of the patient to be
        excluded. The string provided should follow the BIDS standard (e.g.,
        'sub-******').

    Returns
    -------
    patients_excluded_ : tuple of str,
        The validated list of patient IDs to exclude.

    """
    if patients_excluded is None:
        patients_excluded_ = tuple([])
    elif isinstance(patients_excluded, tuple):
        # pattern imposed by BIDS standard
        pattern = re.compile('sub-*')
        # check that all string in the tuple follow the correct pattern
        if all(map(lambda patient_id:
                   True if pattern.match(patient_id) is not None else False,
               patients_excluded)):
            patients_excluded_ = patients_excluded
        else:
            raise ValueError("All patient IDs to be excluded should follow"
                             " the pattern 'sub-'.")
    elif isinstance(patients_excluded, six.string_types):
        if patients_excluded.endswith('.csv'):
            patients_excluded_ = _patients_id_from_csv(patients_excluded)
        else:
            raise ValueError('If a string is provided, a csv file is'
                             ' required.')
    else:
        raise ValueError("'patients_excluded' should be a tuple. Got {}"
                         " instead.".format(type(patients_excluded)))

    return patients_excluded_


def _exclude_patients(data_dir, patients_excluded):
    """Private function to exclude patients.

    Parameters
    ----------
    data_dir : str,
        The path to be investigated.

    patients_excluded : tuple of str,
        The tuple containing all the patients IDs.

    """
    subjects_dir = sorted(glob.glob(join(data_dir, 'sub-*')))
    dir_idx_kept = [dir_idx for dir_idx in range(len(subjects_dir))
                    if relpath(subjects_dir[dir_idx], data_dir)
                    not in patients_excluded]
    return [subjects_dir[i] for i in dir_idx_kept]


def _check_scores(patients_info_csv, subject_ids):
    """Private function to return scores."""
    if patients_info_csv is None:
        scores = Bunch(**{'age': [None] * len(subject_ids),
                          'hand': [None] * len(subject_ids),
                          'gender_text': [None] * len(subject_ids)})
    else:
        scores = _load_camcan_scores(patients_info_csv, subject_ids)
    return scores


def _load_camcan_scores(filename_csv, subjects_selected):
    """Load the scores from the Cam-CAN data set.

    Parameters
    ----------
    filename_csv : str,
        Path to the csv file containing the participants information.

    subjects_selected: list of str,
        A list of strings, contains the ID of the patient to be selected. The
        string provided should follow the BIDS standard (e.g., 'sub-******').

    Returns
    -------
    data : Bunch,
        Dictionary-like object. The interesting attributes are:

        - 'age', the age of the patient;
        - 'hand', handedness of the patient;
        - 'gender_text', gender of the patient.

    """
    if not isfile(filename_csv):
        raise ValueError('The file {} does not exist.'.format(filename_csv))

    if not filename_csv.endswith('.csv'):
        raise ValueError('The file {} is not a CSV file.'.format(filename_csv))

    patients_info = pd.read_csv(filename_csv,
                                usecols=COLUMN_SELECT_PATIENTS_INFO)

    # the id in the CSV is missing 'sub-'
    patients_info['Observations'] = 'sub-' + patients_info['Observations']
    # filter the IDs to be kept and sort just in case
    patients_info = (patients_info.set_index('Observations')
                                  .loc[subjects_selected])

    return Bunch(**patients_info.to_dict('list'))


def load_camcan_rest(data_dir=CAMCAN_DRAGO_STORE,
                     patients_info_csv=None,
                     patients_excluded=None):
    """Path loader for the Cam-CAN resting-state fMRI data.

    This loader returns a Bunch object containing the paths to the data of
    interests. The data which can be loaded are:

    - Functional images;
    - Anatomical images;
    - Motion correction;
    - Tissues segmentation;
    - Patient ID;
    - Scores.

    See the description of the data to get all the information.

    Parameters
    ----------
    data_dir : str,
        Root directory containing the root data.

    patients_info_csv : str or None, (default=None)
        Path to the CSV file containing the patients information.
        If None, data.scores will be a list of None.

    patients_excluded : str, tuple of str or None, optional (default=None)
        - If a string, corresponds to the path of a csv file. The first line
        of this csv file should contain the name of each column.
        - If a tuple of strings, contains the ID of the patient to be
        excluded. The string provided should follow the BIDS standard (e.g.,
        'sub-******').

    Returns
    -------
    data : Bunch,
        Dictionary-like object. The interesting attributes are:

        - 'func', the path to the functional images;
        - 'anat', the path to the anatomical images;
        - 'motion', the path of the file containing the motion parameters
        from the rigid registration performed on the functional images;
        - 'tissues', the path to the images containing the segmentation of the
        brain tissues from the anatomical images;
        - 'subject_id', the ID of the patient;
        - 'scores', a dictionary containing the different scores: age,
        handedness, and gender;
        - 'DESCR', the description of the full dataset.

    """
    patients_excluded_ = _validate_patients_excluded(patients_excluded)

    if not isdir(data_dir):
        raise ValueError("The directory '{}' does not exist.".format(data_dir))

    subjects_dir = _exclude_patients(data_dir, patients_excluded_)

    module_path = dirname(__file__)
    with open(join(module_path, 'descr', 'camcan.rst')) as rst_file:
        fdescr = rst_file.read()

    dataset = {'func': [],
               'motion': [],
               'anat': [],
               'tissues': [],
               'subject_id': [],
               'DESCR': fdescr}

    for subject_dir in subjects_dir:
        dataset['subject_id'].append(relpath(subject_dir, data_dir))
        # Discover one after the other:
        # - functional images;
        # - motion parameters;
        # - anatomical images;
        # - tissues segmented.
        for p, f, k in zip([FUNCTIONAL_PATH] * 2 + [ANATOMICAL_PATH] * 4,
                           [FUNC_PATTERN, MVT_PATTERN, ANAT_PATTERN] +
                           TISSUES_PATTERN,
                           ['func', 'motion', 'anat'] + ['tissues'] * 3):

            nifti_path = glob.glob(join(subject_dir, p, f))
            if not nifti_path:
                warnings.warn("No file match the regular expression {} for"
                              " the subject ID {}".format(
                                  join(p, f), relpath(subject_dir, data_dir)))
                dataset[k].append(None)
            else:
                dataset[k].append(nifti_path[0])

    scores = _check_scores(patients_info_csv, dataset['subject_id'])
    dataset['scores'] = scores

    return Bunch(**dataset)


def load_camcan_timeseries_rest(data_dir=CAMCAN_DRAGO_STORE_TIMESERIES_REST,
                                patients_info_csv=None,
                                atlas='msdl',
                                patients_excluded=None):
    """Load the Cam-CAN time series extracted from resting fMRI.

    Parameters
    ----------
    data_dir : str,
        Root directory containing the root data.

    patients_info_csv : str or None, (default=None)
        Path to the CSV file containing the patients information.
        If None, data.scores will be a list of None.

    atlas : str, (default='msdl')
        The atlas to used during the extraction of the time series. Choices
        are: 'msdl' (default), 'basc064', and 'basc122'.

    patients_excluded : str, tuple of str or None, optional (default=None)
        - If a string, corresponds to the path of a csv file. The first line
        of this csv file should contain the name of each column.
        - If a tuple of strings, contains the ID of the patient to be
        excluded. The string provided should follow the BIDS standard (e.g.,
        'sub-******').

    Returns
    -------
    data : Bunch,
        Dictionary-like object. The interesting attributes are:

        - 'timeseries', the time series for each patient.
        - 'subject_id', the ID of the patient;
        - 'scores', a dictionary containing the different scores: age,
        handedness, and gender.

    """
    patients_excluded_ = _validate_patients_excluded(patients_excluded)

    if not isdir(data_dir):
        raise ValueError("The directory '{}' does not exist.".format(data_dir))

    subjects_dir = _exclude_patients(data_dir, patients_excluded_)

    dataset = {'timeseries': [],
               'subject_id': []}

    for subject_dir in subjects_dir:
        subject_id = relpath(subject_dir, data_dir)
        dataset['subject_id'].append(subject_id)
        filename = join(subject_dir, atlas,
                        subject_id + '_task-Rest_confounds.pkl')
        dataset['timeseries'].append(joblib.load(filename))

    scores = _check_scores(patients_info_csv, dataset['subject_id'])
    dataset['scores'] = scores

    return Bunch(**dataset)


def load_camcan_connectivity_rest(data_dir=CAMCAN_DRAGO_STORE_TIMESERIES_REST,
                                  patients_info_csv=None,
                                  atlas='msdl',
                                  kind='tangent',
                                  patients_excluded=None):
    """Load the Cam-CAN time series extracted from resting fMRI.

    Parameters
    ----------
    data_dir : str,
        Root directory containing the root data.

    patients_info_csv : str or None, (default=None)
        Path to the CSV file containing the patients information.
        If None, data.scores will be a list of None.

    atlas : str, (default='msdl')
        The atlas to used during the extraction of the time series. Choices
        are: 'msdl' (default), 'basc064', and 'basc122'.

    kind : str, (default='tangent)
        The kind of connectivity matrix.

    patients_excluded : str, tuple of str or None, optional (default=None)
        - If a string, corresponds to the path of a csv file. The first line
        of this csv file should contain the name of each column.
        - If a tuple of strings, contains the ID of the patient to be
        excluded. The string provided should follow the BIDS standard (e.g.,
        'sub-******').

    Returns
    -------
    data : Bunch,
        Dictionary-like object. The interesting attributes are:

        - 'connectivity', the connectivity matrix for each patient.
        - 'subject_id', the ID of the patient;
        - 'scores', a dictionary containing the different scores: age,
        handedness, and gender.

    """
    patients_excluded_ = _validate_patients_excluded(patients_excluded)

    if not isdir(data_dir):
        raise ValueError("The directory '{}' does not exist.".format(data_dir))

    subjects_dir = _exclude_patients(data_dir, patients_excluded_)

    dataset = {'connectivity': [],
               'subject_id': []}

    for subject_dir in subjects_dir:
        subject_id = relpath(subject_dir, data_dir)
        dataset['subject_id'].append(subject_id)
        filename = join(subject_dir, atlas, kind,
                        subject_id + '_task-Rest_confounds.pkl')
        dataset['connectivity'].append(joblib.load(filename))

    scores = _check_scores(patients_info_csv, dataset['subject_id'])
    dataset['scores'] = scores

    return Bunch(**dataset)


def _abs_listdir(dir_name):
    dir_name = os.path.abspath(dir_name)
    for file_name in os.listdir(dir_name):
        yield os.path.join(dir_name, file_name)


def load_camcan_contrast_maps(contrast_name, statistic_type='z_score',
                              data_dir=CAMCAN_DRAGO_STORE_CONTRASTS,
                              patients_excluded=None, mask_file=None):
    """Load contrast maps for Cam-CAN dataset.

    Parameters
    ----------
    contrast_name : str,
       Regex pattern that matches the contrasts you want

    statistic_type : str, (default='z_score')
        The kind of statistical map to load.

    data_dir : str, (default=CAMCAN_DRAGO_STORE_CONTRASTS)
        Root directory containing the root data.

    patients_excluded : str, tuple of str or None, optional (default=None)
        - If a string, corresponds to the path of a csv file. The first line
        of this csv file should contain the name of each column.
        - If a tuple of strings, contains the ID of the patient to be
        excluded. The string provided should follow the BIDS standard (e.g.,
        'sub-******').

    mask_file : str or None (default=None)
        where to find the mask. if none we look for it in data_dir

    Returns
    -------
    data : Bunch,
        Dictionary-like object. The interesting attributes are:

        - 'subject_id', the ID of the patient;
        - 'contrast_map', the path to the map;
        - 'contrast_name', the name of the contrast;
        - 'mask', the path to the mask.

    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            2, 'No such file or directory: {}'.format(data_dir))
    if mask_file is None:
        mask_file = os.path.join(data_dir, 'mask_camcan.nii.gz')
    mask_file = os.path.abspath(mask_file)
    patients_excluded_ = _validate_patients_excluded(patients_excluded)
    dataset = {'subject_id': [], 'contrast_map': [], 'contrast_name': []}
    subject_dirs = _exclude_patients(data_dir, patients_excluded_)
    for contrast_map in itertools.chain(*map(_abs_listdir, subject_dirs)):
        match = re.match(
            r'^.*sub-([0-9a-zA-Z]+)_([0-9a-zA-Z-]+)_(.+)\.nii\.gz$',
            contrast_map)
        subject_id, contrast, stat_type = match.groups()
        if stat_type == statistic_type and re.match(contrast_name, contrast):
            dataset['subject_id'].append(subject_id)
            dataset['contrast_map'].append(contrast_map)
            dataset['contrast_name'].append(contrast)

    return Bunch(mask=mask_file, **dataset)


def iterate_masked_contrast_maps(contrast_name, statistic_type='z_score',
                                 data_dir=CAMCAN_DRAGO_STORE_CONTRASTS,
                                 patients_excluded=None, mask_file=None):
    """Load masked contrast maps for Cam-CAN data set.

    Parameters
    ----------
    contrast_name : str,
       Regex pattern that matches the contrasts you want

    statistic_type : str, (default='z_score')
        The kind of statistical map to load.

    data_dir : str, (default=CAMCAN_DRAGO_STORE_CONTRASTS)
        Root directory containing the root data.

    patients_excluded : str, tuple of str or None, optional (default=None)
        - If a string, corresponds to the path of a csv file. The first line
        of this csv file should contain the name of each column.
        - If a tuple of strings, contains the ID of the patient to be
        excluded. The string provided should follow the BIDS standard (e.g.,
        'sub-******').

    mask_file : str or None (default=None)
        where to find the mask. if none we look for it in data_dir

    Yields
    -------
    contrast_data : pandas.DataFrame,
        first columns are:
        - 'contrast_map', the path to the map;
        - 'contrast_name', the name of the contrast;
        - 'mask', the path to the mask.
        - 'subject_id', the ID of the patient;
        subsequent columns contain the voxels
    yields one dataframe per contrast

    """
    contrast_maps = load_camcan_contrast_maps(
        contrast_name, statistic_type, data_dir,
        patients_excluded, mask_file)
    masker = input_data.NiftiMasker(mask_img=contrast_maps.mask)
    contrast_maps = pd.DataFrame(contrast_maps)
    by_contrast = contrast_maps.groupby('contrast_name')
    masker.fit()
    for contrast_name, contrast_maps in by_contrast:
        masked_file = 'masked_contrasts_{}_{}.csv'.format(
            contrast_name, statistic_type)
        masked_file = os.path.join(data_dir, masked_file)
        if os.path.isfile(masked_file):
            contrast_data = pd.read_csv(masked_file)
        else:
            masked_maps = pd.DataFrame(
                masker.transform(contrast_maps.contrast_map),
                index=contrast_maps.index)
            contrast_data = pd.concat((contrast_maps, masked_maps), axis=1)
            contrast_data.to_csv(masked_file, index=False)
        yield contrast_data, masker


def load_masked_contrast_maps(contrast_name, statistic_type='z_score',
                              data_dir=CAMCAN_DRAGO_STORE_CONTRASTS,
                              patients_excluded=None, mask_file=None):
    """Load masked contrast maps for Camcan.

    Parameters
    ----------
    contrast_name : str,
       Regex pattern that matches the contrasts you want

    statistic_type : str, (default='z_score')
        The kind of statistical map to load.

    data_dir : str, (default=CAMCAN_DRAGO_STORE_CONTRASTS)
        Root directory containing the root data.

    patients_excluded : str, tuple of str or None, optional (default=None)
        - If a string, corresponds to the path of a csv file. The first line
        of this csv file should contain the name of each column.
        - If a tuple of strings, contains the ID of the patient to be
        excluded. The string provided should follow the BIDS standard (e.g.,
        'sub-******').

    mask_file : str or None (default=None)
        where to find the mask. if none we look for it in data_dir

    Returns
    -------
    contrast_data : pandas.DataFrame,
        first columns are:
        - 'contrast_map', the path to the map;
        - 'contrast_name', the name of the contrast;
        - 'mask', the path to the mask.
        - 'subject_id', the ID of the patient;
        subsequent columns contain the voxels

    """
    contrasts, maskers = zip(*iterate_masked_contrast_maps(
        contrast_name, statistic_type, data_dir,
        patients_excluded, mask_file))
    return pd.concat(contrasts), maskers[0]


def load_camcan_behavioural(filename_csv,
                            patients_info_csv=None,
                            patients_excluded=None,
                            column_selected=None):
    """Load the Cam-CAN cognitive behavioral data set.

    This loader returns a Bunch object containing the behavioral scores. The
    data of interest are:

    - All cognitive behavioral scores;
    - Patient ID;
    - Scores.

    See the description of the data to get all the information.

    Parameters
    ----------
    filename_csv : str,
        Path to the csv file containing the behavioral information.

    patients_info_csv : str or None, (default=None)
        Path to the CSV file containing the patients information.
        If None, data.scores will be a list of None.

    patients_excluded : str, tuple of str or None, optional (default=None)
        - If a string, corresponds to the path of a csv file. The first line
        of this csv file should contain the name of each column.
        - If a tuple of strings, contains the ID of the patient to be
        excluded. The string provided should follow the BIDS standard (e.g.,
        'sub-******').

    Returns
    -------
    data : Bunch,
        Dictionary-like object. The interesting attributes are:

        - 'data', the behavioral scores;
        - 'subject_id', the ID of the patient;
        - 'scores', a dictionary containing the different scores: age,
        handedness, and gender;
        - 'DESCR', the description of the full dataset.

    """
    if not isfile(filename_csv):
        raise ValueError('The file {} does not exist.'.format(filename_csv))

    if not filename_csv.endswith('.csv'):
        raise ValueError('The file {} is not a CSV file.'.format(filename_csv))

    patients_excluded_ = _validate_patients_excluded(patients_excluded)

    behavioral_data = pd.read_csv(filename_csv,
                                  usecols=column_selected,
                                  sep=';')
    # the id in the CSV is missing 'sub-'
    behavioral_data['Observations'] = 'sub-' + behavioral_data['Observations']
    # filter the IDs to be removed
    behavioral_data = behavioral_data[
        ~behavioral_data['Observations'].isin(list(patients_excluded_))]

    module_path = dirname(__file__)
    with open(join(module_path, 'descr', 'camcan.rst')) as rst_file:
        fdescr = rst_file.read()

    dataset = {'data': behavioral_data.drop('Observations', axis=1),
               'subject_id': behavioral_data['Observations'],
               'DESCR': fdescr}

    scores = _check_scores(patients_info_csv, dataset['subject_id'])
    dataset['scores'] = scores

    return Bunch(**dataset)


def load_camcan_behavioural_feature(exp_feat_map_json, name_experiment):
    """Load the Cam-CAN cognitive behavioral data set.

    This loader returns a list containing the features of a requested dataset.

    Parameters
    ----------
    exp_feat_map_json : str,
        Path to the JSON file containing the mapping experiment-features.

    name_experiment : str,
        Name of the experiment folder containing the behavioural information.
        Choices are: "ForceMatching", "RTchoice", "Hotel", "EkmanEmHex",
        "MotorLearning", "VSTMcolour", "EmotionalMemory", "Synsem", "RTsimple",
        "PicturePriming", "CardioMeasures", "MRI", "TOT", "EmotionRegulation",
        "FamousFace", "Proverbs", "BentonFaces", "Cattell", "HomeInterview".

    Returns
    -------
    features : tuple of str,
        The features associated to the experiment requested.

    """
    if not isfile(exp_feat_map_json):
        raise ValueError('The file {} does not exist.'.format(
            exp_feat_map_json))

    if not exp_feat_map_json.endswith('.json'):
        raise ValueError('The file {} is not a JSON file.'.format(
            exp_feat_map_json))

    with open(exp_feat_map_json, 'r') as filename:
        exp_features_map = json.load(filename)

    for key in exp_features_map:
        exp_features_map[key] = tuple(exp_features_map[key])

    if name_experiment not in exp_features_map:
        raise KeyError('{} does not exist.'.format(name_experiment))

    return tuple(exp_features_map[name_experiment])
