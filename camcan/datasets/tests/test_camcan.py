import types
from os.path import join, split

from sklearn.utils.testing import (assert_equal, assert_true,
                                   assert_array_equal,
                                   assert_raises_regex)

from camcan.datasets.camcan import _validate_patients_excluded
from camcan.datasets.camcan import _exclude_patients
from camcan.datasets.camcan import _load_camcan_scores
from camcan.datasets.camcan import _check_scores
from camcan.datasets.camcan import _abs_listdir

from camcan.datasets import load_camcan_behavioural_feature
from camcan.datasets import load_camcan_behavioural


def test_validate_patients_excluded_errors():
    assert_raises_regex(ValueError, 'should be a tuple.',
                        _validate_patients_excluded, 0)
    assert_raises_regex(ValueError, "follow the pattern 'sub-'",
                        _validate_patients_excluded, tuple('random'))
    assert_raises_regex(ValueError, "a csv file is required",
                        _validate_patients_excluded, 'file.rnd')
    assert_raises_regex(ValueError, "File not found.",
                        _validate_patients_excluded, "file.csv")


def test_validate_patients_excluded():
    patients_excluded = _validate_patients_excluded(None)
    assert_equal(patients_excluded, tuple([]))

    patients_excluded = _validate_patients_excluded(('sub-0', 'sub-1'))
    assert_equal(patients_excluded, ('sub-0', 'sub-1'))

    current_dir = split(__file__)[0]
    filename = join(current_dir, "data", "patients_selection.csv")
    patients_excluded = _validate_patients_excluded(filename)
    assert_equal(patients_excluded, ('sub-0', 'sub-2', 'sub-3'))


def test_exclude_patients():
    current_dir = split(__file__)[0]
    patients_dir = _exclude_patients(join(current_dir, 'data'), tuple([]))
    assert_equal(len(patients_dir), 2)
    for patient_idx, patient_dir in enumerate(patients_dir):
        assert_true('data/sub-' + str(patient_idx) in patient_dir)

    patients_dir = _exclude_patients(join(current_dir, 'data'),
                                     tuple(['sub-0']))
    assert_equal(len(patients_dir), 1)
    assert_true('data/sub-1' in patient_dir)


def test_load_camcan_scores_error():
    assert_raises_regex(ValueError, 'does not exist.',
                        _load_camcan_scores, 'file.csv', ['sub-0'])
    current_dir = split(__file__)[0]
    assert_raises_regex(ValueError, 'is not a CSV file.',
                        _load_camcan_scores, join(current_dir, '__init__.py'),
                        ['sub-0'])


def test_load_camcan_scores():
    current_dir = split(__file__)[0]
    filename = join(current_dir, 'data', 'participants_info.csv')
    scores = _load_camcan_scores(filename, ['sub-CC0', 'sub-CC1'])
    assert_array_equal(scores.age, [18, 20])
    assert_array_equal(scores.hand, [-100, 100])
    assert_array_equal(scores.gender_text, ['FEMALE', 'MALE'])


def test_check_scores():
    scores = _check_scores(None, ['sub-CC0', 'sub-CC1'])
    assert_array_equal(scores.age, [None, None])
    assert_array_equal(scores.hand, [None, None])
    assert_array_equal(scores.gender_text, [None, None])
    current_dir = split(__file__)[0]
    filename = join(current_dir, 'data', 'participants_info.csv')
    scores = _check_scores(filename, ['sub-CC0', 'sub-CC1'])
    assert_array_equal(scores.age, [18, 20])
    assert_array_equal(scores.hand, [-100, 100])
    assert_array_equal(scores.gender_text, ['FEMALE', 'MALE'])


def test_load_camcan_behavioural_feature():
    current_dir = split(__file__)[0]
    filename = join(current_dir, 'data', 'behavioural_features.json')
    features = load_camcan_behavioural_feature(filename, 'Hotel')
    assert_equal(features, ('NumRows', 'Time', 'QCflag'))


def test_abs_listdir():
    current_dir = split(__file__)[0]
    files = _abs_listdir(join(current_dir, 'data', 'sub-1'))
    assert_equal(type(files), types.GeneratorType)
    assert_equal(list(files)[0], join(current_dir, 'data',
                                      'sub-1', '__init__.py'))


def test_load_camcan_behavioural_error():
    assert_raises_regex(ValueError, 'does not exist.',
                        load_camcan_behavioural, 'rnd.json')
    current_dir = split(__file__)[0]
    filename = join(current_dir, 'data', 'behavioural_features.json')
    assert_raises_regex(ValueError, 'is not a CSV file.',
                        load_camcan_behavioural, filename)


def test_load_camcan_behavioural():
    current_dir = split(__file__)[0]
    filename = join(current_dir, 'data', 'total_score.csv')
    dataset = load_camcan_behavioural(filename)
    assert_equal(dataset.data.shape, (708, 679))
    assert_equal(len(dataset.subject_id), 708)
    assert_equal(len(dataset.scores.age), 708)
    assert_true(dataset.scores.age[0] is None)
    current_dir = split(__file__)[0]
    filename = join(current_dir, 'data', 'total_score.csv')
    participant_info = join(current_dir, 'data', 'participant_data.csv')
    dataset = load_camcan_behavioural(filename, participant_info)
    assert_equal(dataset.data.shape, (708, 679))
    assert_equal(len(dataset.subject_id), 708)
    assert_equal(len(dataset.scores.age), 708)
    assert_equal(dataset.scores.age[0], 24)
