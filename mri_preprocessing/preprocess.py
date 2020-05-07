"""Include pypreprocess for fMRI data preprocessing."""
import os
import sys
from pypreprocess.nipype_preproc_spm_utils import do_subjects_preproc

config = sys.argv[-1]
_, extension = os.path.splitext(config)

if extension != '.ini':
    print('%s is not .ini file' % config)
else:
    # preprocess the data
    results = do_subjects_preproc(config)
