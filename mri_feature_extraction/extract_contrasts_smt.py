"""Extract contrasts."""
import numpy as np
import pandas as pd
import os
from os.path import join, isdir
from nistats.first_level_model import FirstLevelModel
from camcan.datasets import load_camcan_rest

# Cam-CAN dataset paths
CAMCAN_PREPROCESSED = '/home/mehdi/data/camcan/camcan_smt_preproc'
CAMCAN_PATIENTS_EXCLUDED = join(CAMCAN_PREPROCESSED, 'excluded_subjects.csv')
CAMCAN_MAPS = '/home/mehdi/data/camcan/camcan_smt_maps'
MASK_IMG = join(CAMCAN_PREPROCESSED, 'mask_camcan.nii.gz')

# path for the caching
CACHE_MAPS = '/home/mehdi/data/camcan/cache/maps'


# load dataset
dataset = load_camcan_rest(
    data_dir=CAMCAN_PREPROCESSED,
    patients_excluded=CAMCAN_PATIENTS_EXCLUDED)

for (subject_id, func, motion) in zip(dataset.subject_id,
                                      dataset.func, dataset.motion):
    print(subject_id)
    events_path = join(CAMCAN_PREPROCESSED, subject_id, 'func',
                       '%s_task-SMT_events.tsv' % subject_id)

    # First-level GLM
    flm = FirstLevelModel(
        t_r=1.97, mask=MASK_IMG, smoothing_fwhm=8, verbose=1,
        memory_level=1, memory=CACHE_MAPS,
        subject_label=subject_id, n_jobs=10)
    flm.fit(run_imgs=func, events=pd.read_csv(events_path, sep='\t'),
            confounds=pd.DataFrame(np.loadtxt(motion)))

    # Prepare contrasts
    contrasts = {}
    contrast_matrix = np.eye(flm.design_matrices_[0].shape[1])
    contrasts = dict(
        [(column, contrast_matrix[i])
         for i, column in enumerate(flm.design_matrices_[0].columns[:5])])
    # 'AudOnly', 'AudVid1200',  'AudVid300',  'AudVid600',    'VidOnly'
    contrasts.update(
        {"Aud-Vid": contrasts["AudOnly"] - contrasts["VidOnly"],
         "AV1200-AV300": contrasts["AudVid1200"] - contrasts["AudVid300"],
         "AV1200-AV600": contrasts["AudVid1200"] - contrasts["AudVid600"],
         "AV600-AV300": contrasts["AudVid600"] - contrasts["AudVid300"],
         "AV-Vid": (contrasts["AudVid1200"] +
                    contrasts["AudVid600"] +
                    contrasts["AudVid300"]) - contrasts["VidOnly"],
         "AV-Aud": (contrasts["AudVid1200"] +
                    contrasts["AudVid600"] +
                    contrasts["AudVid300"]) - contrasts["AudOnly"],
         })

    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        print('  Contrast % 2i out of %i: %s' %
              (index + 1, len(contrasts), contrast_id))
        for output_type in ['stat', 'z_score',
                            'effect_size', 'effect_variance']:
            print(output_type)
            # compute stat_map
            stat_map = flm.compute_contrast(
                contrast_val, output_type=output_type)
            # save stat_map
            subject_path = join(CAMCAN_MAPS, subject_id)
            if not isdir(subject_path):
                os.makedirs(subject_path)
            output_file = '%s_%s_%s.nii.gz' % (subject_id, contrast_id,
                                               output_type)
            stat_map.to_filename(join(subject_path, output_file))
