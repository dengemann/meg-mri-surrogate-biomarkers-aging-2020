"""Structural data from FreeSurfer output."""
from itertools import product
import os

from ..utils import run_fs


def get_structural_data(subjects_dir, subject, out_dir):
    """Extract structural data.

    The data includes cortical sickness, cortical surface area,
    and subcortical volumes from FreeSurfer processing output.

    Parameters
    ----------
    subjects_dir : str
        A directory where subjects data are stored.
    subject : str
        The subject, for which data should be extracted.
    out_dir : str
        The output directory.

    """
    out_files = get_cortex_data(subjects_dir, subject, out_dir)
    out_files["aseg" + "_file"] = get_volumes_data(subjects_dir, subject,
                                                   out_dir)

    return out_files


def get_volumes_data(subjects_dir, subject, out_dir):
    """Extract sub-cortical volumes information from aseg files.

    Parameters
    ----------
    subjects_dir : str
        A directory where subjects data are stored.
    subject : str
        The subject, for which data should be extracted.
    out_dir : str
        The output directory.

    Returns
    -------
    out_file : str
        The path to generated data.

    """
    subject_dir = os.path.join(out_dir, subject)

    if not os.path.isdir(subject_dir):
        os.makedirs(subject_dir)

    out_file = os.path.join(subject_dir, 'aseg.csv')
    cmd = 'python2 $FREESURFER_HOME/bin/asegstats2table'\
          f' --subjects {subject} --tablefile {out_file}'\
          ' -d comma --meas volume'
    print(subject, out_file)
    print(cmd)
    run_fs(cmd, env={'SUBJECTS_DIR': subjects_dir})

    return out_file


def get_cortex_data(subjects_dir, subject, out_dir):
    """Extract cortical thickness and surface area.

    Parameters
    ----------
    subjects_dir : str
        A directory where subjects data are stored.
    subject : str
        The subject, for which data should be extracted.
    out_dir : str
        The output directory.

    Returns
    -------
    out_files : dict
        A dictionary with the paths to generated data.

    """
    out_files = {}
    # surfmeasure
    meas = ('thickness', 'area')
    hemi = ('lh', 'rh')

    for h, m in product(hemi, meas):
        subject_dir = os.path.join(out_dir, subject)

        if not os.path.isdir(subject_dir):
            os.makedirs(subject_dir)

        out_file = os.path.join(subject_dir, f'{h}.{m}.mgh')
        out_files[h + '_' + m + '_file'] = out_file

        cmd = f'mris_preproc --s {subject} --target fsaverage --hemi {h} '\
              f'--meas {m} --out {out_file}'
        run_fs(cmd, env={'SUBJECTS_DIR': subjects_dir})

    return out_files
