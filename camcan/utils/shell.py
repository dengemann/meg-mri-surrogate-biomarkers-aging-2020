"""Functions for shell commands execution."""
import logging
import os
from subprocess import run, PIPE


def run_fs(cmd, env={}, ignore_errors=False):
    """Execute FreeSurfer command.

    Parameters
    ----------
    cmd : str
        A command to execute.
    env : dict
        Environment variables to be defined.
    ignore_errors: bool
        If True exception will be raised in case of command execution error.
        Default is False.

    """
    merged_env = os.environ
    merged_env.update(env)
    # DEBUG env triggers freesurfer to produce gigabytes of files
    merged_env.pop('DEBUG', None)
    logging.basicConfig(level=logging.INFO)

    process = run(cmd, shell=True, stdout=PIPE, stderr=PIPE, env=merged_env)
    logging.info(process.stdout.decode('utf-8'))

    if process.returncode != 0 and not ignore_errors:
        logging.error(f'Non zero return code: {process.returncode}.' +
                      f'Bash: {process.stderr}, cmd: {cmd}')
