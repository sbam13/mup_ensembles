import os
import shutil
import logging

from os.path import exists

import src.run.constants as constants

def create_tmp_folder(folder):
    if not exists(folder):
        logging.info(f'Temp folder {folder} already exists.')
    else:
        logging.info(f'Creating temp folder {folder}.')
        os.mkdir(folder)
    return folder

def copy_results_into_permanent(local_results_folder, remote_results_dirname, 
                                remote_results_directory = constants.REMOTE_RESULTS_FOLDER):
    """Copies the contents of 'local_results_folder' to a folder named 
        'remote_results_dirname' located within 'remote_results_directory'."""
    RESULTS_FOLDER = os.path.join(remote_results_directory, remote_results_dirname)
    return shutil.copytree(local_results_folder, RESULTS_FOLDER)