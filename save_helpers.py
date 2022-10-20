import os
import shutil

def create_tmp_folders(data_folder="/tmp/data", results_folder="/tmp/results"):
    os.mkdir(data_folder)
    os.mkdir(results_folder)
    return data_folder, results_folder

def copy_results_into_permanent(local_results_folder, remote_results_folder_name):
    RESULTS_PREFIX = "/n/holystore01/LABS/pehlevan_lab/Users/sab/results"
    RESULTS_FOLDER = os.path.join(RESULTS_PREFIX, remote_results_folder_name)
    return shutil.copytree(local_results_folder, RESULTS_FOLDER)