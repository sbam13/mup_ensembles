import os
from src.run.save_helpers import create_tmp_folders, copy_results_into_permanent

from os.path import exists

def test_create_tmp_folders(tmp_path):
    dt, rt = tmp_path / 'data', tmp_path / 'results'
    df, rf = create_tmp_folders(str(dt), str(rt))
    assert exists(df)
    assert exists(rf)


def test_copy_results_folder(tmp_path):
    remote_dir = tmp_path / 'remote'
    remote_dir.mkdir()
    remote_dirname = 'rr'
    
    local_results_folder = tmp_path / 'results'
    local_results_folder.mkdir()
    task_loc = local_results_folder / 'result-1'
    task_loc.mkdir()
    task_loc_hello = task_loc / "hello.txt"
    task_loc_hello.write_text('hello')

    res_d = copy_results_into_permanent(str(local_results_folder), 
                                        remote_dirname,
                                        str(remote_dir))
    rpath = remote_dir / remote_dirname
    assert str(rpath) == res_d
    print(os.listdir(res_d))
    assert exists(str(rpath / 'result-1'))
    assert exists(str(rpath / 'result-1' / 'hello.txt'))
    assert (rpath / 'result-1' / 'hello.txt').read_text() == 'hello'

