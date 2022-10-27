from pickle import load

def read_result(fname):
    with open(fname, 'rb') as f:
        res = load(f)
    return res