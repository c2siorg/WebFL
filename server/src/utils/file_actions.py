import pickle
import os

import warnings

def loads(res):
    return pickle.loads(res)


def save(obj, f):
    # disabling warnings from torch.Tensor's reduce function. See issue: https://github.com/pytorch/pytorch/issues/38597
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(f, "wb") as opened_f:
            pickle.dump(obj, opened_f)


def mkdir_save(obj, f):
    dir_name = os.path.dirname(f)
    if dir_name == "":
        save(obj, f)
    else:
        os.makedirs(dir_name, exist_ok=True)
        save(obj, f)


def load(f):
    with open(f, 'rb') as opened_f:
        return pickle.load(opened_f)