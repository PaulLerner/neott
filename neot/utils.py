# -*- coding: utf-8 -*-
import itertools

from jsonargparse.typing import register_type
from pathlib import Path
import numpy as np

register_type(Path, type_check=lambda v, t: isinstance(v, t))


def random_data(data, n=None):
    if n is None:
        n = len(data)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    for i in indices[:n]:
        yield data[i]


def infinite_random_data(data):
    indices = np.arange(len(data))
    while True:
        np.random.shuffle(indices)
        for i in indices:
            yield data[i]


def iter_kwargs_prod(kwargs):
    for values in itertools.product(*kwargs.values()):
        k_v = {}
        for k, v in zip(kwargs, values):
            k_v[k] = v
        yield k_v


class ListOrArg(list):
    """Utility class useful to accept either a single arg or a list of arg"""
    def __init__(self, arg):
        if not isinstance(arg, list):
            arg = [arg]
        super().__init__(arg)
