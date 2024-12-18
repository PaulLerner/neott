# -*- coding: utf-8 -*-
import itertools
import json

import numpy as np


def load_json_line(path):
    with open(path, "rt") as file:
        for line in file.readlines():
            yield json.loads(line)


def random_data(data, n=None):
    if n is None:
        n = len(data)
    assert n > 0
    # TODO refactor with choice
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    for i in indices[:n]:
        yield data[i]


def infinite_random_data(data):
    assert len(data) > 0
    indices = np.arange(len(data))
    while True:
        np.random.shuffle(indices)
        for i in indices:
            yield data[i]


def all_size_combination(iterable):
    for r in range(len(iterable)+1):
        for p in itertools.combinations(iterable, r):
            yield p


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
