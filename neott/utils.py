# -*- coding: utf-8 -*-
import itertools
import json
from jsonargparse.typing import register_type
from pathlib import Path

import numpy as np

register_type(Path, type_check=lambda v, t: isinstance(v, t))


def load_lightning(ckpt: Path = None, config_path: Path = None):
    """
    Lighning experiments are usually organized as follows:

    root
    ├── lightning_logs
    │   └── version_47
    │       ├── checkpoints
    │       │   └── step=912.ckpt <- provide with ckpt
    │       └── config.yaml       <- will resolve to this config
    └── test.yaml                 <- OR provide with config_path (with a "ckpt_path" key)

    Parameters
    ----------
    ckpt: Path, optional
        e.g. root/lightning_logs/checkpoints/step=912.ckpt above
        config_path will then be set to ../../config.yaml -> root/lightning_logs/config.yaml
    config_path: Path, optional (exclusive of ckpt)
        e.g. root/test.yaml above
        If you already have configured (e.g. for test/validation/predict) with a "ckpt_path" key
    """

    if config_path is None:
        assert ckpt is not None, "you must provide either ckpt or config (or both)"
        config_path = ckpt.parent.parent / 'config.yaml'
    with open(config_path, 'rt') as file:
        config = yaml.load(file, yaml.Loader)
    if ckpt is None:
        ckpt = Path(config["ckpt_path"])
    class_name = config['model']['class_path'].split('.')[-1]
    Class = getattr(trainee, class_name)
    model = Class.load_from_checkpoint(ckpt, **config['model']['init_args'])
    return model


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
