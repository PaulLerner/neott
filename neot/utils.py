# -*- coding: utf-8 -*-

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
        