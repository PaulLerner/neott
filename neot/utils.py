# -*- coding: utf-8 -*-

import numpy as np

def random_data(data, n=None):
    if n is None:
        n = len(data)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    for i in indices[:n]:
        yield data[i]
