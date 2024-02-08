import json
from jsonargparse import CLI

import numpy as np


def split(data_path: str, test_ratio: float = 0.5, dev_ratio: float = None):
    """Split data"""
    with open(data_path, 'rt') as file:
        data = json.load(file)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    subsets = {}
    n_test = int(len(data) * test_ratio)
    subsets["test"] = [data[i] for i in indices[: n_test]]
    if dev_ratio is not None:
        n_dev = n_test + int(len(data) * dev_ratio)
        subsets["dev"] = [data[i] for i in indices[n_test: n_dev]]
    else:
        n_dev = n_test
    subsets["train"] = [data[i] for i in indices[n_dev:]]
    for name, subset in subsets.items():
        print(name, len(subset))
    with open(data_path, 'wt') as file:
        json.dump(subsets, file)


if __name__ == "__main__":
    CLI(main, description=main.__doc__)
