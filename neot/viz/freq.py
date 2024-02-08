#!/usr/bin/env python
# coding: utf-8

import json
from collections import Counter
from typing import Union, List

from jsonargparse import CLI

import numpy as np
import pandas as pd

import seaborn as sns

from neot.utils import ListOrArg


def viz(freq, output):
    print(f"Top-50\n{pd.DataFrame(freq.most_common(50)).to_latex(index=False)}")

    freq_v = np.array(list(freq.values()))
    freq_k = np.array(list(freq.keys()))
    deciles = np.quantile(freq_v, np.arange(0, 1.1, 0.1), method="nearest")

    equal_zero = len(np.where(freq_v == 0)[0])
    print(equal_zero, equal_zero/len(freq_k))
    for decile, v in zip(np.arange(0, 1.1, 0.1), deciles):
        indices = np.where(freq_v == v)[0]
        np.random.shuffle(indices)
        print(decile, "&", freq_k[indices[0]], "&", v, r"\\")

    # shift by 1 to plot zero values in log-scale
    fig = sns.displot(freq_v+1, bins=100, log_scale=True)
    fig.savefig(output)


def main(output: str, freq_paths: Union[str, List[str]] = None):
    freq_paths = ListOrArg(freq_paths)
    freq = Counter()
    for freq_path in freq_paths:
        with open(freq_path, "rt") as file:
            for k, v in Counter(json.load(file)).items():
                # Counter.__add__ will remove 0
                freq[k] += v
    viz(freq, output)


if __name__ == "__main__":
    CLI(main)
