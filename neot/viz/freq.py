#!/usr/bin/env python
# coding: utf-8

import json
from collections import Counter
from jsonargparse import CLI

import numpy as np
import pandas as pd

import seaborn as sns


def viz(freq, output):
    print(f"Top-50\n{pd.DataFrame(freq.most_common(50)).to_latex(index=False)}")

    freq_v = np.array(list(freq.values()))
    freq_k = np.array(list(freq.keys()))
    deciles = np.quantile(freq_v, np.arange(0, 1.1, 0.1), method="nearest")

    for v in deciles:
        i = np.where(freq_v == v)[0][0]
        print(freq_k[i], "&", v, r"\\")

    # shift by 1 to plot zero values in log-scale
    fig = sns.displot(freq_v+1, bins=100, log_scale=True)
    fig.savefig(output)


def main(freq: str, output: str):
    with open(freq, "rt") as file:
        freq = Counter(json.load(file))
    viz(freq, output)


if __name__ == "__main__":
    CLI(main)
