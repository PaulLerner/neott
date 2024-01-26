#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import json
from jsonargparse import CLI

import seaborn as sns

import fasttext

from ..utils import Path


class Classifier:
    """Multi-label classifier for neologisms morphology"""
    def __init__(self, train_path: Path, dev_path: Path, test_path: Path, model_path: Path = None):
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.model = None if model_path is None else fasttext.load_model(model_path)

    def train(self):
        assert self.model is None
        # hyper param tuning
        self.model = fasttext.train_supervised(input=str(self.train_path), loss='ova', autotuneValidationFile=str(self.dev_path))
        hyperparameters = {k: v for k, v in self.model.__dict__.items() if isinstance(v, (float, int))}
        print(hyperparameters)
        with open(self.train_path.parent / "hyperparameters.json", "wt") as file:
            json.dump(hyperparameters, file)
        self.model.save_model(str(self.train_path.parent / "model.bin"))

        self.test()

    def test(self):
        results = self.model.test_label(str(self.test_path), k=-1, threshold=0.5)
        _, p, r = self.model.test(str(self.test_path), k=-1, threshold=0.5)
        f = (2 * p * r) / (p + r)
        results["overall"] = {'precision': p,
                              'recall': r,
                              'f1score': f}
        print((pd.DataFrame(results) * 100).T.to_latex(float_format="{:.1f}".format))

        self.viz()

    def viz(self):
        precision, recall = [], []
        for threshold in np.arange(0.0, 1.01, 0.01):
            _, p, r = self.model.test(str(self.test_path), k=-1, threshold=threshold)
            precision.append(p)
            recall.append(r)

        p_v_r = pd.DataFrame(dict(precision=precision, recall=recall))
        fig = sns.relplot(p_v_r, x="precision", y="recall", kind="line", markers=True)
        fig.savefig(self.test_path.parent/"p_v_r.pdf")
        return fig


if __name__ == "__main__":
    CLI(Classifier)
