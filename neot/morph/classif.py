#!/usr/bin/env python
# coding: utf-8
from collections import Counter

import pandas as pd
import numpy as np
import json
from jsonargparse import CLI

import seaborn as sns

import fasttext

from ..utils import Path


class Classifier:
    """Multi-label classifier for neologisms morphology"""
    def __init__(self, model_path: str = None):
        self.model = None if model_path is None else fasttext.load_model(model_path)

    def train(self, train_path: Path, dev_path: Path, test_path: Path = None):
        """Train"""
        assert self.model is None
        # hyper param tuning
        self.model = fasttext.train_supervised(input=str(train_path), loss='ova', autotuneValidationFile=str(dev_path))
        hyperparameters = {k: v for k, v in self.model.__dict__.items() if isinstance(v, (float, int))}
        print(hyperparameters)
        with open(train_path.parent / "hyperparameters.json", "wt") as file:
            json.dump(hyperparameters, file)
        self.model.save_model(str(train_path.parent / "model.bin"))

        if test_path is not None:
            self.test(test_path)

    def test(self, test_path: Path):
        """Test"""
        results = self.model.test_label(str(test_path), k=-1, threshold=0.5)
        _, p, r = self.model.test(str(test_path), k=-1, threshold=0.5)
        f = (2 * p * r) / (p + r)
        results["overall"] = {'precision': p,
                              'recall': r,
                              'f1score': f}
        print((pd.DataFrame(results) * 100).T.to_latex(float_format="{:.1f}".format))

        self.viz(test_path)

    def viz(self, test_path: Path):
        """Visualize"""
        precision, recall = [], []
        for threshold in np.arange(0.0, 1.01, 0.01):
            _, p, r = self.model.test(str(test_path), k=-1, threshold=threshold)
            precision.append(p)
            recall.append(r)

        p_v_r = pd.DataFrame(dict(precision=precision, recall=recall))
        fig = sns.relplot(p_v_r, x="precision", y="recall", kind="line", markers=True)
        fig.savefig(test_path.parent / "p_v_r.pdf")
        return fig

    def predict(self, predict_path: Path, lang: str):
        """Predict on OOD data"""
        with open(predict_path, 'rt') as file:
            data = json.load(file)
        labels = Counter()
        for subset in data.values():
            for item in subset:
                label = [l[len('__label__'):] for l in self.model.predict(item[lang]['text'], k=-1, threshold=0.5)[0]]
                if len(item[lang]["tokens"]) > 1:
                    label.append("Syntagm")
                item[lang]["morph_label"] = label
                labels[" ".join(sorted(label))] += 1
        print(labels.most_common())
        with open(predict_path, 'wt') as file:
            json.dump(data, file)


if __name__ == "__main__":
    CLI(Classifier)
