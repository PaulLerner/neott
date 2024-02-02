#!/usr/bin/env python
# coding: utf-8


import json
from collections import Counter

import pandas as pd

import seaborn as sns

from ..morph.classes import MorphLabel
from ..utils import random_data

with open("data/FranceTerme_triples.json", "rt") as file:
    data = json.load(file)

sym_diff = []

labels, fr_labels, en_labels = [], [], []
eg = {}
en2fr = {}
tps, cps, pps = Counter(), Counter(), Counter()

for item in random_data(data["train"]):
    p_fr = item["fr"]["morph_label"]
    p_en = item["en"]["morph_label"]
    sym_diff.append(len(set(p_en).symmetric_difference(set(p_fr))))
    p_fr = " ".join(sorted(p_fr))
    p_en = " ".join(sorted(p_en))
    en2fr.setdefault(p_en, Counter())
    en2fr[p_en][p_fr] += 1
    fr_labels.append(p_fr)
    en_labels.append(p_en)
    for label in MorphLabel:
        if label.name in p_en:
            pps[label.name] += 1
            if label in p_fr:
                tps[label.name] += 1
        if label in p_fr:
            cps[label.name] += 1

    p_bi = f"{p_en} = {p_fr}"
    if p_bi not in eg:
        eg[p_bi] = (item["en"]["text"], item["fr"]["text"])
    labels.append(p_bi)

metrics_per_label = {}
for k, v in tps.items():
    metrics_per_label[k] = {"precision": v/pps[k], "recall": v/cps[k]}
metrics_per_label = pd.DataFrame(metrics_per_label).T
metrics_per_label['f1'] = (2*metrics_per_label['precision']*metrics_per_label['recall'])/(metrics_per_label['precision']+metrics_per_label['recall'])
print((metrics_per_label*100).to_latex(float_format='%.1f'))

sns.displot(sym_diff, discrete=True)

Counter(sym_diff)

for p_bi, count in Counter(labels).most_common(20):
    print(" & ".join(p_bi.split("=")), "&", count, "&", " & ".join(eg[p_bi]), r"\\")

print(pd.DataFrame(Counter(fr_labels).most_common()).to_latex(index=False))

print(pd.DataFrame(Counter(en_labels).most_common()).to_latex(index=False))

for en, count in Counter(en_labels).most_common(20):
    print(en, "&", count, "&", " / ".join([k for k, _ in en2fr[en].most_common(5)]), r"\\")
