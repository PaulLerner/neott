#!/usr/bin/env python
# coding: utf-8


import json
from collections import Counter

from scipy.stats import entropy
import numpy as np
import pandas as pd

import seaborn as sns

from ..utils import random_data

with open("../data/FranceTerme_triples.json", "rt") as file:
    data = json.load(file)

sym_diff = []

labels, fr_labels, en_labels = [], [], []
eg = {}
en2fr = {}
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
    p_bi = f"{p_en} = {p_fr}"
    if p_bi not in eg:
        eg[p_bi] = (item["en"]["text"], item["fr"]["text"])
    labels.append(p_bi)

sns.displot(sym_diff, discrete=True)

Counter(sym_diff)

for p_bi, count in Counter(labels).most_common(20):
    print(" & ".join(p_bi.split("=")), "&", count, "&", " & ".join(eg[p_bi]), r"\\")

print(pd.DataFrame(Counter(fr_labels).most_common()).to_latex(index=False))

print(pd.DataFrame(Counter(en_labels).most_common()).to_latex(index=False))

for en, count in Counter(en_labels).most_common(20):
    p = en2fr[en].most_common(1)[0][1] / count
    e = entropy(np.array(list(en2fr[en].values())) / count)
    print(en, "&", count, "&", f"{p:.2f} & ", f"{e:.2f} & ", " / ".join([k for k, _ in en2fr[en].most_common(5)]),
          r"\\")
