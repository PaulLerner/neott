#!/usr/bin/env python
# coding: utf-8

import json
from collections import Counter
import random
from jsonargparse import CLI
from tqdm import tqdm

import numpy as np
from scipy.stats import entropy

import spacy
from spacy.lang.en import English
from spacy.lang.fr import French

print(f"{spacy.prefer_gpu()=}")


def tag(data, model, lang, tagging=True, batch_size=2048):
    docs = list(tqdm(model.pipe([item[lang]["text"] for item in data], batch_size=batch_size)))
    for item, doc in zip(data, docs):
        item[lang]["tokens"] = [t.text for t in doc]
        if tagging:
            item[lang]["pos"] = [t.pos_ for t in doc]
            item[lang]["dep"] = [t.dep_ for t in doc]


def viz_dep(data):
    deps = {"en": [], "fr": []}

    en_fr_deps = []

    for item in data:
        en_fr_dep = []
        for lang, l_dep in deps.items():
            l_dep.append(" ".join(item[lang]["dep"]))
            en_fr_dep.append(l_dep[-1])
        en_fr_deps.append(f"{en_fr_dep[0]} = {en_fr_dep[1]}")

    print(f"\n{Counter(deps['en']).most_common(100)=}")
    print(f"\n{Counter(deps['fr']).most_common(100)=}\n")

    random.shuffle(data)
    print(r"\textbf{DEP EN} & \textbf{DEP FR} & \textbf{Occurrences} & \textbf{Exemple EN} & \textbf{Trad FR}\\")
    for pos, count in Counter(en_fr_deps).most_common(20):
        for item in data:
            fr_pos = " ".join(item["fr"]["dep"])
            en_pos = " ".join(item["en"]["dep"])
            if pos == f"{en_pos} = {fr_pos}":
                print(en_pos, "&", fr_pos, "&", count, "&", item["en"]["text"], "&", item["fr"]["text"], r"\\")
                break


def viz_pos(data):
    fr_pos, en_pos = [], []
    en_fr_pos = []
    for item in data:
        fr_pos.append(" ".join(item["fr"]["pos"]))
        en_pos.append(" ".join(item["en"]["pos"]))
        en_fr_pos.append(f"{en_pos[-1]} = {fr_pos[-1]}")

    print(f"\n{Counter(en_pos).most_common(20)=}")
    print(f"\n{Counter(fr_pos).most_common(20)=}\n")

    print(r"\textbf{POS EN} & \textbf{POS FR} & \textbf{Occurrences} & \textbf{Exemple EN} & \textbf{Trad FR}\\")
    for pos, count in Counter(en_fr_pos).most_common(20):
        for item in data:
            fr_pos = " ".join(item["fr"]["pos"])
            en_p = " ".join(item["en"]["pos"])
            if pos == f"{en_p} = {fr_pos}":
                print(en_p, "&", fr_pos, "&", count, "&", item["en"]["text"], "&", item["fr"]["text"], r"\\")
                break

    en_poses = {}
    for item in data:
        fr_pos = " ".join(item["fr"]["pos"])
        en_p = " ".join(item["en"]["pos"])
        en_poses.setdefault(en_p, Counter())
        en_poses[en_p][fr_pos] += 1
    print()
    print(r"\textbf{POS EN}  &\textbf{Occurrences} & \textbf{Prior} & \textbf{Entropie} & \textbf{Top-5 POS FR}\\")
    for pos, count in Counter(en_pos).most_common(20):
        p = en_poses[pos].most_common(1)[0][1] / count
        e = entropy(np.array(list(en_poses[pos].values())) / count)
        print(pos, "&", count, "&", f"{p:.2f} & ", f"{e:.2f} & ",
              " / ".join([k for k, v in en_poses[pos].most_common(5)]), r"\\")


def main(data_path: str, tagging: bool = True):
    """POS tag or tokenization"""
    with open(data_path, "rt") as file:
        data = json.load(file)

    if tagging:
        disable = ["lemmatizer", "ner"]
        en = spacy.load("en_core_web_trf", disable=disable)
        fr = spacy.load("fr_dep_news_trf", disable=disable)
    else:
        en = English()
        fr = French()

    for name, subset in data.items():
        tag(subset, en, "en", tagging=tagging)
        tag(subset, fr, "fr", tagging=tagging)

    with open(data_path, "wt") as file:
        json.dump(data, file)

    if tagging:
        viz_dep(data["test"])
        viz_pos(data["test"])


if __name__ == "__main__":
    CLI(main, description=main.__doc__)
