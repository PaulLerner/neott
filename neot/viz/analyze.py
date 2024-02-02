#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import json
import seaborn as sns
from collections import Counter
from jsonargparse import CLI
import editdistance
import pandas as pd

from transformers import AutoTokenizer

import spacy

from ..utils import Path
from ..morph.classes import MorphLabel


def viz_f1(data, pred, metrics):
    for i, item in enumerate(data["train"][:100]):
        f1 = metrics["f1s"][i]
        if f1 > 0.0 and f1 < 1.0:
            print(f1, pred['predictions'][i], item["fr"]["text"])


def viz_wrong(data, pred, metrics):
    for i, item in enumerate(data["train"][:100]):
        f1 = metrics["f1s"][i]
        if f1 <= 0.0:
            print(f1, pred['predictions'][i], item["fr"]["text"])


def dist_f1(metrics, output):
    fig = sns.displot(metrics["f1s"])
    fig.savefig(output / "f1_dist.pdf")


def gather_results(data, metrics, tokenizer):
    results = []

    fr_ova = {c.name: {True: [], False: []} for c in MorphLabel}
    en_ova = {c.name: {True: [], False: []} for c in MorphLabel}
    per_dom = []
    for i, item in enumerate(data["train"]):
        p_fr = item["fr"]["morph_label"]
        p_en = item["en"]["morph_label"]
        em = metrics["ems"][i]
        for label in MorphLabel:
            fr_ova[label.name][label.name in p_fr].append(em)
            en_ova[label.name][label.name in p_en].append(em)
        if tokenizer is not None:
            term_fertility = len(tokenizer.tokenize(item['fr']["text"]))
            token_fertility = []
            for token in item["fr"]["tokens"]:
                token_fertility.append(len(tokenizer.tokenize(token)))
            token_fertility = max(token_fertility)
        else:
            term_fertility = None
            token_fertility = None
        results.append({
            "Morph. Diff.": len(set(p_en).symmetric_difference(set(p_fr))),
            "EM": em,
            "Edit dist.": editdistance.eval(item['fr']["text"], item['en']["text"]),
            "Term fertility": term_fertility,
            "Word fertility": token_fertility,
            "# words": len(item["fr"]["tokens"])
        })
        for dom in item["Dom"]:
            per_dom.append({"Domain": dom, "EM": em})

    results = pd.DataFrame(results)
    per_dom = pd.DataFrame(per_dom)

    for label in fr_ova:
        for label_exists in fr_ova[label]:
            fr_ova[label][label_exists] = sum(fr_ova[label][label_exists]) / len(fr_ova[label][label_exists])
    for label in en_ova:
        for label_exists in en_ova[label]:
            en_ova[label][label_exists] = sum(en_ova[label][label_exists]) / len(en_ova[label][label_exists])
    fr_ova = pd.DataFrame(fr_ova)
    en_ova = pd.DataFrame(en_ova)

    return results, per_dom, fr_ova, en_ova


def viz_dom(per_dom, output):
    top10 = {k for k, v in Counter(per_dom.Domain).most_common(10)}
    fig = sns.catplot(
        data=per_dom[per_dom.Domain.isin(top10)], x="Domain", y="EM",
        kind="bar"
    )
    plt.xticks(rotation=90)
    fig.savefig(output / "EM_per_domain.pdf")


def viz_dist(results, x, output):
    fig = sns.displot(results, x=x, hue="EM", multiple="dodge", discrete=True, stat="density", common_norm=False,
                      shrink=.8)
    fig.savefig(output / f"{x}_wrt_EM_dist.pdf")


def viz_dists(results, **kwargs):
    for x in ["Morph. Diff.", "Edit dist.", "Term fertility", "Word fertility", "# words"]:
        viz_dist(results, x, **kwargs)


def viz_ova(fr_ova, en_ova):
    print("EN\n", (en_ova * 100).to_latex(float_format="%.1f"))
    print("FR\n", (fr_ova * 100).to_latex(float_format="%.1f"))


def tag(pred, tagger):
    poses = []
    stripped_preds = [p.strip() for p in pred["predictions"]]
    for doc in tagger.pipe(stripped_preds, batch_size=2048):
        poses.append([t.pos_ for t in doc])

    pred["pos"] = poses


def main(data: Path, pred_path: Path, tokenizer: str = None, output: Path = None, tagger: str = None):
    with open(data, "rt") as file:
        data = json.load(file)

    with open(pred_path, "rt") as file:
        pred = json.load(file)
    if tagger is not None:
        print(f"{spacy.prefer_gpu()=}")
        tagger = spacy.load(tagger)
        tag(pred, tagger)
        with open(pred_path, "wt") as file:
            json.dump(pred, file)

    if tokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, add_prefix_space=True)

    metrics = pred["metrics"]
    viz_f1(data, pred, metrics)
    viz_wrong(data, pred, metrics)
    results, per_dom, fr_ova, en_ova = gather_results(data, metrics, tokenizer)
    viz_ova(fr_ova, en_ova)
    if output is not None:
        output.mkdir(exist_ok=True)
        dist_f1(metrics, output)
        viz_dists(results, output=output)


if __name__ == "__main__":
    CLI(main)
