#!/usr/bin/env python
# coding: utf-8
from typing import Union, List

import matplotlib.pyplot as plt
import json
import seaborn as sns
from collections import Counter

import spacy
import yaml
from jsonargparse import CLI
import editdistance
import pandas as pd

from transformers import AutoTokenizer

from ..utils import Path, load_json_line
from ..morph.labels import MorphLabel
from ..morph.classif import Classifier
from ..morph.derif import derifize
from .freq import main as viz_freq


def dist_f1(metrics, output):
    fig = sns.displot(metrics["f1s"])
    fig.savefig(output / "f1_dist.pdf")


def gather_results(data, metrics, tokenizer=None, morpher=None, freq=None, lang: str = "fr",
                   pred_morphs=None, morph_key: str = "morph_label"):
    results = []

    fr_ova = {c.name: {True: [], False: []} for c in MorphLabel}
    en_ova = {c.name: {True: [], False: []} for c in MorphLabel}
    tps = Counter({c.name: 0 for c in MorphLabel})
    pps, cps = tps.copy(), tps.copy()

    per_dom = []
    for i, item in enumerate(data):
        if freq is not None:
            f = freq[f' {item[lang]["text"].lower().strip()} '] + 1
        else:
            f = None
        p_fr = item.get("fr", {}).get(morph_key, [])
        p_en = item.get("en", {}).get(morph_key, [])
        p_tgt = set(item[lang][morph_key])
        if morph_key == "leaf_morph":
            leaf_morph = item[lang][morph_key][0] if item[lang][morph_key] else None
        else:
            leaf_morph = None
        cps += Counter(item[lang][morph_key])
        em = metrics["ems"][i]

        bi_label = {}
        if morph_key == "morph_label" or p_tgt:
            for label in MorphLabel:
                fr_ova[label.name][label.name in p_fr].append(em)
                en_ova[label.name][label.name in p_en].append(em)
                bi_label[label.name] = label.name in p_tgt
                if pred_morphs is not None:
                    labels = pred_morphs[i]
                    if label.name in labels:
                        pps[label.name] += 1
                        if label.name in p_tgt:
                            tps[label.name] += 1
        if tokenizer is not None:
            term_fertility = len(tokenizer.tokenize(item[lang]["text"]))
            if morpher is not None:
                token_fertility = []
                for token in morpher.tokenizer(item[lang]["text"]):
                    token_fertility.append(len(tokenizer.tokenize(token.text)))
                token_fertility = max(token_fertility)
            else:
                token_fertility = None
        else:
            term_fertility = None
            token_fertility = None
        results.append({
            "Morph. Diff.": len(set(p_en).symmetric_difference(set(p_fr))),
            "EM": em,
            "Edit dist.": editdistance.eval(item.get("fr", {}).get("text", ""), item.get('en', {}).get("text", "")),
            "Term fertility": term_fertility,
            "Word fertility": token_fertility,
            "freq": f,
            "morph": leaf_morph,
            **bi_label
        })
        for dom in item.get("Dom", []):
            per_dom.append({"Domain": dom, "EM": em})

    results = pd.DataFrame(results)
    per_dom = pd.DataFrame(per_dom)
    metrics_per_label = {}
    for k, v in tps.items():
        metrics_per_label[k] = {"precision": v / pps[k] if pps[k] > 0 else 0.0,
                                "recall": v / cps[k] if cps[k] > 0 else 0.0}
    metrics_per_label = pd.DataFrame(metrics_per_label).T
    metrics_per_label['f1'] = (2 * metrics_per_label['precision'] * metrics_per_label['recall']) / (
            metrics_per_label['precision'] + metrics_per_label['recall'])
    print((metrics_per_label * 100).to_latex(float_format='%.1f'))
    for label in fr_ova:
        for label_exists in fr_ova[label]:
            if len(fr_ova[label][label_exists]) == 0:
                fr_ova[label][label_exists] = 0
            else:
                fr_ova[label][label_exists] = sum(fr_ova[label][label_exists]) / len(fr_ova[label][label_exists])
    for label in en_ova:
        for label_exists in en_ova[label]:
            if len(en_ova[label][label_exists]) == 0:
                en_ova[label][label_exists] = 0
            else:
                en_ova[label][label_exists] = sum(en_ova[label][label_exists]) / len(en_ova[label][label_exists])
    fr_ova = pd.DataFrame(fr_ova)
    en_ova = pd.DataFrame(en_ova)

    return results, per_dom, fr_ova, en_ova, metrics_per_label, tps, pps, cps


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


def viz_dists(results, tokenizer=None, **kwargs):
    distributions = ["Morph. Diff.", "Edit dist.", "# words"]
    if tokenizer is not None:
        distributions += ["Term fertility", "Word fertility"]
    for x in distributions:
        viz_dist(results, x, **kwargs)


def viz_ova(fr_ova, en_ova):
    print("EN\n", (en_ova * 100).to_latex(float_format="%.1f"))
    print("FR\n", (fr_ova * 100).to_latex(float_format="%.1f"))


def main(data: Union[Path, dict], preds: Path, tokenizer: str = None, output: Path = None, subset: str = "test",
         morpher: str = None, tagger: str = None, lang: str = "fr", freq_paths: Union[str, List[str]] = None,
         morph_key: str = "morph_label"):
    if not isinstance(data, dict):
        with open(data, "rt") as file:
            data = json.load(file)

    if tokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, add_prefix_space=True, trust_remote_code=True)

    if morpher is not None:
        morpher = Classifier(morpher, lang)

    if tagger is not None:
        print(f"{spacy.prefer_gpu()=}")
        tagger = spacy.load(tagger)

    if freq_paths is not None:
        freq_fig, freq = viz_freq(freq_paths)
    else:
        freq_fig, freq = None, None

    outputs = []
    for i, pred in enumerate(load_json_line(preds)):
        if "hyperparameters" not in pred:
            config_path = preds.parent/"config.yaml"
            if (config_path).exists():
                with open(config_path) as file:
                    pred["hyperparameters"] = yaml.safe_load(file)
        print(pred.get("hyperparameters"))
        metrics = pred["metrics"]
        for k, v in metrics.items():
            if isinstance(v, float):
                print(k, v)
        predictions = pred["predictions"]
        assert len(predictions) == len(data[subset])
        pred_morphs = None

        if morph_key == "morph_label":
            if morpher is not None:
                pred_morphs = [morpher(pred[0].split("\n")[0].strip()) for pred in predictions]
        elif tagger is not None:
            pred_morphs = derifize(tagger, predictions, preds.parent, i)[0][morph_key]

        results, per_dom, fr_ova, en_ova, *more_results = gather_results(
            data[subset], metrics, tokenizer=tokenizer, morpher=morpher, freq=freq, lang=lang,
            pred_morphs=pred_morphs, morph_key=morph_key
        )
        outputs.append((pred, pred_morphs, results, per_dom, fr_ova, en_ova, *more_results))
        viz_ova(fr_ova, en_ova)
        if output is not None:
            output.mkdir(exist_ok=True)
            dist_f1(metrics, output)
            viz_dists(results, output=output, tokenizer=tokenizer)
    return freq_fig, freq, outputs


if __name__ == "__main__":
    CLI(main, description=main.__doc__)
