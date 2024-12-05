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
import lightning as pl
from lightning.cli import LightningCLI

from ..utils import load_json_line
from pathlib import Path
from ..morph.labels import MorphLabel
from ..morph.classif import Classifier
from .freq import main as viz_freq


def dist_f1(metrics, output):
    fig = sns.displot(metrics["f1s"])
    fig.savefig(output / "f1_dist.pdf")


def gather_results(data, metrics, tokenizer=None, morpher=None, freq=None, src: str = "en", tgt: str = "fr",
                   pred_morphs=None, morph_key: str = "morph_label", predictions=None):
    results = []

    tgt_ova = {c.name: {True: [], False: []} for c in MorphLabel}
    src_ova = {c.name: {True: [], False: []} for c in MorphLabel}
    tps = Counter({c.name: 0 for c in MorphLabel})
    pps, cps = tps.copy(), tps.copy()

    per_dom = []
    for i, item in enumerate(data):
        if freq is not None:
            f = freq[f' {item[tgt]["text"].lower().strip()} '] + 1
        else:
            f = None
        p_tgt = item.get(tgt, {}).get(morph_key, [])
        p_src = item.get(src, {}).get(morph_key, [])
        if morph_key != "morph_label":
            tgt_leaf_morph = item[tgt][morph_key][0] if item[tgt][morph_key] else None
            src_leaf_morph = item[src][morph_key][0] if item[src][morph_key] else None
        else:
            tgt_leaf_morph = None
            src_leaf_morph = None
        cps += Counter(item[tgt][morph_key])
        em = metrics["ems"][i]

        src_bi_label, tgt_bi_label = {}, {}
        if morph_key == "morph_label" or p_tgt:
            for label in MorphLabel:
                tgt_ova[label.name][label.name in p_tgt].append(em)
                src_ova[label.name][label.name in p_src].append(em)
                tgt_bi_label[f"tgt-{label.name}"] = label.name in p_tgt
                src_bi_label[f"src-{label.name}"] = label.name in p_src
                if pred_morphs is not None:
                    labels = pred_morphs[i]
                    if label.name in labels:
                        pps[label.name] += 1
                        if label.name in p_tgt:
                            tps[label.name] += 1
        if tokenizer is not None:
            tgt_tokens = tokenizer.tokenize(item[tgt]["text"].strip())
            src_tokens = tokenizer.tokenize(item[src]["text"].strip())
            term_fertility = len(tgt_tokens)
            bpe_ref_ed = editdistance.eval(tgt_tokens, src_tokens)
            if morpher is not None:
                token_fertility = []
                for token in morpher.tokenizer(item[tgt]["text"].strip()):
                    token_fertility.append(len(tokenizer.tokenize(token.text)))
                token_fertility = max(token_fertility)
            else:
                token_fertility = None
        else:
            bpe_ref_ed = None
            term_fertility = None
            token_fertility = None
        ref_ed = editdistance.eval(item.get(tgt, {}).get("text", "").strip(), item.get(src, {}).get("text", "").strip())
        if predictions is not None:
            assert len(predictions[i]) == 1
            pred_ed = editdistance.eval(predictions[i][0].strip(), item[src]["text"].strip())
        else:
            pred_ed = None
        results.append({
            "Morph. Diff.": len(set(p_src).symmetric_difference(set(p_tgt))),
            "EM": em,
            "src-ref edit dist.": ref_ed,
            "src-pred edit dist.": pred_ed,
            "src-ref BPE edit dist.": bpe_ref_ed,
            "Term fertility": term_fertility,
            "Word fertility": token_fertility,
            "freq": f,
            "tgt-morph": tgt_leaf_morph,
            "src-morph": src_leaf_morph,
            **src_bi_label,
            **tgt_bi_label
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
    for label in tgt_ova:
        for label_exists in tgt_ova[label]:
            if len(tgt_ova[label][label_exists]) == 0:
                tgt_ova[label][label_exists] = 0
            else:
                tgt_ova[label][label_exists] = sum(tgt_ova[label][label_exists]) / len(tgt_ova[label][label_exists])
    for label in src_ova:
        for label_exists in src_ova[label]:
            if len(src_ova[label][label_exists]) == 0:
                src_ova[label][label_exists] = 0
            else:
                src_ova[label][label_exists] = sum(src_ova[label][label_exists]) / len(src_ova[label][label_exists])
    tgt_ova = pd.DataFrame(tgt_ova)
    src_ova = pd.DataFrame(src_ova)

    return results, per_dom, tgt_ova, src_ova, metrics_per_label, tps, pps, cps


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


def viz_ova(tgt_ova, src_ova):
    print("EN\n", (src_ova * 100).to_latex(float_format="%.1f"))
    print("FR\n", (tgt_ova * 100).to_latex(float_format="%.1f"))


def main(data: Union[Path, dict], preds: Path, tokenizer: str = None, output: Path = None, subset: str = "test",
         morpher: str = None, tagger: str = None, src: str = "en", tgt: str = "fr", freq_paths: Union[str, List[str]] = None,
         morph_key: str = "morph_label", neoseg_path: str = None):
    if not isinstance(data, dict):
        with open(data, "rt") as file:
            data = json.load(file)

    if tokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, add_prefix_space=True, trust_remote_code=True)

    if morpher is not None:
        morpher = Classifier(morpher, tgt)

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
        elif morph_key == "neoseg_morph" and neoseg_path is not None:
            # convert from pred to dataset format
            p_items = {subset: [{tgt: {"text": p[0].strip()}} for p in predictions]}
            p_items_path = preds.parent/f"{i}_predictions_items.json"
            with open(p_items_path, "wt") as file:
                json.dump(p_items, file)
            # load and run neoseg (based on lightning)
            with open(neoseg_path, 'rt') as file:
                neoseg_config = yaml.safe_load(file)
            neoseg_config["data"]['init_args']["predict_path"] = p_items_path
            neoseg_config = {"predict": neoseg_config}
            LightningCLI(
                args=neoseg_config,
                trainer_class=pl.Trainer,
                seed_everything_default=0
            )
            # load and convert back from dataset to pred
            with open(p_items_path, "rt") as file:
                p_items = json.load(file)
            pred_morphs = [item[tgt]["neoseg_morph"] for item in p_items[subset]]
        elif tagger is not None:
            pred_morphs = derifize(tagger, predictions, preds.parent, i)[0][morph_key]

        results, per_dom, tgt_ova, src_ova, *more_results = gather_results(
            data[subset], metrics, tokenizer=tokenizer, morpher=morpher, freq=freq, src=src, tgt=tgt,
            pred_morphs=pred_morphs, morph_key=morph_key, predictions=predictions
        )
        outputs.append((pred, pred_morphs, results, per_dom, tgt_ova, src_ova, *more_results))
        viz_ova(tgt_ova, src_ova)
        if output is not None:
            output.mkdir(exist_ok=True)
            dist_f1(metrics, output)
            viz_dists(results, output=output, tokenizer=tokenizer)
    return freq_fig, freq, outputs


if __name__ == "__main__":
    CLI(main, description=main.__doc__)
