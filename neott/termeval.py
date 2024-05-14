#!/usr/bin/env python
# coding: utf-8

from jsonargparse import CLI
import json
from collections import Counter
import pandas as pd

from .utils import Path
from .metrics import Preprocessor
from .freq import build_automaton


class TermPreprocessor(Preprocessor):
    def __call__(self, text):
        return f' {super().__call__(text)} '


def count(string, automaton):
    counter = Counter()
    for end_index, (insert_order, original_value) in automaton.iter(string):
        counter[original_value] += 1
    return counter


def get_terms(lang, preproc, glossary):
    terms = set()
    for item in glossary:
        terms.add(preproc(item[lang]["text"]))
        if item[lang]["syn"] is not None:
            for synonym in item[lang]["syn"]:
                terms.add(preproc(synonym))
    return terms


def evaluate(model_names, root_data, preproc_en, preproc_fr, src_automaton, tgt_automaton, terms):
    recalls = {model: [] for model in model_names}
    src_uniques, tgt_uniques, bi_uniques = [], [], []
    for target_path in root_data.glob("./*/*sent.fr.txt"):
        corpus_name = target_path.name.split("_")[1]
        src_path = target_path.parent / (target_path.name[:-6] + "en.txt")
        with open(target_path, 'rt') as file:
            tgt_corpus = file.read().split("\n")
        with open(src_path, 'rt') as file:
            src_corpus = file.read().split("\n")
        # pred_path = target_path.parent.parent/"TRAD_DeepLPro"/("DeepLPro_"+"_".join(target_path.name.split("_")[:2])+"-doc.sent.sys")
        pred_corpora = {}
        for model in model_names:
            pred_path = list((target_path.parent.parent / model).glob(f"*_{corpus_name}*"))
            assert len(pred_path) == 1, breakpoint()
            with open(pred_path[0], 'rt') as file:
                pred_corpora[model] = file.read().split("\n")
            assert len(tgt_corpus) == len(src_corpus) and len(tgt_corpus) == len(pred_corpora[model]), breakpoint()#(len(tgt_corpus), len(src_corpus), len(pred_corpora[model]))
        for i, (src_item, tgt_item) in enumerate(zip(src_corpus, tgt_corpus)):
            src_item = preproc_en(src_item)
            tgt_item = preproc_fr(tgt_item)
            pred_counters = {}
            for model, pred_corpus in pred_corpora.items():
                pred_item = preproc_fr(pred_corpus[i])
                pred_counters[model] = count(pred_item, tgt_automaton)
            src_counter = count(src_item, src_automaton)
            tgt_counter = count(tgt_item, tgt_automaton)
            src_uniques.append(len(src_counter))
            tgt_uniques.append(len(tgt_counter))

            recall = {model: 0 for model in model_names}
            ground_truths = 0
            for src_term in src_counter:
                ground_truth = False
                for tgt_term in terms[src_term]:
                    # src_term is found in source and tgt_term is found in reference -> ground-truth term (that the model should predict)
                    if tgt_term in tgt_counter:
                        ground_truth = True
                        ground_truths += 1
                        break
                if ground_truth:
                    for model, pred_counter in pred_counters.items():
                        recalled = False
                        for tgt_term in terms[src_term]:
                            if tgt_term in pred_counter:
                                # print(src_term,tgt_term)
                                recall[model] += 1
                                recalled = True
                                break
                      #  if not recalled and model == "TowerBase/":
                       #     print(src_term, terms[src_term], pred_corpora[model][i], "\n")
            bi_uniques.append(ground_truths)
            if ground_truths > 0:
                for model, r in recall.items():
                    recalls[model].append(r / ground_truths)

    for model, recall in recalls.items():
        print(model, sum(recall) / len(recall), len(recall))
        recalls[model] = sum(recall) / len(recall)

    print((pd.DataFrame([recalls]) * 100).T.to_latex(float_format="%.1f"))


def main(glossary: Path, root_data: Path):
    with open(glossary, "rt") as file:
        glossary = json.load(file)
    preproc_fr = TermPreprocessor("fr")
    preproc_en = TermPreprocessor("en")

    src_terms = get_terms("en", preproc_en, glossary)
    tgt_terms = get_terms("fr", preproc_fr, glossary)
    print(f"{len(src_terms)=}, {len(tgt_terms)=}")

    terms = {}
    for item in glossary:
        for src_syn in [item["en"]["text"]] + (item["en"]["syn"] if item["en"]["syn"] is not None else []):
            src_syn = preproc_en(src_syn)
            terms.setdefault(src_syn, set())
            for tgt_syn in [item["fr"]["text"]] + (item["fr"]["syn"] if item["fr"]["syn"] is not None else []):
                terms[src_syn].add(preproc_fr(tgt_syn))

    src_automaton = build_automaton(src_terms)
    tgt_automaton = build_automaton(tgt_terms)

    model_names = """mBART-FTdoc/
    DeepLPro/
    SystranPro/
    TowerBase/
    TowerInstruct/""".split()

    evaluate(model_names, root_data, preproc_en, preproc_fr, src_automaton, tgt_automaton, terms)


if __name__ == "__main__":
    CLI(main, description=main.__doc__)
