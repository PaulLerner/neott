import warnings
from collections import Counter

import re
import nltk
import numpy as np

from .morph.labels import MorphLabel


def get_stopwords(lang):
    try:
        stopwords = nltk.corpus.stopwords.words(lang)
    except LookupError:
        nltk.download('stopwords')
        return get_stopwords(lang)
    return stopwords


class Preprocessor:
    def __init__(self, lang):
        nltk_lang = {"en": 'english', 'fr': 'french'}[lang]
        stopwords = '|'.join(get_stopwords(nltk_lang))
        self.stopwords = re.compile(rf"\b({stopwords})\b")
        self.words = re.compile(r"\w+")

    def __call__(self, text):
        text = text.lower()
        text = self.stopwords.sub(" ", text)
        # removes punct + duplicated spaces (keeps letters and digits)
        text = " ".join(self.words.findall(text))
        return text


def em(pred, tgt):
    return 1.0 if pred == tgt else 0.0


def f1(pred, tgt):
    pred_tokens = pred.split()
    tgt_tokens = tgt.split()
    common = Counter(pred_tokens) & Counter(tgt_tokens)
    tp = sum(common.values())
    if tp == 0:
        return 0.0
    precision = tp / len(pred_tokens)
    recall = tp / len(tgt_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_metrics(predictions, targets, syns, preproc, morphs=None):
    ems, f1s = [], []
    syn_ems, syn_f1s = [], []
    for pred, tgt, syn in zip(predictions, targets, syns):
        if len(pred) > 1:
            warnings.warn(
                f"Recall is deprecated, will only evaluate the first prediction out of {len(pred)}",
                DeprecationWarning
            )
        tgt = preproc(tgt)
        p = preproc(pred[0])
        ems.append(em(p, tgt))
        f1s.append(f1(p, tgt))

        syn_em = ems[-1]
        syn_f1 = f1s[-1]
        for s in syn:
            s = preproc(s)
            syn_em = max(syn_em, em(p, s))
            syn_f1 = max(syn_f1, f1(p, s))
        syn_ems.append(syn_em)
        syn_f1s.append(syn_f1)

    # TODO regroup variants per entry to compute concept-level scores

    all_scores = {
        "em": sum(ems) / len(ems),
        "f1": sum(f1s) / len(f1s),
        "syn_em": sum(syn_ems) / len(syn_ems),
        "syn_f1": sum(syn_f1s) / len(syn_f1s),
        "ems": ems,
        "f1s": f1s,
        "syn_ems": syn_ems,
        "syn_f1s": syn_f1s
    }

    if morphs is not None:
        morph_i = {label.name: [] for label in MorphLabel}
        for i, morph in enumerate(morphs):
            for label in morph:
                morph_i[label].append(i)
        ems = np.array(ems)
        f1s = np.array(f1s)
        syn_ems = np.array(syn_ems)
        syn_f1s = np.array(syn_f1s)
        for label, i in morph_i.items():
            if not i:
                continue
            i = np.array(i, dtype=int)
            all_scores[f"em_{label}"] = ems[i].mean()
            all_scores[f"f1_{label}"] = f1s[i].mean()
            all_scores[f"syn_em_{label}"] = syn_ems[i].mean()
            all_scores[f"syn_f1_{label}"] = syn_f1s[i].mean()

    return all_scores
