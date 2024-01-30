from collections import Counter

import re
import nltk


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
        self.words = re.compile("\w+")

    def __call__(self, text):
        text = text.lower()
        text = self.stopwords.sub(" ", text)
        # removes punct + duplicated spaces (keeps letters and digits)
        text = " ".join(self.words.findall(text))
        return text


def em(pred, tgt):
    return pred == tgt


def f1(pred, tgt):
    pred_tokens = pred.split()
    tgt_tokens = tgt.split()
    common = Counter(pred_tokens) & Counter(tgt_tokens)
    tp = sum(common.values())
    if tp == 0:
        return 0
    precision = tp / len(pred_tokens)
    recall = tp / len(tgt_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_metrics(predictions, targets, preproc):
    ems, f1s = [], []
    for pred, tgt in zip(predictions, targets):
        pred, tgt = preproc(pred), preproc(tgt)
        ems.append(em(pred, tgt))
        f1s.append(f1(pred, tgt))
    return {
        "em": sum(ems) / len(ems),
        "f1": sum(f1s) / len(f1s),
        "ems": ems,
        "f1s": f1s
    }
