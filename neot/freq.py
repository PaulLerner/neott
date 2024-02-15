# coding: utf-8
import json
import re

from jsonargparse import CLI
from collections import Counter
from tqdm import tqdm
import pandas as pd

import datasets
import ahocorasick

from .utils import Path


def build_automaton(terms):
    automaton = ahocorasick.Automaton()
    for idx, key in enumerate(terms):
        automaton.add_word(key, (idx, key))
    automaton.make_automaton()
    return automaton


def count(batch, automaton, counter, preproc):
    string = preproc(" ".join(batch))
    for end_index, (insert_order, original_value) in automaton.iter(string):
        counter[original_value] += 1


class Preprocessor:
    def __init__(self, whole_word: bool = False):
        self.whole_word = whole_word
        if self.whole_word:
            self.words = re.compile(r"\w+")

    def __call__(self, text):
        text = text.lower()
        # removes punct + duplicated spaces (keeps letters and digits)
        if self.whole_word:
            text = " ".join(self.words.findall(text))
        return text


def main(glossary: str, corpus: Path, output: str, lang: str = "fr", hf: bool = True, batch_size: int = 100000,
         whole_word: bool = False):
    """Compute the frequency of terms in glossary on a given corpus"""
    with open(glossary, "rt") as file:
        glossary = json.load(file)
    preproc = Preprocessor(whole_word)
    # add space around word
    if whole_word:
        terms = set(f' {item[lang]["text"].lower().strip()} ' for subset in glossary.values() for item in subset)
    else:
        terms = set(item[lang]["text"].lower().strip() for subset in glossary.values() for item in subset)

    print(f"{len(terms)=}")
    automaton = build_automaton(terms)
    # not necessary with Counter but eases downstream analysis
    counter = Counter({term: 0 for term in terms})
    if hf:
        corpus = datasets.load_from_disk(corpus)
        corpus.map(count, input_columns="text", batched=True, batch_size=batch_size,
                   fn_kwargs=dict(automaton=automaton, counter=counter, preproc=preproc))
    elif corpus.is_file():
        with open(corpus, 'rt') as file:
            texts = file.read().split("\n")
        for b in tqdm(list(range(0, len(texts), batch_size))):
            count(texts[b: b + batch_size], automaton, counter, preproc)
    else:
        for input_path in tqdm(list(corpus.glob('*.jsonl'))):
            texts = pd.read_json(input_path, lines=True).content
            for b in range(0, len(texts), batch_size):
                count(texts[b: b + batch_size], automaton, counter, preproc)
    with open(output, "wt") as file:
        json.dump(counter, file)


if __name__ == "__main__":
    CLI(main, description=main.__doc__)