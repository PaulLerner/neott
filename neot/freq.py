# coding: utf-8
import json
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


def count(batch, automaton, counter):
    string = " ".join(batch).lower()
    for end_index, (insert_order, original_value) in automaton.iter(string):
        counter[original_value] += 1


def main(glossary: str, corpus: Path, output: str, lang: str = "fr", hf: bool = True, batch_size: int = 100000):
    """Compute the frequency of terms in glossary on a given corpus"""
    with open(glossary, "rt") as file:
        glossary = json.load(file)
    terms = set(item[lang]["text"].lower().strip() for subset in glossary.values() for item in subset)
    print(f"{len(terms)=}")
    automaton = build_automaton(terms)
    counter = Counter()
    if hf:
        corpus = datasets.load_from_disk(corpus)
        corpus.map(count, input_columns="text", batched=True, batch_size=batch_size, fn_kwargs=dict(automaton=automaton, counter=counter))
    else:
        for input_path in tqdm(list(corpus.glob('*.jsonl'))):
            texts = pd.read_json(input_path, lines=True).content
            for b in range(0, len(texts), batch_size):
                count(texts[b: b+batch_size], automaton, counter)
    with open(output, "wt") as file:
        json.dump(counter, file)


if __name__ == "__main__":
    CLI(main)
