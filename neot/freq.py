# coding: utf-8
import json
from jsonargparse import CLI
from collections import Counter

import datasets
import ahocorasick

    
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


def main(glossary: str, corpus: str, output:str, lang: str = "fr"):
    with open(glossary,"rt") as file:
        glossary = json.load(file)    
    terms = set(item[lang]["text"].lower().strip() for item in glossary)
    automaton = build_automaton(terms)
    corpus = datasets.load_from_disk(corpus)
    counter = Counter()
    corpus.map(count, input_columns="text", batched=True, fn_kwargs=dict(automaton=automaton, counter=counter))
    with open(output, "wt") as file:
        json.dump(counter, file)
    
    
if __name__ == "__main__":
    CLI(main)
