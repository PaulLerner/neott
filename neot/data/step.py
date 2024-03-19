#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import json


def get_data():
    data = pd.read_csv("../data/STEP/terms/tsv/ARTES_glossaire_STEP_2023.tsv", delimiter="\t")

    # sum([len(t.split("   -   ")) for t in data["Equivalents français"] if len(t.split("   -   ")) > 1])

    triples = []
    for i, row in data.iterrows():
        fr_terms = row["Equivalents français"].split("   -   ")
        fr_term = fr_terms.pop(0)
        en_terms = row["Terme anglais"].split("   -   ")
        en_term = en_terms.pop(0)
        triples.append({
            "Dom": None,
            "S-dom": None,
            "fr": {
                "text": fr_term,
                "def": {"text": None},
                "syn": fr_terms
            },
            "en": {
                "text": en_term,
                "def": {"text": None},
                "syn": en_terms
            },
            "id": str(i)
        })

    return triples


if __name__ == '__main__':
    data = get_data()
    with open("../data/STEP/terms/step.json", "wt") as file:
        json.dump(data, file)
