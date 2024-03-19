#!/usr/bin/env python
# coding: utf-8


import json
import pandas as pd


def get_data():
    data = pd.read_csv("../data/TAL/terms/Terminologie_TAL/TerminologieTAL1.csv")

    data = data.dropna(subset=["skos:prefLabel@fr", "skos:prefLabel@en"])

    # sum(x.lower() == y.lower() for x,y in zip(data["skos:hiddenLabel@fr"], data["skos:prefLabel@fr"]))

    def get_if_not_nan(row, key):
        if row[key] == row[key]:
            return row[key]
        return None

    triples = []
    for i, row in data.iterrows():
        fr_terms = get_if_not_nan(row, "skos:altLabel@fr")
        fr_terms = fr_terms.split("##") if fr_terms is not None else None
        en_terms = get_if_not_nan(row, "skos:altLabel@en")
        en_terms = en_terms.split("##") if en_terms is not None else None
        triples.append({
            "Dom": row.origine,
            "S-dom": None,
            "fr": {
                "text": row["skos:prefLabel@fr"],
                "def": {"text": get_if_not_nan(row, "skos:definition@fr")},
                "syn": fr_terms
            },
            "en": {
                "text": row["skos:prefLabel@en"],
                "def": {"text": get_if_not_nan(row, "skos:definition@en")},
                "syn": en_terms
            },
            "id": row['URI'] if row['URI'] == row['URI'] else f"TAL_{i}"
        })
    return triples


if __name__ == '__main__':
    data = get_data()

    with open("../data/TAL/terms/TAL.json", "wt") as file:
        json.dump(data, file)
