#!/usr/bin/env python
# coding: utf-8


import json
import pandas as pd
import re

def get_data():
    data = pd.read_csv("/home/paul/code/WP1-Resources-and-Corpora/TerminologieTAL/TerminologieTALpoc.csv",delimiter=";")

    data = data.dropna(subset=["skos:prefLabel@fr", "skos:prefLabel@en"])

    def get_if_not_nan(row, key):
        if row[key] == row[key]:
            return row[key]
        return None

    def get_fr(string):
        if string is None:
            return None

        assert string.startswith('"')
        assert string.endswith('"@fr')
        return string[1: -4]

    def get_alt_fr(string):
        return re.findall(r'"(.+?)"@fr', string) if '##' not in string else re.split('##', get_fr(string))

    def get_en(string):
        if string is None:
            return None
        assert string.startswith('"')
        assert string.endswith('"@en')
        return string[1: -4]

    def get_alt_en(string):
        return re.findall(r'"(.+?)"@en', string) if '##' not in string else re.split('##', get_en(string))

    triples = []
    for i, row in data.iterrows():
        fr_terms = get_if_not_nan(row, "skos:altLabel@fr")
        fr_terms = [{"text": fr_term} for fr_term in get_alt_fr(fr_terms)] if fr_terms is not None else None
        en_terms = get_if_not_nan(row, "skos:altLabel@en")
        en_terms = [{"text": en_term} for en_term in get_alt_en(en_terms)] if en_terms is not None else None
        triples.append({
            "Dom": None,
            "S-dom": None,
            "fr": {
                "text": get_fr(row["skos:prefLabel@fr"]),
                "def": {"text": get_fr(get_if_not_nan(row, "skos:definition@fr"))},
                "syn": fr_terms
            },
            "en": {
                "text": get_en(row["skos:prefLabel@en"]),
                "def": {"text": get_en(get_if_not_nan(row, "skos:definition@en"))},
                "syn": en_terms
            },
            "id": f"TAL_{i}"
        })
    return triples


if __name__ == '__main__':
    data = get_data()

    with open("./data/TAL/terms/TAL_poc.json", "wt") as file:
        json.dump({"test": data}, file)
