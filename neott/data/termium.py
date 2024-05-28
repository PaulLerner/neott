#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from pathlib import Path

import json


def get_data():
    root = Path("data/termium/")
    data = []

    for path in root.rglob("*.csv"):
        print(path)
        domain = path.parent.name.split("-")[-1]
        table = pd.read_csv(path).dropna(subset=["TERM_EN", "TERME_FR"])
        for i, row in table.iterrows():
            if row.TEXTUAL_SUPPORT_1_EN == row.TEXTUAL_SUPPORT_1_EN and row.TEXTUAL_SUPPORT_1_EN.startswith("DEF:"):
                en_def = row.TEXTUAL_SUPPORT_1_EN[len("DEF:"):].strip()
            else:
                en_def = None
            if row.JUSTIFICATION_1_FR == row.JUSTIFICATION_1_FR and row.JUSTIFICATION_1_FR.startswith("DEF:"):
                fr_def = row.JUSTIFICATION_1_FR[len("DEF:"):].strip()
            else:
                fr_def = None

            term = {
                "id": f"{path.name}_{i}",
                "Dom": [domain],
                "S-dom": [row.SUBJECT_EN],
                "en": {
                    "text": row.TERM_EN,
                    "syn": row.SYNONYMS_EN.split(";") if row.SYNONYMS_EN == row.SYNONYMS_EN else None,
                    "def": {"text": en_def},
                },
                "fr": {
                    "text": row.TERME_FR,
                    "syn": row.SYNONYMES_FR.split(";") if row.SYNONYMES_FR == row.SYNONYMES_FR else None,
                    "def": {"text": fr_def}
                }
            }
            data.append(term)

    print(len(data))

    return data


if __name__ == '__main__':
    data = get_data()
    with open("data/termium.json", "wt") as file:
        json.dump(data, file)
