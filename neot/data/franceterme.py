#!/usr/bin/env python
# coding: utf-8

from xml.etree import ElementTree as ET
import re
import json


def get_data(tree):
    root = tree.getroot()
    articles = root.findall("Article")
    triples = []
    fem_reg = re.compile(r", -\w+")
    article_reg = re.compile(r" \(.+\)")

    for article in articles:
        def_fr = article.find("Definition").text.strip()
        fr_terms = []
        for term in article.findall("Terme"):
            if "Terme" not in term.attrib:
                continue
            fr = term.attrib["Terme"].strip()
            fr = fem_reg.sub("", fr)
            fr = article_reg.sub("", fr)
            fr_terms.append(fr)

        en_terms = []
        for eq in article.findall("Equivalent"):
            if eq.attrib['langue'] != 'en':
                continue
            for ep in eq.findall("Equi_prop"):
                ep = ep.text.strip()
                en_terms.append(ep)

        if (not def_fr) or (not en_terms):
            continue

        fr_term = fr_terms.pop(0)
        en_term = en_terms.pop(0)

        surdoms = []
        sousdoms = []
        for domaine in article.findall("Domaine"):
            for surdom in domaine.findall("Dom"):
                surdoms.append(surdom.text.strip())
            for sousdom in domaine.findall("S-dom"):
                sousdoms.append(sousdom.text)

        triples.append({
            "Dom": surdoms,
            "S-dom": sousdoms,
            "fr": {
                "text": fr_term,
                "def": {"text": def_fr},
                "syn": fr_terms
            },
            "en": {
                "text": en_term,
                "syn": en_terms
            },
            "id": article.attrib['numero']
        })

    return triples


if __name__ == '__main__':
    tree = ET.parse('data/FranceTerme.xml')
    triples = get_data(tree)
    with open("data/FranceTerme_triples.json", "wt") as file:
        json.dump(triples, file)
