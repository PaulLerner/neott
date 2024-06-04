#!/usr/bin/env python
# coding: utf-8
import json
import subprocess
import xml.etree.ElementTree as ET

from tqdm import tqdm

SPACY_2_DERIF = {'NOUN': 'NOM',
                 'ADJ': 'ADJ',
                 'ADV': 'ADV',
                 'VERB': 'VERBE'}
DERIF_PATH = "/home/paul/code/derif"


def tag(tagger, predictions, derif_input_path, indices_path):
    if derif_input_path.exists() and indices_path.exists():
        with open(indices_path, 'rt') as file:
            indices = json.load(file)
        return indices

    indices, lemmas, poses = [], [], []
    for i, prediction in enumerate(tqdm(predictions, desc="POS tagging")):
        assert len(prediction) == 1
        for token in tagger(prediction[0]):
            if token.pos_ not in SPACY_2_DERIF:
                continue
            lemma = token.lemma_.replace("œ", "oe")
            try:
                lemma.encode("ISO-8859-1")
            except UnicodeEncodeError as e:
                print(f"UnicodeEncodeError {e}: {lemma}")
                continue
            indices.append(i)
            lemmas.append(lemma)
            poses.append(SPACY_2_DERIF[token.pos_])

    with open(indices_path, 'wt') as file:
        json.dump(indices, file)
    with open(derif_input_path, "wt", encoding="ISO-8859-1") as file:
        file.write("\n".join(f"{lemma},{pos}" for lemma, pos in zip(lemmas, poses)))

    return indices


def flatten_xml(indices, derif_output_path, num_pred):
    tree = ET.parse(derif_output_path)
    root = tree.getroot()
    morphs = [[] for _ in range(num_pred)]
    for i, lemma in enumerate(root):
        derif_morph = []
        for analysis in lemma.find("Analyses"):
            for step in analysis.find("Steps"):
                morph = step.find("MorphologicalProcessType")
                derif_morph.append(morph.text)
        morphs[indices[i]].append(derif_morph)
    return morphs


def derifize(tagger, predictions, root_path, i=0):
    derif_input_path = root_path / f"{i}_derif_input.txt"
    indices_path = root_path / f"{i}_lemma_indices.json"
    indices = tag(tagger, predictions, derif_input_path, indices_path)

    derif_output_path = root_path / f"{i}_derif_out.xml"
    if not derif_output_path.exists():
        subprocess.call(["perl",
                          "-I",
                          DERIF_PATH,
                          f"{DERIF_PATH}/derif.pl",
                          f"--entree={derif_input_path}",
                          f"--sortie={derif_output_path}",
                          "--format=xml"
                         ])

    morphs = flatten_xml(indices, derif_output_path, len(predictions))
    return morphs
