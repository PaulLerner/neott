#!/usr/bin/env python
# coding: utf-8
import json
import os
import subprocess
import warnings
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError
from tqdm import tqdm
import re
from jsonargparse import CLI

import spacy

from .labels import MorphLabel
from ..utils import Path

SPACY_2_DERIF = {'NOUN': 'NOM',
                 'ADJ': 'ADJ',
                 'ADV': 'ADV',
                 'VERB': 'VERBE'}

DERIF_2_NEOTT = {
    'suf': MorphLabel.Suffix.name,
    'pre': MorphLabel.Prefix.name,
    'comp': MorphLabel.Neoclassical.name
}


def tag(tagger, predictions, derif_input_path, indices_path, pos_path):
    if derif_input_path.exists() and indices_path.exists() and pos_path.exists():
        with open(indices_path, 'rt') as file:
            indices = json.load(file)
        with open(pos_path, 'rt') as file:
            spacy_poses = json.load(file)
        return indices, spacy_poses

    indices, lemmas, poses = [], [], []
    spacy_poses = []
    for i, prediction in enumerate(tqdm(predictions, desc="POS tagging")):
        assert len(prediction) == 1
        pos = []
        for token in tagger(prediction[0].strip()):
            pos.append(token.pos_)
            if token.pos_ not in SPACY_2_DERIF:
                continue
            lemma = token.lemma_.replace("œ", "oe")
            if "&" in lemma:
                print(f"Skipping {lemma}")
                continue
            try:
                lemma.encode("ISO-8859-1")
            except UnicodeEncodeError as e:
                print(f"UnicodeEncodeError {e}: {lemma}")
                continue
            indices.append(i)
            lemmas.append(lemma)
            poses.append(SPACY_2_DERIF[token.pos_])
        spacy_poses.append(pos)

    with open(indices_path, 'wt') as file:
        json.dump(indices, file)
    with open(pos_path, 'wt') as file:
        json.dump(spacy_poses, file)
    with open(derif_input_path, "wt", encoding="ISO-8859-1") as file:
        file.write("\n".join(f"{lemma},{pos}" for lemma, pos in zip(lemmas, poses)))

    return indices, spacy_poses


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


def get_leaf_morph(pos, morph_derif, native_compound):
    if len(pos) > 1:
        if {'ADP', 'DET', 'CCONJ'} & set(pos):
            return [MorphLabel.Syntagm.name]
        if native_compound.search(" ".join(pos)) is not None:
            return [MorphLabel.Compound.name]
        return [MorphLabel.Syntagm.name]
    if morph_derif and morph_derif[0]:
        assert len(morph_derif) < 2
        # filter conv
        if morph_derif[0][0] in DERIF_2_NEOTT:
            return [DERIF_2_NEOTT[morph_derif[0][0]]]
        elif len(morph_derif[0]) > 1 and morph_derif[0][1] in DERIF_2_NEOTT:
            return [DERIF_2_NEOTT[morph_derif[0][1]]]
    return []


def derifize(tagger, predictions, root_path, i=0):
    derif_input_path = root_path / f"{i}_derif_input.txt"
    indices_path = root_path / f"{i}_lemma_indices.json"
    pos_path = root_path / f"{i}_pos.json"
    indices, poses = tag(tagger, predictions, derif_input_path, indices_path, pos_path)

    derif_output_path = root_path / f"{i}_derif_out.xml"
    if not derif_output_path.exists():
        derif_path = os.environ.get("DERIF_PATH", "/home/paul/code/derif")
        subprocess.call(["perl",
                         "-I",
                         derif_path,
                         f"{derif_path}/derif.pl",
                         f"--entree={derif_input_path}",
                         f"--sortie={derif_output_path}",
                         "--format=xml"
                         ])

    derif_morphs = flatten_xml(indices, derif_output_path, len(predictions))
    leaf_morphs, leaf_morphs_coarse = [], []
    # noun noun compounds OR verb noun compounds OR verb verb compounds (always with optional punct like dash N-N)
    native_compound = re.compile("((PROPN|NOUN|VERB) (PUNCT )?(PROPN|NOUN))|((VERB) (PUNCT )?VERB)")
    for pos, derif_morph in zip(poses, derif_morphs):
        leaf_morph = get_leaf_morph(pos, derif_morph, native_compound)
        leaf_morphs.append(leaf_morph)
        # group Affixation and Neoclassical compounds + override Derif for monolexical terms that could not be analyzed
        if (not leaf_morph) or leaf_morph[0] in {MorphLabel.Suffix.name, MorphLabel.Prefix.name, MorphLabel.Neoclassical.name}:
            leaf_morphs_coarse.append([MorphLabel.Neoaffix.name])
        else:
            leaf_morphs_coarse.append(leaf_morph)
    morphs = {
        "derif_morph": derif_morphs,
        "leaf_morph": leaf_morphs,
        "leaf_morph_coarse": leaf_morphs_coarse
    }
    return morphs, poses


def main(data_path: Path, do_syn: bool = False):
    print(f"{spacy.prefer_gpu()=}")
    tagger = spacy.load("fr_dep_news_trf", disable=["ner"])
    with open(data_path, 'rt') as file:
        data = json.load(file)
    for name, subset in data.items():
        predictions = [[item["fr"]["text"]] for item in subset]
        morphs, poses = derifize(tagger, predictions, data_path.parent, name)
        for i, (item, pos) in enumerate(zip(subset, poses)):
            item["fr"]["pos"] = pos
            for k, morph in morphs.items():
                item["fr"][k] = morph[i]
        if do_syn:
            indices, syns = [], []
            for i, item in enumerate(subset):
                for j, syn in enumerate(item["fr"]["syn"]):
                    indices.append((i, j))
                    syns.append([syn["text"]])
            morphs, poses = derifize(tagger, syns, data_path.parent, "syn_"+name)
            for l, ((i, j), pos) in enumerate(zip(indices, poses)):
                subset[i]["fr"]["syn"][j]["pos"] = pos
                for k, morph in morphs.items():
                    subset[i]["fr"]["syn"][j][k] = morph[l]
    with open(data_path, 'wt') as file:
        json.dump(data, file)


if __name__ == "__main__":
    CLI(main)
