#!/usr/bin/env python
# coding: utf-8

import json
from collections import Counter
import enum
from jsonargparse import CLI
from tqdm import tqdm

import pandas as pd
import seaborn as sns

from .classes import Term, Inflected, Prefixed, Suffixed, Converted, Native, Neoclassical, Syntagm
from ..utils import random_data, Path


class MorphyType(enum.Enum):
    prefix = 0
    suffix = 1


def get_morphy_table(root_path, lang="eng"):
    table = pd.read_csv(root_path / f"{lang}/{lang}.derivational.v1.tsv", delimiter="\t",
                        names=["source word", "target word", "source POS", "target POS", "morpheme", "type"])

    morphy_affixes = {}
    for t, subset in table.groupby("type"):
        morphy_affixes[MorphyType[t]] = subset

    morphy_prefixes = Counter(morphy_affixes[MorphyType.prefix].morpheme)

    morphy_suffixes = Counter(morphy_affixes[MorphyType.suffix].morpheme)
    return morphy_prefixes, morphy_suffixes


def get_morph_table(root_path, morphy_root_path, lang="en"):
    lang = {"en": "eng", "fr": "fra"}[lang]
    # load and preprocess sigmorphon
    train = pd.read_csv(root_path / f"{lang}.word.train.tsv", delimiter="\t", names=["word", "morpheme", "process"],
                        dtype=str)
    dev = pd.read_csv(root_path / f"{lang}.word.dev.tsv", delimiter="\t", names=["word", "morpheme", "process"],
                      dtype=str)
    test = pd.read_csv(root_path / f"{lang}.word.test.gold.tsv", delimiter="\t", names=["word", "morpheme", "process"],
                       dtype=str)
    table = pd.concat((train, dev, test))

    process = [[], [], []]
    for c in table.process:
        for i, value in enumerate(c):
            process[i].append(True if value == "1" else False)
    for i, k in enumerate(["Inflection", "Derivation", "Compound"]):
        table[k] = process[i]

    per_target = {}
    for target, sources in tqdm(table.groupby("word"), desc="Building morph"):
        per_target[target.lower().strip()] = sources

    prefixes, suffixes = get_morphy_table(morphy_root_path, lang)

    return per_target, prefixes, suffixes


def maybe_rec_get_morph(morphemes, original_token, front, from_morphemes=True, memory=None, *args, **kwargs):
    if front:
        morpheme = morphemes.pop(0)
    else:
        morpheme = morphemes.pop()
    if from_morphemes:
        compound = "".join(morphemes)
    else:
        if front:
            compound = original_token[len(morpheme):]
        else:
            compound = original_token[:-len(morpheme)]
            # avoids infinite recursion
    if compound not in memory:
        compound = get_morph(compound, *args, memory=memory, **kwargs)
    else:
        compound = Term(compound, trust=0.0)

        # retry with original token form
    if from_morphemes and compound.trust == 0.0:
        maybe_rec_get_morph(morphemes, original_token, front, from_morphemes=False, memory=memory, *args, **kwargs)

    return compound, morpheme


def get_compound(token, morphemes, **kwargs):
    if len(morphemes) > 2:
        r, l = maybe_rec_get_morph(morphemes, token, front=True, **kwargs)
        l = Term(l, trust=0.5)
    else:
        l, r = morphemes
        l, r = Term(l, trust=0.5), Term(r, trust=0.5)
    return l, r


def is_neo_classical(morphemes, pos, prefixes, suffixes):
    return pos not in {"ADP", "DET"} and all((m in prefixes or m in suffixes) for m in morphemes)


def get_morph(token, pos=None, per_target=None, prefixes=None, suffixes=None, memory=None):
    if token not in per_target:
        return Term(term=token, pos=pos, trust=0.0)

    memory.add(token)
    # empirically, there is always a single option
    morph = per_target[token].iloc[0]
    morphemes = morph.morpheme.split(" @@")

    # assumes that Inflection is always the right-most process
    if morph.Inflection:
        compound, inflection = maybe_rec_get_morph(morphemes, token, front=False, per_target=per_target,
                                                   prefixes=prefixes, suffixes=suffixes, memory=memory)
        term = Inflected(term=token, inflection=inflection, stem=compound, trust=0.5)
    # before derivation
    elif morph.Derivation:
        # catch neoclassical hidden in derivation
        if is_neo_classical(morphemes, pos, prefixes, suffixes):
            l, r = get_compound(token, morphemes, per_target=per_target, prefixes=prefixes, suffixes=suffixes,
                                memory=memory)
            term = Neoclassical(term=token, stem_l=l, stem_r=r, pos=pos, trust=0.5)
            # back to real affixes
        else:
            p_freq = prefixes.get(morphemes[0], 0)
            s_freq = suffixes.get(morphemes[-1], 0)
            # the prefix is more frequent than the suffix OR (both equal or UNK but prefix shorther than suffix)
            if (p_freq > s_freq) or ((p_freq == s_freq) and (len(morphemes[0]) < len(morphemes[-1]))):
                compound, prefix = maybe_rec_get_morph(morphemes, token, front=True, per_target=per_target,
                                                       prefixes=prefixes, suffixes=suffixes, memory=memory)
                term = Prefixed(term=token, prefix=prefix, stem=compound, pos=pos, trust=0.5)
            else:
                compound, suffix = maybe_rec_get_morph(morphemes, token, front=False, per_target=per_target,
                                                       prefixes=prefixes, suffixes=suffixes, memory=memory)
                term = Suffixed(term=token, suffix=suffix, stem=compound, pos=pos, trust=0.5)
    # and before Compounding (last process before root)
    elif morph.Compound:
        # catch neoclassical hidden in native compounds
        neo_classical = is_neo_classical(morphemes, pos, prefixes, suffixes)
        l, r = get_compound(token, morphemes, per_target=per_target, prefixes=prefixes, suffixes=suffixes,
                            memory=memory)
        if neo_classical:
            term = Neoclassical(term=token, stem_l=l, stem_r=r, pos=pos, trust=0.5)
        else:
            term = Native(term=token, stem_l=l, stem_r=r, pos=pos, trust=0.5)
            # monomorpheme
    else:
        assert len(morphemes) == 1
        term = Term(token, pos=pos, trust=0.5)

    return term


def parse_data(data, per_target, lang="en", **kwargs):
    for item in tqdm(data, desc="Morph parsing"):
        term = item[lang]["text"].lower().strip()
        # init memory to avoid infinite recursion
        memory = set()
        # prioritize morph tokenization over spacy's tokenization (equivalent for single-token terms)
        if term in per_target:
            pos = item[lang].get("pos", [None])[0]
            term = get_morph(term, pos, per_target=per_target, memory=memory, **kwargs)
        else:
            tokens = []
            for i, token in enumerate(item[lang]["tokens"]):
                pos = item[lang]["pos"][i] if "pos" in item[lang] else None
                token = token.lower().strip()
                tokens.append(get_morph(token, pos, per_target=per_target, memory=memory, **kwargs))
            term = Syntagm(terms=tokens, term=item[lang]["text"])
        # TODO .to_dict() to save to JSON
        item[lang]["morph"] = term


def save(data, data_path):
    for item in data:
        item["en"]["morph"] = item["en"]["morph"].to_dict()
        item["fr"]["morph"] = item["fr"]["morph"].to_dict()

    with open(data_path, "wt") as file:
        json.dump(data, file)


def viz_mono(data, lang, viz_path):
    tuples = Counter()
    trusts_terms = Counter()
    trusts_morphs = Counter()
    signatures = Counter()
    morphemes_l = []
    for item in data:
        morph = item[lang]["morph"]
        trusts_terms[morph.trust > 0.0] += 1
        if morph.trust <= 0.0:
            continue
        signatures[morph.signature()] += 1
        labels = morph.labels()
        tuples[tuple(sorted(labels))] += 1
        morphemes_l.append({"length": len(morph), "unit": "morpheme"})
        morphemes_l.append({"length": len(item[lang]["tokens"]), "unit": "word"})
        if not isinstance(morph, Syntagm):
            morph = [morph]
        for m in morph:
            trusts_morphs[m.trust > 0.0] += 1

    print(lang)
    print(f"{trusts_terms=} {trusts_terms[True] / sum(trusts_terms.values()):.1%}")
    print(f"{trusts_morphs=} {trusts_morphs[True] / sum(trusts_morphs.values()):.1%}")

    print(f"\n{len(signatures)=}\n{pd.DataFrame(signatures.most_common(100)).to_latex(index=False)}")

    print(f"\n{len(tuples)=}")
    print(pd.DataFrame(tuples.most_common()).to_latex(index=False))

    morphemes_l = pd.DataFrame(morphemes_l)
    fig = sns.displot(morphemes_l, x="length", hue="unit", discrete=True)
    fig.savefig(viz_path / f"{lang}_morph_sig.pdf")

    print()
    for item in random_data(data, 50):
        morph = item[lang]["morph"]
        print(morph, morph.signature())


def viz_bi(data):
    tuples = Counter()
    trusts_terms = Counter()
    signatures = Counter()
    for item in data:
        if not (item["en"]["morph"].trust > 0.0 and item["fr"]["morph"].trust > 0.0):
            trusts_terms[False] += 1
            continue
        trusts_terms[True] += 1
        signatures[f'{item["en"]["morph"].signature()}={item["fr"]["morph"].signature()}'] += 1
        en_labels = " ".join(sorted(item["en"]["morph"].labels()))
        fr_labels = " ".join(sorted(item["fr"]["morph"].labels()))
        tuples[f"{en_labels}={fr_labels}"] += 1
    print("\nEN-FR")
    print(f"{trusts_terms=} {trusts_terms[True] / sum(trusts_terms.values()):.1%}")

    print(f"{len(signatures)=}")
    print(f"{len(tuples)=}\n")

    r_data = list(random_data(data))
    for signature, c in signatures.most_common(100):
        for item in r_data:
            if f'{item["en"]["morph"].signature()}={item["fr"]["morph"].signature()}' == signature:
                print(
                    f'{item["en"]["morph"].signature()} & {item["fr"]["morph"].signature()} & {c} & {item["en"]["text"]} & {item["fr"]["text"]}')
                break
    print()
    for label_tuple, c in tuples.most_common(100):
        for item in r_data:
            en_labels = " ".join(sorted(item["en"]["morph"].labels()))
            fr_labels = " ".join(sorted(item["fr"]["morph"].labels()))
            if f"{en_labels}={fr_labels}" == label_tuple:
                print(f'{en_labels} & {fr_labels} & {c} & {item["en"]["text"]} & {item["fr"]["text"]}')
                break


def main(data_path: Path, sigmorphon: Path, morphynet: Path, viz_path: Path):
    """Morphological parsing"""
    viz_path.mkdir(exist_ok=True)
    with open(data_path, "rt") as file:
        data = json.load(file)
    for lang in ["en", "fr"]:
        per_target, prefixes, suffixes = get_morph_table(sigmorphon, morphynet, lang=lang)
        parse_data(data, per_target=per_target, prefixes=prefixes, suffixes=suffixes, lang=lang)
        viz_mono(data, lang, viz_path)
    viz_bi(data)
    save(data, data_path)


if __name__ == "__main__":
    CLI(main)
