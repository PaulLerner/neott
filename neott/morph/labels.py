#!/usr/bin/env python
# coding: utf-8
import itertools
from collections import Counter
from jsonargparse import CLI
import enum
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


class MorphLabel(enum.Enum):
    Prefix = 0
    Suffix = 1
    Neoclassical = 2
    Compound = 3
    Syntagm = 4
    # Affixation | Neoclassical
    Neoaffix = 5


def get_morphy(root_path, lang):
    morphyd = pd.read_csv(root_path / f"{lang}/{lang}.derivational.v1.tsv", delimiter="\t",
                          names=["source word", "target word", "source POS", "target POS", "morpheme", "type"])
    morphyi = pd.read_csv(root_path / f"{lang}/{lang}.inflectional.v1.tsv", delimiter="\t",
                          names=["source word", "target word",
                                 'morphological features',
                                 'morpheme segmentation'])

    morphy = {}
    for target_word, word in morphyi.groupby("target word"):
        morphy[target_word] = word
    for target_word, word in morphyd.groupby("target word"):
        morphy[target_word] = word

    morphy_affixes = {}
    for t, subset in morphyd.groupby("type"):
        morphy_affixes[t] = subset
    morphy_prefixes = Counter(morphy_affixes["prefix"].morpheme)
    morphy_suffixes = Counter(morphy_affixes["suffix"].morpheme)
    return morphy, morphy_prefixes, morphy_suffixes


def get_sigmorphon(root_path, lang):
    # load and preprocess sigmorphon
    train = pd.read_csv(root_path / f"{lang}.word.train.tsv", delimiter="\t", names=["word", "morpheme", "process"],
                        dtype=str)
    dev = pd.read_csv(root_path / f"{lang}.word.dev.tsv", delimiter="\t", names=["word", "morpheme", "process"],
                      dtype=str)
    test = pd.read_csv(root_path / f"{lang}.word.test.gold.tsv", delimiter="\t", names=["word", "morpheme", "process"],
                       dtype=str)
    sigmorph = pd.concat((train, dev, test)).dropna()

    process = [[], [], []]
    for c in sigmorph.process:
        for i, value in enumerate(c):
            process[i].append(True if value == "1" else False)
    for i, k in enumerate(["Inflection", "Derivation", MorphLabel.Native.name]):
        sigmorph[k] = process[i]

    sigmorph_per_word = {}
    for target_word, word in sigmorph.groupby("word"):
        # all entries are unique in sigmorph
        sigmorph_per_word[target_word] = word.iloc[0]

    return sigmorph, sigmorph_per_word


def morphemes_are_affixes(morphemes, morphy_prefixes, morphy_suffixes):
    return all(m in morphy_prefixes or m in morphy_suffixes for m in morphemes)


def is_neoclassical(word, memory, sigmorph_per_word, morphy, morphy_prefixes, morphy_suffixes):
    memory.add(word.word)
    morphemes = word.morpheme.split(" @@")
    # inflection comes after suffixation
    if word.Inflection:
        morphemes.pop()
    p_freq = morphy_prefixes.get(morphemes[0], 0)
    s_freq = morphy_suffixes.get(morphemes[-1], 0)

    # the prefix is more frequent than the suffix OR (both equal or UNK but prefix shorter than suffix)
    if (p_freq > s_freq) or ((p_freq == s_freq) and (len(morphemes[0]) < len(morphemes[-1]))):
        prefix = True
        suffix = False
    else:
        prefix = False
        suffix = True

    if len(morphemes) < 3:
        pass
    elif word.word in morphy:
        for source in morphy[word.word]["source word"]:
            if source in sigmorph_per_word and source not in memory:
                source_neo, _, _ = is_neoclassical(sigmorph_per_word[source], memory, sigmorph_per_word, morphy,
                                                   morphy_prefixes, morphy_suffixes)
                if source_neo:
                    return True, prefix, suffix
                elif morphemes_are_affixes(morphemes, morphy_prefixes, morphy_suffixes):
                    return True, False, False
                # 3 or more morphemes but not neo -> prefix and suffix
                elif len(morphemes) > 2:
                    return False, True, True
                else:
                    return False, prefix, suffix
        # did not find source word in sigmorph_per_word

    if morphemes_are_affixes(morphemes, morphy_prefixes, morphy_suffixes):
        return True, False, False

    return False, prefix, suffix


def process_data(sigmorph, output, verbose: int = 0, **kwargs):
    neoclassicals, prefixes, suffixes = [], [], []
    for _, word in sigmorph.iterrows():
        memory = set()
        # some neoclassicals are hidden in derivation
        if word.Derivation:
            is_neo, prefix, suffix = is_neoclassical(word, memory, **kwargs)
            prefixes.append(prefix)
            suffixes.append(suffix)
            neoclassicals.append(is_neo)
        else:
            neoclassicals.append(False)
            prefixes.append(False)
            suffixes.append(False)
    sigmorph[MorphLabel.Neoclassical.name] = neoclassicals
    sigmorph[MorphLabel.Prefixed.name] = prefixes
    sigmorph[MorphLabel.Suffixed.name] = suffixes

    for k in MorphLabel:
        print(k, "&", len(sigmorph[sigmorph[k.name]]), r"\\")

    multi_label = Counter()
    for _, row in sigmorph.iterrows():
        multi_label[(row[MorphLabel.Native.name], row[MorphLabel.Neoclassical.name], row[MorphLabel.Prefixed.name], row[MorphLabel.Suffixed.name])] += 1
    print(" & ".join(l.name for l in MorphLabel) + r" & Count \\")
    for k in itertools.product([False, True], repeat=4):
        v = multi_label[k]
        print(" & ".join(["X" if b else " " for b in k]), "&", v, r"\\")

    if verbose:
        for k in MorphLabel:
            print(k, "\n", sigmorph[sigmorph[k.name]].sample(verbose))

    sigmorph.to_csv(output, sep="\t")
    return sigmorph


def to_fasttext_subset(data):
    fasttext_data = []
    for _, word in data.iterrows():
        example = []
        mono_or_inflection = True
        for k in MorphLabel:
            if word[k.name]:
                mono_or_inflection = False
                example.append(f"__label__{k.name}")
        # OPTION 2: remove __label__MonoOrInflection: will not be used as negative
        # if mono_or_inflection:
        #    continue
        # OPTION 3: use __label__MonoOrInflection as positive class
        #      example.append("__label__MonoOrInflection")
        example.append(word.word)
        fasttext_data.append(" ".join(example))
    return fasttext_data


def to_fasttext(train, dev, test, output, lang):
    for name, subset in [("train", train), ("test", test), ("dev", dev)]:
        fasttext_subset = to_fasttext_subset(subset)
        with open(output/f"{lang}_{name}.txt", "wt") as file:
            file.write("\n".join(fasttext_subset))


def main(sigmorph: Path, morphynet: Path, lang: str, output: Path, verbose: int = 0):
    """Build multi-label classification dataset from SIGMORPHON 2022 and MorphyNet"""
    output.mkdir(exist_ok=True)
    lang = {"en": "eng", "fr": "fra"}[lang]
    morphy, morphy_prefixes, morphy_suffixes = get_morphy(morphynet, lang)
    sigmorph, sigmorph_per_word = get_sigmorphon(sigmorph, lang)
    data = process_data(sigmorph, output/f"{lang}.tsv", sigmorph_per_word=sigmorph_per_word, morphy=morphy,
                        morphy_prefixes=morphy_prefixes, morphy_suffixes=morphy_suffixes, verbose=verbose)

    # OPTION 2: remove __label__MonoOrInflection: will not be used as negative
    # data = data[data.Compound|data.Neoclassical|data.Prefix|data.Suffix]
    train, devtest = train_test_split(data, test_size=0.2)
    dev, test = train_test_split(devtest, test_size=0.5)

    to_fasttext(train, dev, test, output, lang)


if __name__ == "__main__":
    CLI(main, description=main.__doc__)
