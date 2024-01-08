#!/usr/bin/env python
# coding: utf-8

import random
import json
from collections import Counter

import pandas as pd

from .morph import Term, Inflected, Prefixed, Suffixed, Converted, Compound, Syntagmatic


def get_morph_table(lang="eng"):
    # load and preprocess sigmorphon
    train=pd.read_csv(f"/home/lerner/open/2022SegmentationST/data/{lang}.word.train.tsv","\t",names=["word","morpheme","process"],dtype=str)
    dev = pd.read_csv(f"/home/lerner/open/2022SegmentationST/data/{lang}.word.dev.tsv","\t",names=["word","morpheme","process"],dtype=str)
    test=pd.read_csv(f"/home/lerner/open/2022SegmentationST/data/{lang}.word.test.gold.tsv","\t",names=["word","morpheme","process"],dtype=str)
    table = pd.concat((train,dev,test))
    
    process=[[],[],[]]
    for c in table.process:
        for i, value in enumerate(c):
            process[i].append(True if value=="1" else False)
    for i, k in enumerate(["Inflection","Derivation","Compound"]):
        table[k] = process[i]
    
    compounds = table[~table.Inflection & ~table.Derivation & table.Compound]
    compounds = Counter(m for morph in compounds.morpheme for m in morph.split(" @@"))
    
    derivations = table[~table.Inflection & table.Derivation & ~table.Compound]
    prefixes = Counter(morph.split(" @@")[0] for morph in derivations.morpheme)
    suffixes = Counter(morph.split(" @@")[-1] for morph in derivations.morpheme)
    
    per_target = {}
    for target, sources in table.groupby("word"):
        per_target[target.lower().strip()]=sources
    
    return per_target, prefixes, suffixes, compounds


def maybe_rec_get_morph(morphemes, original_token):
    compound = "".join(morphemes)
    # avoids infinite recursion
    if compound != original_token:
        compound = get_morph(compound)
    else:
        compound = Term(compound)
    return compound


def get_morph(token, spacy_pos=None, mwe=None, per_target, prefixes, suffixes, compounds):
    if token not in per_target:
        return Term(term=token, pos=spacy_pos)
    # empirically, there is always a single option
    morph = per_target[token].iloc[0]
    morphemes = morph.morpheme.split(" @@")
    
    # assumes that Inflection is always the right-most process
    if morph.Inflection:
        inflection = morphemes.pop()
        compound = maybe_rec_get_morph(morphemes, token)
        term = Inflected(term=token, inflection=inflection, stem=compound)
    # before derivation
    # TODO a lot of Neoclassical are tagged Derivation
    elif morph.Derivation:
        p_freq = prefixes.get(morphemes[0], 0)
        s_freq = suffixes.get(morphemes[-1], 0)
        # the prefix is more frequent than the suffix OR (both equal or UNK but prefix shorther than suffix)
        if (p_freq > s_freq) or ((p_freq == s_freq) and (len(morphemes[0]) < len(morphemes[-1]))):
            prefix = morphemes.pop(0)
            compound = maybe_rec_get_morph(morphemes, token)
            term = Prefixed(term=token, prefix=prefix, stem=compound, pos=spacy_pos)
        else:
            suffix = morphemes.pop()
            compound = maybe_rec_get_morph(morphemes, token)
            term = Suffixed(term=token, suffix=suffix, stem=compound, pos=spacy_pos)
    # and before Compounding (last process before root)
    elif morph.Compound:
        if len(morphemes) > 2:
            # assumes right-headed compound (English and Neoclassical)
            l = Term(morphemes.pop(0))
            r = maybe_rec_get_morph(morphemes, token)
        else:
            l, r = morphemes
            l, r = Term(l), Term(r)
        term = Compound(term=token, stem_l=l, stem_r=r, pos=spacy_pos)        
    # monomorpheme
    else:
        assert len(morphemes) == 1
        term = Term(token, pos=spacy_pos)
               
    return term


def parse_data(data, per_target, **kwargs):
    for item in data:
        en = item["en"]["text"].lower().strip()
        if en in per_target:   
            terms = get_morph(en, item["en"]["pos"][0], False, **kwargs)
        else:
            terms = []
            mwe = len(item["en"]["tokens"]) > 1
            for token, pos in zip(item["en"]["tokens"], item["en"]["pos"]):
                token = token.lower().strip()
                terms.append(get_morph(token, pos, mwe, **kwargs))
            terms = Syntagmatic(terms)
        item["en"]["morph"]=terms
    
    
if __name__ == "__main__":
    with open("../data/FranceTerme_triples.json","rt") as file:
        data = json.load(file)
    per_target, prefixes, suffixes, compounds = get_morph_table()
    parse_data(data, per_target, prefixes, suffixes, compounds)
    # TODO save data
    
    random.shuffle(data)
    for item in data[:50]:
        morph = item["en"]["morph"]
        print(morph, morph.signature())
