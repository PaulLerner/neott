#!/usr/bin/env python
# coding: utf-8

import random
import json
from collections import Counter

import pandas as pd
import seaborn as sns

from .morph import Term, Inflected, Prefixed, Suffixed, Converted, Native, Neoclassical, Syntagm


def get_morph_table(lang="en"):
    lang = {"en": "eng", "fr": "fra"}[lang]
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


def get_morph(token, spacy_pos=None, mwe=None, per_target=None, prefixes=None, suffixes=None, compounds=None, memory=None):
    if token not in per_target:
        return Term(term=token, pos=spacy_pos, trust=0.0)
    
    memory.add(token)
    # empirically, there is always a single option
    morph = per_target[token].iloc[0]
    morphemes = morph.morpheme.split(" @@")
    
    # assumes that Inflection is always the right-most process
    if morph.Inflection:
        compound, inflection = maybe_rec_get_morph(morphemes, token, front=False, per_target=per_target, prefixes=prefixes, suffixes=suffixes, compounds=compounds, memory=memory)           
        term = Inflected(term=token, inflection=inflection, stem=compound, trust=0.5)
    # before derivation
    # TODO a lot of Neoclassical are tagged Derivation
    elif morph.Derivation:
        p_freq = prefixes.get(morphemes[0], 0)
        s_freq = suffixes.get(morphemes[-1], 0)
        # the prefix is more frequent than the suffix OR (both equal or UNK but prefix shorther than suffix)
        if (p_freq > s_freq) or ((p_freq == s_freq) and (len(morphemes[0]) < len(morphemes[-1]))):
            compound, prefix = maybe_rec_get_morph(morphemes, token, front=True, per_target=per_target, prefixes=prefixes, suffixes=suffixes, compounds=compounds, memory=memory)
            term = Prefixed(term=token, prefix=prefix, stem=compound, pos=spacy_pos, trust=0.5)
        else:
            compound, suffix = maybe_rec_get_morph(morphemes, token, front=False, per_target=per_target, prefixes=prefixes, suffixes=suffixes, compounds=compounds, memory=memory)
            term = Suffixed(term=token, suffix=suffix, stem=compound, pos=spacy_pos, trust=0.5)
    # and before Compounding (last process before root)
    elif morph.Compound:
        if len(morphemes) > 2:
            # assumes right-headed compound (English and Neoclassical)
            r, l = maybe_rec_get_morph(morphemes, token, front=True, per_target=per_target, prefixes=prefixes, suffixes=suffixes, compounds=compounds, memory=memory)
            l = Term(l, trust=0.5)
        else:
            l, r = morphemes
            l, r = Term(l, trust=0.5), Term(r, trust=0.5)
        term = Native(term=token, stem_l=l, stem_r=r, pos=spacy_pos, trust=0.5)        
    # monomorpheme
    else:
        assert len(morphemes) == 1
        term = Term(token, pos=spacy_pos, trust=0.5)
               
    return term


def parse_data(data, per_target, lang="en", **kwargs):
    for item in data:
        term = item[lang]["text"].lower().strip()
        # init memory to avoid infinite recursion
        memory = set()
        if term in per_target: 
            term = get_morph(term, item[lang]["pos"][0], False, per_target=per_target, memory=memory, **kwargs)
        else:
            tokens = []
            mwe = len(item[lang]["tokens"]) > 1
            for token, pos in zip(item[lang]["tokens"], item[lang]["pos"]):
                token = token.lower().strip()
                tokens.append(get_morph(token, pos, mwe, per_target=per_target, memory=memory, **kwargs))
            term = Syntagm(terms=tokens, term=item[lang]["text"])
        # TODO .to_dict() to save to JSON
        item[lang]["morph"] = term
    

def save(data, lang="en"):    
    for item in data:
        item[lang]["morph"] = item[lang]["morph"].to_dict()
        
    with open("data/FranceTerme_triples.json", "wt") as file:
        json.dump(data, file)
        
        
def viz(data, lang="en"):
    tuples=Counter()
    trusts_terms = Counter()
    trusts_morphs = Counter()
    morphemes_l = []
    for item in data:
        morph = item[lang]["morph"]
        trusts_terms[morph.trust > 0.0] += 1
        if morph.trust <= 0.0:
            continue
        labels = morph.labels()
        tuples[tuple(sorted(labels))]+=1
        morphemes_l.append({"length":len(morph), "unit":"morpheme"})
        morphemes_l.append({"length":len(item[lang]["tokens"]), "unit":"word"})
        if not isinstance(morph, Syntagm):
            morph = [morph]
        for m in morph:
            trusts_morphs[m.trust > 0.0]+=1
        
    print(f"{trusts_terms=} {trusts_terms[True]/sum(trusts_terms.values()):.1%}")
    print(f"{trusts_morphs=} {trusts_morphs[True]/sum(trusts_morphs.values()):.1%}")    
    print(pd.DataFrame(tuples.most_common()).to_latex(index=False))
    
    morphemes_l = pd.DataFrame(morphemes_l)
    fig = sns.displot(morphemes_l,x="length", hue="unit",discrete=True)
    fig.savefig(f"viz/FranceTerme_{lang}_morph_sig.pdf")


if __name__ == "__main__":
    with open("data/FranceTerme_triples.json","rt") as file:
        data = json.load(file)
    lang="en"
    per_target, prefixes, suffixes, compounds = get_morph_table(lang=lang)
    parse_data(data, per_target=per_target, prefixes=prefixes, suffixes=suffixes, compounds=compounds, lang=lang)
    viz(data, lang)
    save(data, lang)

    random.shuffle(data)
    for item in data[:50]:
        morph = item[lang]["morph"]
        print(morph, morph.signature())
