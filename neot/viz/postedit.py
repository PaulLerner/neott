#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import re

import pandas as pd

# install from python3 script https://github.com/google/diff-match-patch/tree/master
from dmp import diff_match_patch


def fixquote(x):
    x = re.sub('&#039;', "'", x)
    x = re.sub('&amp;', '&', x)
    # &quot; for text 569450-9996-169 par exemple (id_hal-Translation-id_Postedit-id)
    x = re.sub('“', '"', x)
    x = re.sub('”', '"', x)
    x = re.sub('&quot;', '"', x)
    return x.strip()


def preproc(x):
    return fixquote(x).lower()


def get_steg():
    data = pd.read_csv("data/STEG/corpora/raw/trads_post-edit_upc/PE_EXPL_280-V2.txt", skiprows=23, sep="\t")
    root = Path("/home/lerner/code/neot/data/STEG/corpora/raw/trads_post-edit_upc/")
    dfs = []
    for path in root.glob("PE*"):
        try:
            dfs.append(pd.read_csv(path, skiprows=23, sep="\t"))
        except Exception as e:
            print(e)
    {'en', 'TA (DeepL)', 'fr'} - dfs[0].columns
    data = pd.concat(dfs).dropna(subset=['en', 'TA (DeepL)', 'fr'])
    return data


def diffhtml(data):
    spans = []
    for _, row in data.iterrows():
        # FIXME consistent column naming in get_steg
        trans = (row["TA (DeepL)"])  # .translation)
        pe = (row.fr)  # postedition)
        trans = preproc(trans)
        pe = preproc(pe)
        if trans == pe:
            continue
        diff = dmper.diff_main(trans, pe)
        dmper.diff_cleanupSemantic(diff)
        spans.append(f"<p>{dmper.diff_prettyHtml(diff)}<p/>")
    return spans


# data = pd.read_csv("data/TAL/corpora/raw/2023-postedition/postedition_aligned.final.community.tsv",'\t')
# data = pd.concat((data,pd.read_csv("data/TAL/corpora/raw/2023-postedition/postedition_aligned.final.translator.tsv",'\t')))
data = get_steg()

dmper = diff_match_patch()
spans = diffhtml(data)
with open("viz/pe_steg.html", "wt") as file:
    file.write("\n".join(spans))
