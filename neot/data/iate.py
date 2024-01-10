#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import json


RELIABILITY = {'Reliable': 3,
         'Reliability not verified': 2,
         'Very reliable': 4,
         'Minimum reliability': 1}


def get_data():
    data = pd.read_csv("data/IATE_export.csv","|")    
    print(data.describe(include="all"))  
    table = []    
    for e_id, terms in data.groupby("E_ID"):
        syns = {}
        domain = set()
        for lang, lterm in terms.groupby("L_CODE"):  
            syns.setdefault(lang, {})
            syns[lang][lterm.T_TERM.iloc[0]] = RELIABILITY[lterm.T_RELIABILITY.iloc[0]]
            domain.add(lterm.E_DOMAINS.iloc[0])
        assert len(domain) == 1
        domain = next(iter(domain))
        term = {}
        term["id"] = e_id
        term["domain"] = domain
        for lang, lsyn in syns.items():
            term.setdefault(lang, {})
            best = sorted(lsyn, key=lsyn.get, reverse=True)[0]
            term[lang]["text"] = best
            lsyn.pop(best)
            term[lang]["syn"] = list(lsyn.values())
        table.append(term)
        
    return table


if __name__ == '__main__':
    data = get_data()
    with open("data/iate.json","wt") as file:
        json.dump(data,file)
