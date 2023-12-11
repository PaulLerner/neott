#!/usr/bin/env python
# coding: utf-8

# In[24]:


from xml.etree import ElementTree as ET
import random
import re
import json


# In[2]:


import pandas as pd


# In[4]:


from collections import Counter


# In[3]:


tree = ET.parse('data/FranceTerme.xml')
root = tree.getroot()


# # viz

# In[6]:


c = Counter()
articles = root.findall("Article")
random.shuffle(articles)
defs = []
for article in articles[:100]:
    def_fr = article.find("Definition").text.strip()
    c['def'] += min(1,len(def_fr))
    c['concept']+=1
    print(def_fr,end=' / ')
    for term in article.findall("Terme"):
        if "Terme" not in term.attrib:
            continue
        c['fr']+=1
        fr = term.attrib["Terme"].strip()
        print(fr,end=' / ')
        c['fem'] += 1 if fr.find(', -') >= 0 else 0
    found_en = False
    for eq in article.findall("Equivalent"):
        c[eq.attrib['langue']]+= 1
        if eq.attrib['langue'] != 'en':
            continue
        found_en = True
        for ep in eq.findall("Equi_prop"):
            ep = ep.text.strip()
            print(ep,end=' / ')
            #print(fr,'/',en,'/',def_fr)
    c[(found_en,bool(def_fr))] += 1
    if def_fr and found_en:
        defs.append(def_fr)
    
    print("\n")


# # preproc

# In[21]:


c = Counter()
articles = root.findall("Article")
triples = []
fem_reg = re.compile(r", -\w+")
article_reg = re.compile(r" \(l[’ae]s?\)")

for article in articles:
    def_fr = article.find("Definition").text.strip()
    fr_terms = []
    for term in article.findall("Terme"):
        if "Terme" not in term.attrib:
            continue
        c['fr']+=1
        fr = term.attrib["Terme"].strip()
        fr = fem_reg.sub("", fr)
        fr = article_reg.sub("", fr)
        fr_terms.append(fr)
        
    en_terms = []
    for eq in article.findall("Equivalent"):
        c[eq.attrib['langue']]+= 1
        if eq.attrib['langue'] != 'en':
            continue
        for ep in eq.findall("Equi_prop"):
            ep = ep.text.strip()
            en_terms.append(ep)
    
    if (not def_fr) or (not en_terms):
        continue
    
    fr_term = fr_terms.pop(0)
    en_term = en_terms.pop(0)
    
    surdoms=[]
    sousdoms=[]
    for domaine in article.findall("Domaine"):
        for surdom in domaine.findall("Dom"):
            surdoms.append(surdom.text.strip())
        for sousdom in domaine.findall("S-dom"):
            sousdoms.append(sousdom.text)
    
    
    triples.append({
        "Dom": surdoms, 
        "S-dom": sousdoms,
        "fr": fr_term,
        "fr_syn": fr_terms,
        "en": en_term,
        "en_syn": en_terms,
        "id":article.attrib['numero']
    })


# In[22]:


triples


# In[9]:


len(set( for article in articles))


# In[23]:


len(triples)


# In[25]:


with open("data/FranceTerme_triples.json","wt") as file:
    json.dump(triples, file)


# # stat

# ## domaines

# In[27]:


import seaborn as sns


# In[26]:


surdoms=[]
sousdoms=[]

for triple in triples:
    surdoms.extend(triple["Dom"])
    sousdoms.extend(triple["S-dom"])


# In[33]:


sum(Counter(surdoms).values())/len(triples)


# In[35]:


pd.DataFrame(Counter(surdoms).most_common()).to_csv("viz/FranceTerme_domain.csv")


# ## levenshtein transliteration

# In[36]:


import editdistance


# In[37]:


triples[0]


# In[38]:


triple


# In[52]:


distances = []
random.shuffle(triples)
for triple in triples:
    editd = editdistance.eval(triple['fr'],triple['en'])
    if editd not in distances:
        print(editd,triple['fr'],triple['en'])
    distances.append(editd)


# In[47]:


fig = sns.displot(distances,discrete=True)


# In[48]:


fig.savefig("viz/FranceTerme_editdist.pdf")

