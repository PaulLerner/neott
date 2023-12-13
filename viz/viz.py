import seaborn as sns
import editdistance
import pandas as pd
from collections import Counter
import random


surdoms=[]
sousdoms=[]

for triple in triples:
    surdoms.extend(triple["Dom"])
    sousdoms.extend(triple["S-dom"])


# In[35]:


pd.DataFrame(Counter(surdoms).most_common()).to_csv("viz/FranceTerme_domain.csv")


# ## levenshtein transliteration

# In[36]:




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