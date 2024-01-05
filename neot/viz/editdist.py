import seaborn as sns
import editdistance
import json
#import random


with open("data/FranceTerme_triples.json","rt") as file:
    data = json.load(file)
    
distances = []
#random.shuffle(data)
for triple in data:
    editd = editdistance.eval(triple['fr']["text"],triple['en']["text"])
    #if editd not in distances:
    #    print(editd,triple['fr'],triple['en'])
    distances.append(editd)

fig = sns.displot(distances,discrete=True)

fig.savefig("viz/FranceTerme_editdist.pdf")