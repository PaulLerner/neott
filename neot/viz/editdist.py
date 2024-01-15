import json
from tqdm import tqdm
from jsonargparse import CLI
# import random

import pandas as pd
import seaborn as sns

import editdistance


def edit_dist(data):
    distances = []
    # random.shuffle(data)
    for triple in tqdm(data):
        editd = editdistance.eval(triple['fr']["text"], triple['en']["text"])
        # if editd not in distances:
        #    print(editd,triple['fr'],triple['en'])
        distances.append(editd)
    return pd.DataFrame({"distance": distances})


def viz(distances, output):
    desc = distances.describe(percentiles=[.25, .5, .75, .9, .95, .99])
    print(desc)
    fig = sns.displot(distances[distances.distance < desc.distance["99%"]], discrete=True)
    fig.savefig(output)


def main(data: str, output: str):
    with open(data, "rt") as file:
        data = json.load(file)
    distances = edit_dist(data)
    viz(distances, output)


if __name__ == "__main__":
    CLI(main)
