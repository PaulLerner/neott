import json
from jsonargparse import CLI

import re

acronym = re.compile("[A-Z]{2,}")


def filter_item(item):
    en, fr = item["en"]["text"], item["fr"]["text"]
    if en.lower().strip() == fr.lower().strip():
        return False
    if acronym.search(en) is not None or acronym.search(fr) is not None:
        return False
    return True


def filter_data(data):
    l = len(data)
    data = [item for item in data if (filter_item(item))]
    print(f"Filtered data from {l} to {len(data)}")
    return data


def main(data_path: str):
    """Filter data"""
    with open(data_path, 'rt') as file:
        data = json.load(file)
    data = filter_data(data)
    with open(data_path, 'wt') as file:
        json.dump(data, file)


if __name__ == "__main__":
    CLI(main)
