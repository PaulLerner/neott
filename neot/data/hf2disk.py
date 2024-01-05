# -*- coding: utf-8 -*-

from jsonargparse import CLI
from typing import List
from pathlib import Path

import datasets


def main(output: str, names: List[str] = None, cache_dir: str = None):
    output = Path(output)
    output.mkdir(exist_ok=True)
    for name in names:
        d = datasets.load_dataset(name, cache_dir=cache_dir)
        d.save_to_disk(output/name.split("/")[-1])
        
        
if __name__ == "__main__":
    CLI(main)