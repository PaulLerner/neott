from jsonargparse import CLI
from typing import List
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


def main(output: str, names: List[str] = None, cache_dir: str = None):
    """Save all datasets in names in the specified output directory"""
    output = Path(output)
    output.mkdir(exist_ok=True)
    for name in names:
        model = AutoModelForCausalLM.from_pretrained(name, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
        output_path = output / name.split("/")[-1]
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    CLI(main, description=main.__doc__)