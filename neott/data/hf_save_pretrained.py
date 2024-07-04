from jsonargparse import CLI
from typing import List
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


def main(output: str, names: List[str] = None, cache_dir: str = None):
    """Save all datasets in names in the specified output directory"""
    output = Path(output)
    output.mkdir(exist_ok=True)
    for name in names:
        tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(name, cache_dir=cache_dir, trust_remote_code=True)
        output_path = output / name.split("/")[-1]
        tokenizer.save_pretrained(output_path)
        model.save_pretrained(output_path)


if __name__ == "__main__":
    CLI(main, description=main.__doc__)