import argparse

from transformers import AutoModelForCausalLM


def main(model):
    for name in model:
        m = AutoModelForCausalLM.from_pretrained(name)
        n = sum(p.numel() for p in m.parameters())
        print(f"{name}: {n:,d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute the number of parameters in a Module')
    parser.add_argument('model', type=str, nargs='+', help='pretrained model name or path')
    main(**vars(parser.parse_args()))