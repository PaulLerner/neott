#!/usr/bin/env python
# coding: utf-8
import itertools
import warnings
import logging
from jsonargparse import CLI
import enum

import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


logger = logging.getLogger(__name__)


class Negatives(enum.Enum):
    intra_pair = 0
    all_intra = 1
    all = 2


def compute_accuracy(argmax, sorted_vocab, start_indices, intra_indices, START_OF_WORD_CHAR):
    strict_accuracy, case_accuracy, matches_other_start = 0, 0, 0
    matches_upper = 0
    for i, pred in enumerate(argmax.tolist()):
        start_token = sorted_vocab[start_indices[i]]
        intra_token = sorted_vocab[intra_indices[pred]]

        if intra_token[0] == START_OF_WORD_CHAR:
            matches_other_start += 1

        if start_token[1:] == intra_token:
            strict_accuracy += 1
        elif start_token[1:].lower() == intra_token.lower():
            case_accuracy += 1
        elif intra_token[0] != START_OF_WORD_CHAR and intra_token[0].isupper():
            matches_upper += 1
        else:
            pass  # print(start_token, intra_token)

    for v in [strict_accuracy, (strict_accuracy + case_accuracy), matches_upper, matches_other_start]:
        print(f"{(v/len(argmax)) * 100:.1f}", end="% | ")
    print()


def compute_alignment(model_name: str, word_embeddings, vocab, START_OF_WORD_CHAR, alpha_filter: bool = False,
                      negatives: Negatives = Negatives.intra_pair):
    not_word = re.compile(r"[^A-Za-z]")
    starts = {token[1:] for token in vocab if token[0] == START_OF_WORD_CHAR and ((not alpha_filter) or not_word.search(token[1:]) is None)}
    pairs = starts & vocab.keys()
    logger.debug(f"{len(starts)=} {len(pairs)=}")

    # defaults to Negatives.intra_pair but may get overwritten
    start_indices, intra_indices = [], []
    for token in pairs:
        start_indices.append(vocab[START_OF_WORD_CHAR + token])
        intra_indices.append(vocab[token])
    sorted_vocab = sorted(vocab, key=vocab.get)

    start_indices = torch.tensor(start_indices, dtype=int)

    if negatives == Negatives.all_intra:
        intra_indices = torch.tensor([v for k, v in vocab.items() if k[0] != START_OF_WORD_CHAR], dtype=int)
    elif negatives == Negatives.all:
        intra_indices = torch.arange(len(word_embeddings), dtype=int)
    else:
        intra_indices = torch.tensor(intra_indices, dtype=int)

    logger.debug(f"{len(start_indices)=} {len(intra_indices)=}")

    start_embeddings = word_embeddings[start_indices]
    intra_embeddings = word_embeddings[intra_indices]
    sim = start_embeddings @ intra_embeddings.T

    logger.debug(f"{sim.shape=} {sum(sim.shape)=}")

    # remove self from top-2
    if negatives == Negatives.all:
        values, indices = sim.topk(2, 1)
        argmax = indices[:, 0]
        # should be everywhere
        where = start_indices == indices[:, 0]
        argmax[where] = indices[where, 1]
    else:
        argmax = sim.argmax(1)

    print(f"{model_name} | {alpha_filter} | {len(pairs):,d} | {len(intra_indices)-1:,d} {negatives.name} | ", end=" ")
    compute_accuracy(argmax, sorted_vocab, start_indices, intra_indices, START_OF_WORD_CHAR)


def main(model_name: str, alpha_filter: bool = None, negatives: Negatives = None, verbose: int = logging.INFO):
    logging.basicConfig(format='%(message)s', level=verbose)
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, trust_remote_code=True)
    vocab = tokenizer.vocab
    logger.debug(tokenizer.tokenize("foo", " foo"))
    START_OF_WORD_CHAR = tokenizer.tokenize("foo")[0][0]
    logger.debug(f"{START_OF_WORD_CHAR=} {len(vocab)=}")

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    # FIXME get_input_embeddings
    if hasattr(model, "model"):
        word_embeddings = model.model.embed_tokens.weight
    else:
        word_embeddings = model.transformer.word_embeddings.weight
    word_embeddings = word_embeddings.detach().to(torch.float32)
    if len(word_embeddings) > len(vocab):
        warnings.warn(f"Trimming {word_embeddings.shape=} to {len(vocab)=}")
        word_embeddings = word_embeddings[:len(vocab)]
    norm = word_embeddings.norm(2, 1, keepdim=True)
    word_embeddings /= norm
    logger.debug(f"{word_embeddings.shape=} {norm.shape=}")

    alpha_filter = [alpha_filter] if alpha_filter is not None else [False, True]
    negatives = [negatives] if negatives is not None else Negatives
    print("model | filter `A-Za-z` | #pairs | negatives | P@1 (cs) | P@1 (ci) | matches upper | matches other start")
    print("------|-----------------|--------|-----------|----------|----------|---------------|--------------------")
    for a, n in itertools.product(alpha_filter, negatives):
        compute_alignment(model_name, word_embeddings, vocab, START_OF_WORD_CHAR, a, n)


if __name__ == "__main__":
    CLI(main)
