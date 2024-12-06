import warnings

import pandas as pd
import torch
from jsonargparse import CLI
from dataclasses import dataclass, asdict
import json
from typing import List, Union
from string import Formatter

from transformers import AutoModelForCausalLM, AutoTokenizer

from .trainee import ModelKwargs, GenKwargs
from .data.train import (TokenizerKwargs, DataKwargs, PromptKwargs, PROMPTS, ICL_SEP, LANGUAGES, fill_template,
                         CHAT_USER_START, CHAT_USER_END)
from .prompt import SelectorKwargs, SelectorFusion


def prompt(item, model, tokenizer, icl_gen, template_kwargs, gen_kwargs):
    inputs = tokenizer(input_texts, **tokenizer_kwargs)

    output = model.generate(eos_token_id=eos_token_id, return_dict_in_generate=True,
                            pad_token_id=tokenizer.pad_token_id, **inputs, **gen_kwargs).sequences
    # keep only newly generated tokens
    if tokenizer.padding_side == 'left':
        output = output[:, seq_len:].cpu()
    else:
        raise NotImplementedError(f"{tokenizer.padding_side=}")
    output_text = tokenizer.batch_decode(output, skip_special_tokens=True)


def user_loop(required_fields, src: str = "en", def_lang: str = "fr", *args, **kwargs):
    while True:
        if 'src_term' in required_fields:
            src_term = input(f">>> Input the source term (in {src})").strip()
        else:
            src_term = None
        if 'src_def' in required_fields:
            src_def = input(f">>> Input the definition of the term (in {def_lang})").strip()
        else:
            src_def = None
        item = {}
        item[src] = {}
        item[def_lang] = {}
        item[src]["text"] = src_term
        item[def_lang]["def"] = {"text": src_def}
        output = prompt(item, *args, **kwargs)[0]
        print(f"{output}\n")


def main(icl_path, prompt_kwargs: PromptKwargs = PromptKwargs(), model_kwargs: ModelKwargs = ModelKwargs(),
         tokenizer_name: str = None, tokenizer_kwargs: TokenizerKwargs = TokenizerKwargs(), add_prefix_space: bool = False,
         gen_kwargs: GenKwargs = GenKwargs(), selector_kwargs: Union[SelectorKwargs, List[SelectorKwargs]] = None):
    """Interactively prompt an LLM to generate a term. Input either source term to translate or definition"""

    if tokenizer_name is None:
        tokenizer_name = model_kwargs.pretrained_model_name_or_path

    if selector_kwargs is None:
        selector_kwargs = [SelectorKwargs()]
    elif not isinstance(selector_kwargs, list):
        selector_kwargs = [selector_kwargs]

    selector_kwargs = [asdict(kwarg) for kwarg in selector_kwargs]
    with open(icl_path, 'rt') as file:
        icl_set = json.load(file)["train"]
    icl_gen = SelectorFusion(selector_kwargs, icl_set=icl_set, def_lang=def_lang, src=src)

    model = AutoModelForCausalLM.from_pretrained(**asdict(model_kwargs), trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=add_prefix_space, add_eos_token=False, trust_remote_code=True)
    src_lang = LANGUAGES[template_lang][src]
    tgt_lang = LANGUAGES[template_lang][tgt]
    template = PROMPTS[template_lang][template_form]
    required_fields = {field[1] for field in Formatter().parse(template)}
    template_kwargs = dict(src_lang=src_lang, tgt_lang=tgt_lang, template=template, tgt=tgt)
    user_loop(required_fields, model=model, tokenizer=tokenizer, icl_gen=icl_gen, template_kwargs=template_kwargs,
              gen_kwargs=gen_kwargs, tokenizer_kwargs=tokenizer_kwargs, **asdict(prompt_kwargs))


if __name__ == "__main__":
    CLI(main, description=main.__doc__)
