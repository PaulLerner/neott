import os
from typing import Optional, Union, Dict
from jsonargparse import CLI
import json
from dataclasses import dataclass, asdict

import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, PretrainedConfig, AutoTokenizer

from .utils import infinite_random_data

PROMPTS = {
    "en": {
        # bawden and yvon
        "version": "If the original version says {src_term} then the {tgt_lang} version should say: {tgt_term}",
        # PL
        "term": "The term {src_term} can be translated in {tgt_lang} as {tgt_term}",
        # bloomz (instruction)
        "tatoeba_mt": "Translate the following term from {src_lang} to {tgt_lang} {src_term} {tgt_term}"
    },
    "fr": {
        # PL
        "term": "Le terme {src_lang} {src_term} peut se traduire en {tgt_lang} par {tgt_term}",
        # PL
        "def": "{src_def} définit le terme {tgt_term}",
        # bloomz (instruction)
        "tatoeba_mt": "Traduis le terme {src_lang} suivant en {tgt_lang} {src_term} {tgt_term}"
    }
}

LANGUAGES = {
    "en": {"en": "English", "fr": "French"},
    "fr": {"en": "anglais", "fr": "français"}
}


@dataclass
class ModelKwargs:
    pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None
    config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None
    cache_dir: Optional[Union[str, os.PathLike]] = None
    ignore_mismatched_sizes: bool = False
    force_download: bool = False
    local_files_only: bool = False
    token: Optional[Union[str, bool]] = None
    revision: str = "main"
    use_safetensors: bool = None
    resume_download: bool = False
    output_loading_info: bool = False
    torch_dtype: str = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention_2: bool = False


@dataclass
class DataKwargs:
    batch_size: Optional[int] = 1
    shuffle: Optional[bool] = False
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    timeout: float = 0
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False
    pin_memory_device: str = ""


@dataclass
class TokenizerKwargs:
    return_tensors: str = 'pt'
    padding: str = 'longest'
    truncation: bool = False
    return_overflowing_tokens: bool = False


@dataclass
class GenKwargs:
    num_beams: int = 4
    max_new_tokens: int = 64


def fill_template(item, template, icl=False, src="en", tgt="fr", src_lang="anglais", tgt_lang="français"):
    if icl:
        tgt_term = item[tgt]["text"]
    else:
        tgt_term = ""
    return template.format(tgt_term=tgt_term, src_lang=src_lang, tgt_lang=tgt_lang, src_term=item[src]["text"],
                           src_def=item[src]["def"]["text"])


def icl(eval_set, icl_set, n_icl: int = 5, seed: int = 0, **kwargs):
    # less pythonic way of randomly looping through icl_set:
    # indices = np.arange(len(icl_set))
    # np.random.shuffle(indices)
    # i = 0
    # for item in eval_set:
    #     icl_eg = []
    #     for _ in range(n_icl):
    #         icl_eg.append(fill_template(icl_set[indices[i]], icl=True, **kwargs))
    #         i += 1
    #         if i >= indices.shape[0]:
    #             np.random.shuffle(indices)
    #             i = 0

    np.random.seed(seed)
    icl_gen = infinite_random_data(icl_set)
    for item in eval_set:
        icl_eg = []
        for _ in range(n_icl):
            icl_eg.append(fill_template(next(icl_gen), icl=True, **kwargs))
        icl_eg.append(fill_template(item, icl=False, **kwargs))
        item["input_text"] = " ### ".join(icl_eg)


def evaluate(eval_set, model, tokenizer, gen_kwargs):
    for inputs in eval_set:
        batch_size, seq_len = inputs["input_ids"].shape
        target_text = inputs.pop("target_text")
        # TODO top-K hypothesis
        output = model.generate(**inputs, **gen_kwargs)
        # keep only newly generated tokens
        if tokenizer.padding_side == 'left':
            output = output[:, seq_len:]
        else:
            raise NotImplementedError("")
        output_text = tokenizer.batch_decode(output)
        # TODO eval
        for o, t in zip(output_text, target_text):
            print(o, "&", t)
        break


class DataCollator:
    def __init__(self, tokenizer, device="cuda", tgt: str = "fr", **kwargs):
        self.tokenizer = tokenizer
        self.device = device
        self.tgt = tgt
        self.kwargs = kwargs

    def collate_fn(self, items):
        inputs = self.tokenizer([item["input_text"] for item in items], **self.kwargs)
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
        inputs["target_text"] = [item[self.tgt]["text"] for item in items]
        return inputs


def prompt(data_path: str, seed: int = 0, eval_set: str = "dev", icl_set: str = "train", src: str = "en",
           tgt: str = "fr", n_icl: int = 5, template_lang: str = "fr", template_form: str = "term",
           model_kwargs: ModelKwargs = ModelKwargs(), data_kwargs: DataKwargs = DataKwargs(),
           tokenizer_name: str = None,
           tokenizer_kwargs: TokenizerKwargs = TokenizerKwargs(), device: str = "cuda",
           gen_kwargs: GenKwargs = GenKwargs()):
    """Prompt LLMs to generate terms (by translating them and/or given their definition)"""
    with open(data_path, 'rt') as file:
        data = json.load(file)
    eval_set = data[eval_set]
    icl_set = data[icl_set]

    src_lang = LANGUAGES[template_lang][src]
    tgt_lang = LANGUAGES[template_lang][tgt]
    template = PROMPTS[template_lang][template_form]
    icl(eval_set, icl_set, n_icl=n_icl, seed=seed, src_lang=src_lang, tgt_lang=tgt_lang, template=template, src=src,
        tgt=tgt)

    model = AutoModelForCausalLM.from_pretrained(**asdict(model_kwargs))
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    data_collator = DataCollator(tokenizer, device=device, tgt=tgt, **asdict(tokenizer_kwargs))
    eval_set = DataLoader(eval_set, collate_fn=data_collator.collate_fn, **asdict(data_kwargs))
    evaluate(eval_set, model, tokenizer, gen_kwargs=asdict(gen_kwargs))


if __name__ == "__main__":
    CLI(prompt)
