from dataclasses import dataclass, asdict
import json
from typing import Optional, Union, List
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from transformers import AutoTokenizer

ICL_SEP = "###"
PROMPTS = {
    "en": {
        # bawden and yvon
        "version": "If the original version says {src_term} then the {tgt_lang} version should say :{tgt_term}",
        # PL
        "term": "The term {src_term} can be translated in {tgt_lang} as :{tgt_term}",
        # bloomz (instruction)
        "tatoeba_mt": "Translate the following term from {src_lang} to {tgt_lang} {src_term} :{tgt_term}"
    },
    "fr": {
        # PL
        "term": "Le terme {src_lang} {src_term} peut se traduire en {tgt_lang} par :{tgt_term}",
        "def": "{src_def} définit le terme :{tgt_term}",
        "def+term": "{src_def} définit le terme {src_lang} {src_term} qui peut se traduire en {tgt_lang} par :{tgt_term}",
        # bloomz (instruction)
        "tatoeba_mt": "Traduis le terme {src_lang} suivant en {tgt_lang} {src_term} :{tgt_term}"
    }
}
LANGUAGES = {
    "en": {"en": "English", "fr": "French"},
    "fr": {"en": "anglais", "fr": "français"}
}


@dataclass
class PromptKwargs:
    seed: Union[int, List[int]] = 0
    src: Union[str, List[str]] = "en"
    tgt: Union[str, List[str]] = "fr"
    n_icl: Union[int, List[int]] = 5
    template_lang: Union[str, List[str]] = "fr"
    def_lang: Union[str, List[str]] = "fr"
    template_form: Union[str, List[str]] = "term"
    selector: Union[str, List[str]] = "random"
    morph_lang: Union[str, List[str]] = "fr"
    morph: Union[str, List[str]] = None


@dataclass
class DataKwargs:
    batch_size: Optional[int] = 1
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
    max_length: int = None


class DataModule(pl.LightningDataModule):
    def __init__(self, tokenizer_name: str = None, tokenizer_kwargs: TokenizerKwargs = TokenizerKwargs(),
                 add_prefix_space: bool = False, data_path: str = None, data_kwargs: DataKwargs = DataKwargs(),
                 prompt_kwargs: PromptKwargs = PromptKwargs(), filter_def: str = None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=add_prefix_space)
        assert self.tokenizer.padding_side == 'left'
        prompt_sep = self.tokenizer.encode(':', add_special_tokens=False)
        assert len(prompt_sep) == 1, prompt_sep
        self.prompt_sep = prompt_sep[0]
        self.tokenizer_kwargs = asdict(tokenizer_kwargs)
        self.data_path = data_path
        self.dataset = {}
        self.data_kwargs = asdict(data_kwargs)
        assert prompt_kwargs.n_icl == 0
        self.src_lang = LANGUAGES[prompt_kwargs.template_lang][prompt_kwargs.src]
        self.tgt_lang = LANGUAGES[prompt_kwargs.template_lang][prompt_kwargs.tgt]
        self.template = PROMPTS[prompt_kwargs.template_lang][prompt_kwargs.template_form]
        self.prompt_kwargs = prompt_kwargs
        self.filter_def = filter_def

    def setup(self, stage=None):
        with open(self.data_path, 'rt') as file:
            self.dataset = json.load(file)
        for name, subset in self.dataset.items():
            for item in subset:
                item["input_text"] = self.template.format(tgt_term=" " + item[self.prompt_kwargs.tgt]["text"],
                                                          src_lang=self.src_lang, tgt_lang=self.tgt_lang,
                                                          src_term=item[self.prompt_kwargs.src]["text"],
                                                          src_def=item[self.prompt_kwargs.def_lang]["def"]["text"])

    def train_dataloader(self):
        if 'train' not in self.dataset:
            return None
        if self.filter_def is not None:
            before = len(self.dataset['train'])
            self.dataset['train'] = [item for item in self.dataset['train'] if item[self.filter_def]['def']['text']]
            print(f"filtered training set from {before} to {len(self.dataset['train'])} with {self.filter_def} definitions")
        return DataLoader(
            self.dataset['train'],
            collate_fn=self.collate_fn,
            shuffle=True,
            **self.data_kwargs
        )

    def val_dataloader(self):
        if 'validation' not in self.dataset:
            return None
        return DataLoader(
            self.dataset['validation'],
            collate_fn=self.collate_fn,
            shuffle=False,
            **self.data_kwargs
        )

    def test_dataloader(self):
        if 'test' not in self.dataset:
            return None
        return DataLoader(
            self.dataset['test'],
            collate_fn=self.collate_fn,
            shuffle=False,
            **self.data_kwargs
        )

    def collate_fn(self, items):
        inputs = self.tokenizer([item["input_text"] for item in items], **self.tokenizer_kwargs)
        labels = inputs['input_ids'].clone()
        for label in labels:
            where = (label == self.prompt_sep).nonzero()
            # did not find prompt_sep in input (only legal if tokenizer truncates) -> mask the whole label
            if where.shape[0] == 0:
                where = label.shape[0] - 1
            else:
                where = where[0, 0]
            label[: where + 1] = self.trainer.lightning_module.loss_fct.ignore_index
        inputs["labels"] = labels
        return inputs
