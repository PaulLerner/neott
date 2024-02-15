import os
from typing import Optional, Union, List

import pandas as pd
from jsonargparse import CLI
import json
from dataclasses import dataclass, asdict
from tqdm import tqdm

import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, PretrainedConfig, AutoTokenizer

from .utils import infinite_random_data, Path, ListOrArg
from .metrics import compute_metrics, Preprocessor

ICL_SEP = "###"
PROMPTS = {
    "en": {
        # bawden and yvon
        "version": "If the original version says {src_term} then the {tgt_lang} version should say:{tgt_term}",
        # PL
        "term": "The term {src_term} can be translated in {tgt_lang} as:{tgt_term}",
        # bloomz (instruction)
        "tatoeba_mt": "Translate the following term from {src_lang} to {tgt_lang} {src_term}:{tgt_term}"
    },
    "fr": {
        # PL
        "term": "Le terme {src_lang} {src_term} peut se traduire en {tgt_lang} par:{tgt_term}",
        "def": "{src_def} définit le terme:{tgt_term}",
        "def+term": "{src_def} définit le terme {src_lang} {src_term} qui peut se traduire en {tgt_lang} par:{tgt_term}",
        # bloomz (instruction)
        "tatoeba_mt": "Traduis le terme {src_lang} suivant en {tgt_lang} {src_term}:{tgt_term}"
    }
}
LANGUAGES = {
    "en": {"en": "English", "fr": "French"},
    "fr": {"en": "anglais", "fr": "français"}
}


@dataclass
class ModelKwargs:
    pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None
    device_map: str = "cuda"
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
    num_return_sequences: int = 1


def fill_template(item, template, icl=False, src="en", tgt="fr", src_lang="anglais", tgt_lang="français",
                  def_lang: str = "fr"):
    if icl:
        tgt_term = " " + item[tgt]["text"]
    else:
        tgt_term = ""
    return template.format(tgt_term=tgt_term, src_lang=src_lang, tgt_lang=tgt_lang, src_term=item[src]["text"],
                           src_def=item[def_lang]["def"]["text"])


class ExampleSelector:
    def __init__(self, icl_set):
        self.icl_set = icl_set

    def __call__(self, item):
        self.item = item
        return self

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError()


class RandomExampleSelector(ExampleSelector):
    def __init__(self, icl_set):
        super().__init__(icl_set)
        self.icl_set = infinite_random_data(self.icl_set)

    def __next__(self):
        for eg in self.icl_set:
            # do not use self in the prompt (may happen if using eval_set as icl_set)
            if eg["id"] == self.item["id"]:
                continue
            return eg


def icl(eval_set, icl_set, n_icl: int = 5, seed: int = 0, **kwargs):
    np.random.seed(seed)
    icl_gen = RandomExampleSelector(icl_set)
    for item in eval_set:
        icl_eg = []
        for _, eg in zip(range(n_icl), icl_gen(item)):
            icl_eg.append(fill_template(eg, icl=True, **kwargs))
        icl_eg.append(fill_template(item, icl=False, **kwargs))
        item["input_text"] = f" {ICL_SEP} ".join(icl_eg)


def post_proc(predictions):
    proc_predictions = []
    for pred in predictions:
        # because ICL_SEP is not a special token it gets decoded by the tokenizer
        i = pred.find(ICL_SEP)
        if i < 0:
            proc_predictions.append(pred)
        else:
            proc_predictions.append(pred[:i])
    return proc_predictions


def evaluate(eval_set, model, tokenizer, gen_kwargs, preproc, device="cuda"):
    predictions, targets = [], []
    icl_sep_id = tokenizer.encode(" " + ICL_SEP, add_special_tokens=False)
    assert len(icl_sep_id) == 1, icl_sep_id
    assert tokenizer.eos_token_id is not None
    eos_token_id = [tokenizer.eos_token_id, icl_sep_id[0]]
    for inputs in tqdm(eval_set):
        batch_size, seq_len = inputs["input_ids"].shape
        target_text = inputs.pop("target_text")
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        # TODO top-K hypothesis
        output = model.generate(eos_token_id=eos_token_id, return_dict_in_generate=True, **inputs,
                                **gen_kwargs).sequences
        # keep only newly generated tokens
        if tokenizer.padding_side == 'left':
            output = output[:, seq_len:]
        else:
            raise NotImplementedError("")
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        predictions.extend(output_text)
        targets.extend(target_text)
    predictions = post_proc(predictions)
    k = gen_kwargs["num_return_sequences"]
    assert len(predictions) % k == 0
    predictions_per_input = []
    for i in range(0, len(predictions), k):
        predictions_per_input.append(predictions[i: i + k])
    metrics = compute_metrics(predictions_per_input, targets, preproc, k=k)
    return {"metrics": metrics, "predictions": predictions_per_input}


class DataCollator:
    def __init__(self, tokenizer, tgt: str = "fr", **kwargs):
        self.tokenizer = tokenizer
        self.tgt = tgt
        self.kwargs = kwargs

    def collate_fn(self, items):
        inputs = self.tokenizer([item["input_text"] for item in items], **self.kwargs)
        inputs["target_text"] = [item[self.tgt]["text"] for item in items]
        return inputs


@dataclass
class PromptKwargs:
    seed: int = 0
    src: str = "en"
    tgt: str = "fr"
    n_icl: int = 5
    template_lang: Union[str, List[str]] = "fr"
    def_lang: str = "fr"
    template_form: Union[str, List[str]] = "term"


def prompt(eval_set, icl_set, model, tokenizer, data_collator, seed: int = 0, src: str = "en", tgt: str = "fr",
           n_icl: int = 5, template_lang: str = "fr", def_lang: str = "fr", template_form: str = "term", device="cuda",
           data_kwargs: DataKwargs = DataKwargs(), gen_kwargs: GenKwargs = GenKwargs(), output_path: Path = None):
    preproc = Preprocessor(tgt)
    src_lang = LANGUAGES[template_lang][src]
    tgt_lang = LANGUAGES[template_lang][tgt]
    template = PROMPTS[template_lang][template_form]
    icl(eval_set, icl_set, n_icl=n_icl, seed=seed, src_lang=src_lang, tgt_lang=tgt_lang, template=template, src=src,
        tgt=tgt, def_lang=def_lang)

    eval_set = DataLoader(eval_set, collate_fn=data_collator.collate_fn, **asdict(data_kwargs))
    output = evaluate(eval_set, model, tokenizer, gen_kwargs=asdict(gen_kwargs), preproc=preproc,
                      device=device)
    metrics = {}
    for k, v in output["metrics"].items():
        if isinstance(v, float):
            metrics[k] = v
    print(metrics)
    if output_path is not None:
        with open(output_path / f"{template_lang}_{template_form}.json", 'wt') as file:
            json.dump(output, file)
    return metrics


def main(data_path: str, eval_set: str = "dev", icl_set: str = "train", prompt_kwargs: PromptKwargs = PromptKwargs(),
         model_kwargs: ModelKwargs = ModelKwargs(), data_kwargs: DataKwargs = DataKwargs(), tokenizer_name: str = None,
         tokenizer_kwargs: TokenizerKwargs = TokenizerKwargs(), add_prefix_space: bool = False,
         gen_kwargs: GenKwargs = GenKwargs(), output_path: Path = None):
    """Prompt LLMs to generate terms (by translating them and/or given their definition)"""
    output_path.mkdir(exist_ok=True)
    with open(data_path, 'rt') as file:
        data = json.load(file)
    eval_set = data[eval_set]
    icl_set = data[icl_set]

    model = AutoModelForCausalLM.from_pretrained(**asdict(model_kwargs))
    if not model_kwargs.load_in_8bit:
        model = model.to(model_kwargs.device_map)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=add_prefix_space)
    data_collator = DataCollator(tokenizer, tgt=prompt_kwargs.tgt, **asdict(tokenizer_kwargs))
    template_langs = PROMPTS.keys() if prompt_kwargs.template_lang is None else ListOrArg(prompt_kwargs.template_lang)
    search_templates = prompt_kwargs.template_form is None
    results = []
    for template_lang in template_langs:
        prompt_kwargs.template_lang = template_lang
        template_forms = PROMPTS[template_lang].keys() if search_templates else ListOrArg(prompt_kwargs.template_form)
        for template_form in template_forms:
            prompt_kwargs.template_form = template_form
            metrics = prompt(eval_set, icl_set, model, tokenizer, data_collator, **asdict(prompt_kwargs),
                             device=model_kwargs.device_map, data_kwargs=data_kwargs, gen_kwargs=gen_kwargs,
                             output_path=output_path)
            metrics.update({"template_lang": template_lang, "template_form": template_form})
            results.append(metrics)
    print(results)
    if output_path is not None:
        pd.DataFrame(results).to_csv(output_path / "results.csv")


if __name__ == "__main__":
    CLI(main, description=main.__doc__)