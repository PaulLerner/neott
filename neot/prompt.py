import enum
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

from .utils import infinite_random_data, all_size_combination, Path, ListOrArg
from .metrics import compute_metrics, Preprocessor
from .morph.labels import MorphLabel

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
    def __init__(self, icl_set, n_icl: int = 5, **kwargs):
        self.icl_set = icl_set
        self.n_icl = n_icl
        self.i = 0

    def __next__(self):
        raise NotImplementedError("subclass and implement")

    def __call__(self, item):
        self.i = 0
        self.item = item
        while self.i < self.n_icl:
            eg = next(self)
            # do not use self in the prompt (may happen if using eval_set as icl_set)
            if eg["id"] == item["id"]:
                continue
            self.i += 1
            yield eg


class RandomExampleSelector(ExampleSelector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.icl_set = infinite_random_data(self.icl_set)

    def __next__(self):
        return next(self.icl_set)


class DomainExampleSelector(ExampleSelector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        domains = {}
        for item in self.icl_set:
            item_domains = item.get("Dom") if item.get("Dom") is not None else [None]
            for domain in item_domains:
                domains.setdefault(domain, [])
                domains[domain].append(item)
        domain_sizes = {}
        for domain, domain_icl_set in domains.items():
            domain_sizes[domain] = len(domain_icl_set)
            domains[domain] = infinite_random_data(domain_icl_set)
        self.domains = domains
        self.domain_sizes = domain_sizes
        self.fallback = infinite_random_data(self.icl_set)

    def __next__(self):
        # note this gives oracle domain -> should be considered oracle/topline
        domain = self.item.get("Dom")
        if domain is not None:
            domain = np.random.choice(domain)
        # you do not want to have always the same example in the prompt for domains with fewer examples than n_icl
        if domain is not None and self.i < self.domain_sizes.get(domain, 0):
            eg = next(self.domains[domain])
        else:
            eg = next(self.fallback)
        return eg


class MorphExampleSelector(ExampleSelector):
    """
    Parameters
    ----------
    morph_lang: str
        Language to which retrieve morphologically similar examples.
        E.g. when translating from EN to FR, you might want to get:
        - morphologically similar EN examples (source-similar)
        - morphologically similar FR examples (target-similar)
    """

    def __init__(self, *args, morph_lang: str = "fr", **kwargs):
        super().__init__(*args, **kwargs)
        self.lang = morph_lang
        morphs = {}
        for item in self.icl_set:
            morph = tuple(sorted(MorphLabel[l].value for l in item[self.lang]['morph_label']))
            morphs.setdefault(morph, [])
            morphs[morph].append(item)

        infinite_morphs = {}
        # should have at least n_icl examples for each possible case
        for m_o in all_size_combination(range(len(MorphLabel))):
            # easy: exact match
            if len(morphs.get(m_o, [])) >= self.n_icl:
                infinite_morphs[m_o] = morphs[m_o]
                continue
            # more difficult: find closest morph (e.g. only 1 difference)
            infinite_morphs[m_o] = []
            sym_diffs = {}
            for m_i in all_size_combination(range(len(MorphLabel))):
                sym_diff = len(set(m_o).symmetric_difference(set(m_i)))
                sym_diffs.setdefault(sym_diff, [])
                sym_diffs[sym_diff].append(m_i)
            # maximum symmetric difference is at most |MorphLabel|
            for diff in range(len(MorphLabel)):
                # all equally close morph get the same chance
                for m_i in sym_diffs[diff]:
                    infinite_morphs[m_o] += morphs.get(m_i, [])
                if len(infinite_morphs[m_o]) >= self.n_icl:
                    break

        # done. here we simply turn potentially small sets in infinite loops
        for morph, morph_icl_set in infinite_morphs.items():
            infinite_morphs[morph] = infinite_random_data(morph_icl_set)
        self.infinite_morphs = infinite_morphs

    def __next__(self):
        morph = tuple(sorted(MorphLabel[l].value for l in self.item[self.lang]['morph_label']))
        return next(self.infinite_morphs[morph])


class ConstrainedMorphExampleSelector(MorphExampleSelector):
    def __init__(self, *args, morph: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.morph = MorphLabel[morph].name
        morphs = []
        for item in self.icl_set:
            for label in item[self.lang]['morph_label']:
                if label == self.morph:
                    morphs.append(item)
        self.morphs = infinite_random_data(morphs)

    def __next__(self):
        return next(self.morphs)


ExampleSelectors = dict(
    random=RandomExampleSelector,
    domain=DomainExampleSelector,
    morph=MorphExampleSelector,
    cmorph=ConstrainedMorphExampleSelector
)


def icl(eval_set, icl_set, template_kwargs, seed: int = 0, selector: str = "random", **kwargs):
    np.random.seed(seed)
    icl_gen = ExampleSelectors[selector](icl_set, **kwargs)
    for item in eval_set:
        icl_eg = [fill_template(eg, icl=True, **template_kwargs) for eg in icl_gen(item)]
        icl_eg.append(fill_template(item, icl=False, **template_kwargs))
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
    selector: str = "random"
    morph_lang: str = "fr"
    morph: str = None


def prompt(eval_set, icl_set, model, tokenizer, data_collator, src: str = "en", tgt: str = "fr",
           template_lang: str = "fr", template_form: str = "term", device="cuda", def_lang: str = "fr",
           data_kwargs: DataKwargs = DataKwargs(), gen_kwargs: GenKwargs = GenKwargs(), output_path: Path = None,
           **kwargs):
    preproc = Preprocessor(tgt)
    src_lang = LANGUAGES[template_lang][src]
    tgt_lang = LANGUAGES[template_lang][tgt]
    template = PROMPTS[template_lang][template_form]
    template_kwargs = dict(src_lang=src_lang, tgt_lang=tgt_lang, template=template, src=src, tgt=tgt, def_lang=def_lang)
    icl(eval_set, icl_set, template_kwargs, **kwargs)
    eval_set = DataLoader(eval_set, collate_fn=data_collator.collate_fn, **asdict(data_kwargs))
    output = evaluate(eval_set, model, tokenizer, gen_kwargs=asdict(gen_kwargs), preproc=preproc,
                      device=device)
    metrics = {}
    for k, v in output["metrics"].items():
        if isinstance(v, float):
            metrics[k] = v
    print(metrics)
    if output_path is not None:
        output["hyperparameters"] = dict(src=src, tgt=tgt, template_lang=template_lang, template_form=template_form,
                                         **kwargs)
        with open(output_path / f"output.json", 'at') as file:
            json.dump(output, file)
            file.write("\n")
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
            metrics.update(asdict(prompt_kwargs))
            results.append(metrics)
    print(results)
    if output_path is not None:
        mode = "a" if (output_path / "results.csv").exists() else "w"
        pd.DataFrame(results).to_csv(output_path / "results.csv", mode=mode)


if __name__ == "__main__":
    CLI(main, description=main.__doc__)
