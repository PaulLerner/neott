import pandas as pd
import torch
from jsonargparse import CLI
import json
from dataclasses import asdict

from torch.nn import CrossEntropyLoss
from tqdm import tqdm

import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import infinite_random_data, all_size_combination, Path, ListOrArg, iter_kwargs_prod
from .metrics import compute_metrics, Preprocessor
from .morph.labels import MorphLabel
from .trainee import ModelKwargs, GenKwargs
from .data.train import (TokenizerKwargs, DataKwargs, PromptKwargs, PROMPTS, ICL_SEP, LANGUAGES, fill_template,
                         CHAT_USER_START, CHAT_USER_END)


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


class LongestStartExampleSelector(ExampleSelector):
    def __init__(self, *args, def_lang: str = "fr", **kwargs):
        super().__init__(*args, **kwargs)
        self.def_lang = def_lang
        self.definitions = np.array([item[self.def_lang]["def"]["text"] for item in self.icl_set])
        self.indices = self.definitions.argsort()

    def __call__(self, item):
        definition = item[self.def_lang]["def"]["text"]
        i = self.definitions.searchsorted(definition, sorter=self.indices)
        common_chars = []
        # the closest is i but the n_icl closest may be before or after
        for j in self.indices[i-self.n_icl: i+self.n_icl]:
            d = self.definitions[j]
            eg = self.icl_set[j]
            # do not use self in the prompt (may happen if using eval_set as icl_set)
            if eg["id"] == item["id"]:
                common_chars.append(-1)
                continue
            cs = 0
            for c_p, c_c in zip(definition, d):
                if c_p != c_c:
                    break
                cs += 1
            common_chars.append(cs)

        for j in self.indices[i-self.n_icl: i+self.n_icl][(-np.array(common_chars)).argsort()[:self.n_icl]]:
            yield self.icl_set[j]


class DomainExampleSelector(ExampleSelector):
    def __init__(self, *args, domain_key="Dom", **kwargs):
        super().__init__(*args, **kwargs)
        self.domain_key = domain_key
        domains = {}
        for item in self.icl_set:
            item_domains = item.get(self.domain_key) if item.get(self.domain_key) is not None else [None]
            for domain in item_domains:
                domains.setdefault(domain, [])
                domains[domain].append(item)
        domain_sizes = {}
        for domain, domain_icl_set in domains.items():
            domain_sizes[domain] = len(domain_icl_set)
            domains[domain] = infinite_random_data(domain_icl_set)
        self.domains = domains
        self.domain_sizes = domain_sizes
        print(self.domain_key, self.domain_sizes)
        self.fallback = infinite_random_data(self.icl_set)

    def __next__(self):
        # note this gives oracle domain -> should be considered oracle/topline
        domain = self.item.get(self.domain_key)
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


class ConstrainedMorphExampleSelector(ExampleSelector):
    def __init__(self, *args, morph_lang: str = "fr", morph: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lang = morph_lang
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
    cmorph=ConstrainedMorphExampleSelector,
    start=LongestStartExampleSelector
)


def icl(eval_set, icl_set, template_kwargs, seed: int = 0, selector: str = "random", ppl: bool = False,
        chat: bool = False, def_lang: str = "fr", **kwargs):
    np.random.seed(seed)
    icl_gen = ExampleSelectors[selector](icl_set, def_lang=def_lang, **kwargs)
    for item in eval_set:
        icl_eg = [fill_template(eg, icl=True, def_lang=def_lang, **template_kwargs) for eg in icl_gen(item)]
        icl_eg.append(fill_template(item, icl=ppl, def_lang=def_lang, **template_kwargs))
        item["input_text"] = f" {ICL_SEP} ".join(icl_eg)
        if chat:
            item["input_text"] = CHAT_USER_START + item["input_text"] + CHAT_USER_END


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


def compute_ppl(eval_set, model, tokenizer, device="cuda"):
    loss_fct = CrossEntropyLoss(reduction="none")
    prompt_sep = tokenizer.encode(':', add_special_tokens=False)
    assert len(prompt_sep) == 1, prompt_sep
    prompt_sep = prompt_sep[0]
    losses, all_logits = [], []
    for inputs in tqdm(eval_set):
        batch_size, seq_len = inputs["input_ids"].shape
        inputs.pop("target_text")
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = inputs['input_ids'].clone()
        for label in labels:
            where = (label == prompt_sep).nonzero()
            # no prompt_sep with "empty" template
            if where.shape[0] == 0:
                continue
            label[: where[-1, 0] + 1] = loss_fct.ignore_index

        with torch.no_grad():
            logits = model(**inputs, return_dict=True).logits
        # there's one token shift between input and output (causal LM)
        logits = logits[:, :-1].contiguous().view(-1, model.config.vocab_size)
        labels = labels[:, 1:].contiguous().view(-1)
        loss = loss_fct(logits, labels).view(batch_size, seq_len-1).cpu()
        losses.append(loss)
        logits = logits.view(batch_size, seq_len - 1, model.config.vocab_size)
        labels = labels.view(batch_size, seq_len - 1)
        all_logits.append(logits[labels != loss_fct.ignore_index].cpu())
        # TODO compute PPL from cross-entropy/normalize with #chars
    return losses, all_logits


def evaluate(eval_set, model, tokenizer, gen_kwargs, preproc, device="cuda"):
    predictions, targets = [], []
    icl_sep_id = tokenizer.encode(ICL_SEP, add_special_tokens=False)
    assert len(icl_sep_id) == 1, icl_sep_id
    assert tokenizer.eos_token_id is not None
    eos_token_id = [tokenizer.eos_token_id, icl_sep_id[0]]
    for inputs in tqdm(eval_set):
        batch_size, seq_len = inputs["input_ids"].shape
        target_text = inputs.pop("target_text")
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        output = model.generate(eos_token_id=eos_token_id, return_dict_in_generate=True,
                                pad_token_id=tokenizer.pad_token_id, **inputs, **gen_kwargs).sequences
        # keep only newly generated tokens
        if tokenizer.padding_side == 'left':
            output = output[:, seq_len:]
        else:
            raise NotImplementedError(f"{tokenizer.padding_side=}")
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
        # FIXME: maybe refactor with apply_chat_template?
        # https://huggingface.co/docs/transformers/main/en/chat_templating
        inputs = self.tokenizer([item["input_text"] for item in items], **self.kwargs)
        inputs["target_text"] = [item[self.tgt]["text"] for item in items]
        return inputs


def prompt(eval_set, icl_set, model, tokenizer, data_collator, src: str = "en", tgt: str = "fr",
           template_lang: str = "fr", template_form: str = "term", device="cuda",
           data_kwargs: DataKwargs = DataKwargs(), gen_kwargs: GenKwargs = GenKwargs(), output_path: Path = None,
           ppl: bool = False, **kwargs):
    preproc = Preprocessor(tgt)
    src_lang = LANGUAGES[template_lang][src]
    tgt_lang = LANGUAGES[template_lang][tgt]
    template = PROMPTS[template_lang][template_form]
    template_kwargs = dict(src_lang=src_lang, tgt_lang=tgt_lang, template=template, src=src, tgt=tgt)
    icl(eval_set, icl_set, template_kwargs, ppl=ppl, **kwargs)
    eval_set = DataLoader(eval_set, collate_fn=data_collator.collate_fn, shuffle=False, **asdict(data_kwargs))
    if ppl:
        losses, all_logits = compute_ppl(eval_set, model, tokenizer, device=device)
        torch.save(losses, output_path/"losses.bin")
        torch.save(all_logits, output_path/"logits.bin")
        return {}
    else:
        output = evaluate(eval_set, model, tokenizer, gen_kwargs=asdict(gen_kwargs), preproc=preproc, device=device)
    metrics = {}
    for k, v in output["metrics"].items():
        if isinstance(v, float):
            metrics[k] = v
    print(metrics)
    if output_path is not None:
        output["hyperparameters"] = dict(template_lang=template_lang, template_form=template_form) | template_kwargs | kwargs
        with open(output_path / f"output.json", 'at') as file:
            json.dump(output, file)
            file.write("\n")
    return metrics


def main(data_path: str, eval_set: str = "dev", icl_set: str = "train", prompt_kwargs: PromptKwargs = PromptKwargs(),
         model_kwargs: ModelKwargs = ModelKwargs(), data_kwargs: DataKwargs = DataKwargs(), tokenizer_name: str = None,
         tokenizer_kwargs: TokenizerKwargs = TokenizerKwargs(), add_prefix_space: bool = False,
         gen_kwargs: GenKwargs = GenKwargs(), output_path: Path = None, filter_def: str = None, ppl: bool = False):
    """Prompt LLMs to generate terms (by translating them and/or given their definition)"""
    assert not (prompt_kwargs.chat and ppl)
    hyperparameters = dict(data_path=data_path, eval_set=eval_set, icl_set=icl_set, model=model_kwargs.pretrained_model_name_or_path)
    if tokenizer_name is None:
        tokenizer_name = model_kwargs.pretrained_model_name_or_path
    output_path.mkdir(exist_ok=True)
    with open(data_path, 'rt') as file:
        data = json.load(file)
    eval_set = data[eval_set]
    icl_set = data[icl_set]
    if filter_def is not None:
        before = len(icl_set)
        icl_set = [item for item in icl_set if item[filter_def]['def']['text']]
        print(f"filtered training set from {before} to {len(icl_set)} with {filter_def} definitions")

    model = AutoModelForCausalLM.from_pretrained(**asdict(model_kwargs))
    if not model_kwargs.load_in_8bit:
        model = model.to(model_kwargs.device_map)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=add_prefix_space, add_eos_token=ppl)
    data_collator = DataCollator(tokenizer, tgt=prompt_kwargs.tgt, **asdict(tokenizer_kwargs))
    prompt_kwargs = asdict(prompt_kwargs)
    for k, v in prompt_kwargs.items():
        prompt_kwargs[k] = ListOrArg(v)
    results = []
    for kwarg in iter_kwargs_prod(prompt_kwargs):
        if kwarg["template_form"] not in PROMPTS[kwarg["template_lang"]]:
            continue
        metrics = prompt(eval_set, icl_set, model, tokenizer, data_collator, **kwarg,
                         device=model_kwargs.device_map, data_kwargs=data_kwargs, gen_kwargs=gen_kwargs,
                         output_path=output_path, ppl=ppl)
        metrics.update(kwarg|hyperparameters)
        results.append(metrics)
    print(results)
    if output_path is not None:
        pd.DataFrame(results).to_csv(output_path / "results.csv", mode="a")


if __name__ == "__main__":
    CLI(main, description=main.__doc__)
