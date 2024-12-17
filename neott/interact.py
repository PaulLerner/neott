from jsonargparse import CLI
from dataclasses import asdict
import json
from typing import List, Union
from string import Formatter
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from .trainee import ModelKwargs, GenKwargs
from .data.train import TokenizerKwargs, PromptKwargs
from .prompt import SelectorKwargs, SelectorFusion, get_template_kwargs, get_eos, post_proc, icl_item


def prompt(item, model, tokenizer, icl_gen, template_kwargs, gen_kwargs, tokenizer_kwargs, device="cuda", **kwargs):
    eos_token_id = get_eos(tokenizer)
    input_text, _ = icl_item(item, icl_gen, template_kwargs, ppl=False, **kwargs)
    inputs = tokenizer([input_text], **tokenizer_kwargs)
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    batch_size, seq_len = inputs["input_ids"].shape
    output = model.generate(eos_token_id=eos_token_id, return_dict_in_generate=True,
                            pad_token_id=tokenizer.pad_token_id, **inputs, **gen_kwargs).sequences
    # keep only newly generated tokens
    output = output[:, seq_len:].cpu()
    output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
    output_text = post_proc(output_text)
    return output_text[0]


def user_loop(required_fields, src: str = "en", def_lang: str = "fr", *args, **kwargs):
    while True:
        if 'src_term' in required_fields:
            src_term = input(f">>> Input the source term (in {src})\n").strip()
        else:
            src_term = None
        if 'src_def' in required_fields:
            src_def = input(f">>> Input the definition of the term (in {def_lang})\n").strip()
        else:
            src_def = None
        # N.B. it's ok if src == def_lang
        item = {"id": None, src: {}, def_lang: {}}
        item[src]["text"] = src_term
        item[def_lang]["def"] = {"text": src_def}
        output = prompt(item, *args, **kwargs)
        print(f"{output}\n")


def main(icl_path: Path, prompt_kwargs: PromptKwargs = PromptKwargs(), model_kwargs: ModelKwargs = ModelKwargs(),
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
    icl_gen = SelectorFusion(selector_kwargs, icl_set=icl_set, def_lang=prompt_kwargs.def_lang, src=prompt_kwargs.src)

    model = AutoModelForCausalLM.from_pretrained(**asdict(model_kwargs), trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=add_prefix_space, add_eos_token=False, trust_remote_code=True)

    template_kwargs = get_template_kwargs(src=prompt_kwargs.src, tgt=prompt_kwargs.tgt,
                                          template_lang=prompt_kwargs.template_lang,
                                          template_form=prompt_kwargs.template_form)
    # FIXME ensure that icl_gen (SelectorFusion) is compatible with required_fields
    required_fields = {field[1] for field in Formatter().parse(template_kwargs["template"])}
    user_loop(required_fields, model=model, tokenizer=tokenizer, icl_gen=icl_gen, template_kwargs=template_kwargs,
              gen_kwargs=asdict(gen_kwargs), tokenizer_kwargs=asdict(tokenizer_kwargs), src=prompt_kwargs.src,
              def_lang=prompt_kwargs.def_lang, device=model_kwargs.device_map)


if __name__ == "__main__":
    CLI(main, description=main.__doc__)
