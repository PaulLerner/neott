from dataclasses import dataclass, asdict
import json
from typing import Optional, Union, List

from torch.utils.data import DataLoader

import lightning as pl
from transformers import AutoTokenizer

from ..metrics import Preprocessor
from ..morph.labels import MorphLabel

ICL_SEP = "###"
CHAT_USER_START = "<|im_start|>user\n"
CHAT_USER_END = "<|im_end|>\n<|im_start|>assistant\n"
PROMPTS = {
    "en": {
        # bawden and yvon
        "version": "If the original version says {src_term:s} then the {tgt_lang:s} version should say :{tgt_term:s}",
        # PL
        "term": "The term {src_term:s} can be translated in {tgt_lang:s} as :{tgt_term:s}",
        "def": "{src_def:s} defines the term :{tgt_term:s}",
        # bloomz (instruction)
        "tatoeba_mt": "Translate the following term from {src_lang:s} to {tgt_lang:s} {src_term:s} :{tgt_term:s}",
        "tower_base": "{src_lang:s} : {src_term:s}\n{tgt_lang:s} :{tgt_term:s}",
        "def_chat": "Which {tgt_lang:s} term is defined in the following way?\n{src_def:s}\n{tgt_term:s}",
        "def_chat2": "Generate a {tgt_lang:s} term that could be defined in the following way\n{src_def:s}\n{tgt_term:s}"
    },
    "fr": {
        "empty": ":{tgt_term:s}",
        # PL
        "term": "Le terme {src_lang:s} {src_term:s} peut se traduire en {tgt_lang:s} par :{tgt_term:s}",
        "def": "{src_def:s} définit le terme :{tgt_term:s}",
        "def_chat": "Quel terme est défini de la façon suivante ? (réponds seulement le terme, ne fait pas de phrases) : {src_def:s}\n{tgt_term:s}",
        "def_chat2": "Génère un terme qui pourrait être défini de la façon suivante (réponds seulement le terme, ne fait pas de phrases) : {src_def:s}\n{tgt_term:s}",
        "def_morph": "Génère un terme qui pourrait être défini de la façon suivante (réponds seulement le terme, ne fait pas de phrases) : {src_def:s}\nTu peux utiliser les procédés morphologiques suivants :\nLa préfixation, où un affixe est concaténé au début d’un mot pour en former un nouveau (pré+entraînement = préentraînement)\nLa suffixation, où l’affixation se fait à la fin du mot (généraliser+tion = généralisation)\nLa composition ordinaire, qui compose deux mots indépendants (timbre-poste)\nLa composition néoclassique, qui compose uniquement des morphèmes liés (azo+phile = azophile)\nLa composition syntagmatique, où des syntagmes qui suivent les règles syntaxiques de la langue se lexicalisent et donnent lieu à des termes, souvent non-compositionnels\n{tgt_term:s}",
        "def+term": "{src_def:s} définit le terme {src_lang:s} {src_term:s} qui peut se traduire en {tgt_lang:s} par :{tgt_term:s}",
        # bloomz (instruction)
        "tatoeba_mt": "Traduis le terme {src_lang:s} suivant en {tgt_lang:s} {src_term:s} :{tgt_term:s}"
    }
}
LANGUAGES = {
    "en": {"en": "English", "fr": "French"},
    "fr": {"en": "anglais", "fr": "français"}
}


def fill_template(item, template, icl=False, src="en", tgt="fr", src_lang="anglais", tgt_lang="français",
                  def_lang: str = "fr"):
    if icl:
        tgt_term = " " + item[tgt]["text"]
    else:
        tgt_term = ""
    return template.format(tgt_term=tgt_term, src_lang=src_lang, tgt_lang=tgt_lang,
                           src_term=item.get(src, {}).get("text"), src_def=item[def_lang]["def"]["text"])


class Identity:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x, *args, **kwargs):
        return x


class MorphCondition:
    def __init__(self, morph_lang, vocab, morph_key: str = 'morph_label', *args, **kwargs):
        self.morph_lang = morph_lang
        self.vocab = vocab
        self.morph_key = morph_key

    def __call__(self, input_text, item, *args, **kwargs):
        special_tokens = []
        for label in sorted(item[self.morph_lang][self.morph_key]):
            # FIXME: pattern for CroissantLLM: how to extend to other LLMs?
            special_token = f"<extra_id_{MorphLabel[label].value}>"
            assert special_token in self.vocab
            special_tokens.append(special_token)
        return "".join(special_tokens) + input_text


class ConstantMorphCondition:
    def __init__(self, morph: Union[str, List[str]], vocab, *args, **kwargs):
        if isinstance(morph, str):
            morph = [morph]
        special_tokens = []
        for label in sorted(morph):
            special_token = f"<extra_id_{MorphLabel[label].value}>"
            assert special_token in vocab
            special_tokens.append(special_token)
        self.special_token = "".join(special_tokens)

    def __call__(self, input_text, *args, **kwargs):
        return self.special_token + input_text


@dataclass
class PromptKwargs:
    seed: Union[int, List[int]] = 0
    src: Union[str, List[str]] = "en"
    tgt: Union[str, List[str]] = "fr"
    template_lang: Union[str, List[str]] = "fr"
    def_lang: Union[str, List[str]] = "fr"
    template_form: Union[str, List[str]] = "term"
    fallback_template: str = None
    chat: Union[bool, List[bool]] = False


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


MorphClass = {
    "morph": MorphCondition,
    "identity": Identity,
    "cmorph": ConstantMorphCondition
}


class DataModule(pl.LightningDataModule):
    def __init__(self, tokenizer_name: str = None, tokenizer_kwargs: TokenizerKwargs = TokenizerKwargs(),
                 add_prefix_space: bool = False, data_path: str = None, data_kwargs: DataKwargs = DataKwargs(),
                 prompt_kwargs: PromptKwargs = PromptKwargs(), filter_def: str = None, morph_lang: str = "fr",
                 morph: str = None, condition: str = "identity", morph_key: str = 'morph_label',
                 filter_morph: bool = False, split_syn: bool = False):
        super().__init__()
        self.non_tensor_keys = ["text", morph_key, "syn", "term_indices"]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=add_prefix_space,
                                                       add_eos_token=True, trust_remote_code=True)
        assert self.tokenizer.padding_side == 'left'
        prompt_sep = self.tokenizer.encode(':', add_special_tokens=False)
        assert len(prompt_sep) == 1, prompt_sep
        self.prompt_sep = prompt_sep[0]
        self.tokenizer_kwargs = asdict(tokenizer_kwargs)
        self.data_path = data_path
        self.dataset = {}
        self.data_kwargs = asdict(data_kwargs)
        self.src_lang = LANGUAGES[prompt_kwargs.template_lang][prompt_kwargs.src]
        self.tgt_lang = LANGUAGES[prompt_kwargs.template_lang][prompt_kwargs.tgt]
        self.template = PROMPTS[prompt_kwargs.template_lang][prompt_kwargs.template_form]
        if prompt_kwargs.fallback_template is not None:
            self.fallback_template = PROMPTS[prompt_kwargs.template_lang][prompt_kwargs.fallback_template]
        else:
            self.fallback_template = None
        self.prompt_kwargs = prompt_kwargs
        assert not prompt_kwargs.chat
        self.filter_def = filter_def
        self.filter_morph = filter_morph
        self.preproc = Preprocessor(prompt_kwargs.tgt)
        self.morph_lang = morph_lang
        self.morph = morph
        self.morph_key = morph_key
        self.split_syn = split_syn
        self.morph_condition = MorphClass[condition](vocab=self.tokenizer.vocab, morph_lang=self.morph_lang,
                                                     morph=self.morph, morph_key=self.morph_key)

    def prepare_data(self):
        print("loading data...")
        with open(self.data_path, 'rt') as file:
            self.dataset = json.load(file)

    def train_dataloader(self):
        if 'train' not in self.dataset:
            return None
        if self.filter_def is not None:
            before = len(self.dataset['train'])
            self.dataset['train'] = [item for item in self.dataset['train'] if item[self.filter_def]['def']['text']]
            print(f"filtered training set from {before} to {len(self.dataset['train'])} with {self.filter_def} definitions")
        if self.filter_morph:
            before = len(self.dataset['train'])
            self.dataset['train'] = [item for item in self.dataset['train'] if item[self.morph_lang][self.morph_key]]
            print(f"filtered training set from {before} to {len(self.dataset['train'])} with {self.morph_lang} morphs")

        return DataLoader(
            self.dataset['train'],
            collate_fn=self.train_collate_fn,
            shuffle=True,
            **self.data_kwargs
        )

    def val_dataloader(self):
        if 'dev' not in self.dataset:
            return None
        return DataLoader(
            self.dataset['dev'],
            collate_fn=self.eval_collate_fn,
            shuffle=False,
            **self.data_kwargs
        )

    def test_dataloader(self):
        if 'test' not in self.dataset:
            return None
        return DataLoader(
            self.dataset['test'],
            collate_fn=self.eval_collate_fn,
            shuffle=False,
            **self.data_kwargs
        )

    def train_collate_fn(self, items):
        self.tokenizer.add_eos_token = True
        input_texts = []
        for item in items:
            if self.filter_def is None and self.fallback_template is not None and not item[self.prompt_kwargs.def_lang]['def']['text']:
                template = self.fallback_template
            else:
                template = self.template
            # icl -> target is part of the input (to be causal-masked)
            input_text = fill_template(item, template, icl=True, src=self.prompt_kwargs.src,
                                       tgt=self.prompt_kwargs.tgt, src_lang=self.src_lang,
                                       tgt_lang=self.tgt_lang, def_lang=self.prompt_kwargs.def_lang)
            input_text = self.morph_condition(input_text, item, self.morph_lang, self.tokenizer.vocab)
            input_texts.append(input_text)
        inputs = self.tokenizer(input_texts, **self.tokenizer_kwargs)
        labels = inputs['input_ids'].clone()
        for label in labels:
            where = (label == self.prompt_sep).nonzero()
            # did not find prompt_sep in input (only legal if tokenizer truncates) -> mask the whole label
            if where.shape[0] == 0:
                where = label.shape[0] - 1
            else:
                where = where[-1, 0]
            label[: where + 1] = self.trainer.lightning_module.loss_fct.ignore_index
        inputs["labels"] = labels
        return inputs

    def eval_collate_fn(self, items):
        self.tokenizer.add_eos_token = False
        input_texts = []
        keep = {k: [] for k in self.non_tensor_keys}
        for item in items:
            term_indices = []
            # no icl -> target is not part of the input (to be auto-regressively decoded)
            input_text = fill_template(item, self.template, icl=False, src=self.prompt_kwargs.src,
                                       tgt=self.prompt_kwargs.tgt, src_lang=self.src_lang,
                                       tgt_lang=self.tgt_lang, def_lang=self.prompt_kwargs.def_lang)
            input_text = self.morph_condition(input_text, item, self.morph_lang, self.tokenizer.vocab)
            term_indices.append(len(input_texts))
            input_texts.append(input_text)
            for k in ["text", self.morph_key]:
                keep[k].append(item[self.prompt_kwargs.tgt][k])

            # duplicate term for every morph variant
            if self.split_syn:
                assert len(item[self.prompt_kwargs.tgt][self.morph_key]) == 1
                syns_per_morph = {}
                for syn in item[self.prompt_kwargs.tgt]['syn']:
                    assert len(syn[self.morph_key]) == 1
                    syns_per_morph.setdefault(syn[self.morph_key][0], [])
                    syns_per_morph[syn[self.morph_key][0]].append(syn)
                # the synonym of the canonical term are only those that have the same morph
                keep["syn"].append([syn["text"] for syn in syns_per_morph.pop(item[self.prompt_kwargs.tgt][self.morph_key][0], [])])

                for _, syns in syns_per_morph.items():
                    # it does not matter which synonym we keep, they all have the same input
                    syn = syns.pop()
                    syn[self.prompt_kwargs.src] = {"text": item[self.prompt_kwargs.src]['text']}
                    syn.setdefault(self.prompt_kwargs.def_lang, {})
                    syn[self.prompt_kwargs.def_lang]["def"] = {"text": item[self.prompt_kwargs.def_lang]["def"]["text"]}
                    syn[self.morph_lang][self.morph_key] = syn[self.morph_key]

                    input_text = fill_template(syn, self.template, icl=False, src=self.prompt_kwargs.src,
                                               tgt=self.prompt_kwargs.tgt, src_lang=self.src_lang,
                                               tgt_lang=self.tgt_lang, def_lang=self.prompt_kwargs.def_lang)
                    input_text = self.morph_condition(input_text, syn, self.morph_lang, self.tokenizer.vocab)
                    term_indices.append(len(input_texts))
                    input_texts.append(input_text)
                    for k in ["text", self.morph_key]:
                        keep[k].append(item[self.prompt_kwargs.tgt][k])
                    # other synonyms with the same morph are kept to compute soft EM
                    keep["syn"].append([syn["text"] for syn in syns])

            # all synonyms of the term, regardless of morph
            else:
                keep["syn"].append([syn["text"] for syn in item[self.prompt_kwargs.tgt].get('syn', [])])

            keep["term_indices"].append(term_indices)

        inputs = self.tokenizer(input_texts, **self.tokenizer_kwargs)
        inputs.update(keep)
        return inputs

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Keep strings etc in batch. Does not try to cast them as Tensor of any dtype or device."""
        keep = {}
        for k in self.non_tensor_keys:
            if k in batch:
                keep[k] = batch.pop(k)
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        batch.update(keep)
        return batch
