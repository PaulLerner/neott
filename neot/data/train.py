from dataclasses import dataclass, asdict
import json
from typing import Optional, Union, List

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from transformers import AutoTokenizer

from neot.metrics import Preprocessor
from neot.morph.labels import MorphLabel

ICL_SEP = "###"
CHAT_USER_START = "<|im_start|>user\n"
CHAT_USER_END = "<|im_end|>\n<|im_start|>assistant\n"
PROMPTS = {
    "en": {
        # bawden and yvon
        "version": "If the original version says {src_term} then the {tgt_lang} version should say :{tgt_term}",
        # PL
        "term": "The term {src_term} can be translated in {tgt_lang} as :{tgt_term}",
        # bloomz (instruction)
        "tatoeba_mt": "Translate the following term from {src_lang} to {tgt_lang} {src_term} :{tgt_term}",
        "tower_base": "{src_lang} : {src_term}\n{tgt_lang} :{tgt_term}",
        "def_chat": "Which {tgt_lang} term is defined in the following way?\n{src_def}\n{tgt_term}",
        "def_chat2": "Generate a {tgt_lang} term that could be defined in the following way\n{src_def}\n{tgt_term}"
    },
    "fr": {
        "empty": ":{tgt_term}",
        # PL
        "term": "Le terme {src_lang} {src_term} peut se traduire en {tgt_lang} par :{tgt_term}",
        "def": "{src_def} définit le terme :{tgt_term}",
        "def_chat": "Quel terme est défini de la façon suivante ? (réponds seulement le terme, ne fait pas de phrases) : {src_def}\n{tgt_term}",
        "def_chat2": "Génère un terme qui pourrait être défini de la façon suivante (réponds seulement le terme, ne fait pas de phrases) : {src_def}\n{tgt_term}",
        "def_morph": "Génère un terme qui pourrait être défini de la façon suivante (réponds seulement le terme, ne fait pas de phrases) : {src_def}\nTu peux utiliser les procédés morphologiques suivants :\nLa préfixation, où un affixe est concaténé au début d’un mot pour en former un nouveau (pré+entraînement = préentraînement)\nLa suffixation, où l’affixation se fait à la fin du mot (généraliser+tion = généralisation)\nLa composition ordinaire, qui compose deux mots indépendants (timbre-poste)\nLa composition néoclassique, qui compose uniquement des morphèmes liés (azo+phile = azophile)\nLa composition syntagmatique, où des syntagmes qui suivent les règles syntaxiques de la langue se lexicalisent et donnent lieu à des termes, souvent non-compositionnels\n{tgt_term}",
        "def+term": "{src_def} définit le terme {src_lang} {src_term} qui peut se traduire en {tgt_lang} par :{tgt_term}",
        # bloomz (instruction)
        "tatoeba_mt": "Traduis le terme {src_lang} suivant en {tgt_lang} {src_term} :{tgt_term}"
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
    return template.format(tgt_term=tgt_term, src_lang=src_lang, tgt_lang=tgt_lang, src_term=item[src]["text"],
                           src_def=item[def_lang]["def"]["text"])


class Identity:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x, *args, **kwargs):
        return x


class MorphCondition:
    def __init__(self, morph_lang, vocab, *args, **kwargs):
        self.morph_lang = morph_lang
        self.vocab = vocab

    def __call__(self, input_text, item, *args, **kwargs):
        special_tokens = []
        for label in item[self.morph_lang]["morph_label"]:
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
        for label in morph:
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
    n_icl: Union[int, List[int]] = 5
    template_lang: Union[str, List[str]] = "fr"
    def_lang: Union[str, List[str]] = "fr"
    template_form: Union[str, List[str]] = "term"
    fallback_template: str = None
    selector: Union[str, List[str]] = "random"
    domain_key: Union[str, List[str]] = "Dom"
    morph_lang: Union[str, List[str]] = "fr"
    morph: Union[str, List[str]] = None
    chat: bool = False


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
    "random": Identity,
    "cmorph": ConstantMorphCondition
}


class DataModule(pl.LightningDataModule):
    def __init__(self, tokenizer_name: str = None, tokenizer_kwargs: TokenizerKwargs = TokenizerKwargs(),
                 add_prefix_space: bool = False, data_path: str = None, data_kwargs: DataKwargs = DataKwargs(),
                 prompt_kwargs: PromptKwargs = PromptKwargs(), filter_def: str = None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=add_prefix_space,
                                                       add_eos_token=True)
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
        # FIXME: tower_instruct template uses "\n" as delimiter
        assert prompt_kwargs.template_form != "tower_instruct"
        self.template = PROMPTS[prompt_kwargs.template_lang][prompt_kwargs.template_form]
        if prompt_kwargs.fallback_template is not None:
            assert prompt_kwargs.fallback_template != "tower_instruct"
            self.fallback_template = PROMPTS[prompt_kwargs.template_lang][prompt_kwargs.fallback_template]
        else:
            self.fallback_template = None
        self.prompt_kwargs = prompt_kwargs
        self.filter_def = filter_def
        self.preproc = Preprocessor(prompt_kwargs.tgt)
        self.morph_condition = MorphClass[self.prompt_kwargs.selector](vocab=self.tokenizer.vocab,
                                                                       morph_lang=self.prompt_kwargs.morph_lang,
                                                                       morph=self.prompt_kwargs.morph)

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
            print(
                f"filtered training set from {before} to {len(self.dataset['train'])} with {self.filter_def} definitions")
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
            input_text = self.morph_condition(input_text, item, self.prompt_kwargs.morph_lang, self.tokenizer.vocab)
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
        input_texts, target_texts = [], []
        for item in items:
            # no icl -> target is not part of the input (to be auto-regressively decoded)
            input_text = fill_template(item, self.template, icl=False, src=self.prompt_kwargs.src,
                                       tgt=self.prompt_kwargs.tgt, src_lang=self.src_lang,
                                       tgt_lang=self.tgt_lang, def_lang=self.prompt_kwargs.def_lang)
            input_text = self.morph_condition(input_text, item, self.prompt_kwargs.morph_lang, self.tokenizer.vocab)
            input_texts.append(input_text)
            target_texts.append(item[self.prompt_kwargs.tgt]["text"])
        inputs = self.tokenizer(input_texts, **self.tokenizer_kwargs)
        inputs["target_text"] = target_texts
        return inputs

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Keep target_text in batch. Does not try to cast them as Tensor of any dtype or device."""
        target_text = batch.pop('target_text', None)
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        if target_text is not None:
            batch['target_text'] = target_text
        return batch
