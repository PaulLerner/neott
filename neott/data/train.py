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