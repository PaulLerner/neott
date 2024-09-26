# -*- coding: utf-8 -*-
from jsonargparse import CLI
from pathlib import Path
from .utils import load_lightning


def main(ckpt: Path = None, config_path: Path = None):
    """Save the PreTrainedModel(s) wrapped inside the Trainee (LightningModule)."""
    model = load_lightning(ckpt=ckpt, config_path=config_path)
    ckpt_path = ckpt.with_suffix('')
    model.model.save_pretrained(ckpt_path)


if __name__ == '__main__':
    CLI(main)
