# -*- coding: utf-8 -*-
from jsonargparse import CLI
import yaml

from . import trainee
from .utils import Path


def main(ckpt: Path = None, config_path: Path = None):
    """Save the PreTrainedModel(s) wrapped inside the Trainee (LightningModule)."""
    if config_path is None:
        assert ckpt is not None, "you must provide either ckpt or config (or both)"
        config_path = ckpt.parent.parent / 'config.yaml'
    with open(config_path, 'rt') as file:
        config = yaml.load(file, yaml.Loader)
    if ckpt is None:
        ckpt = Path(config["ckpt_path"])
    class_name = config['model']['class_path'].split('.')[-1]
    Class = getattr(trainee, class_name)
    model = Class.load_from_checkpoint(ckpt, **config['model']['init_args'])
    ckpt_path = ckpt.with_suffix('')
    model.model.save_pretrained(ckpt_path)


if __name__ == '__main__':
    CLI(main)
