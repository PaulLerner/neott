from dataclasses import dataclass, asdict
import warnings
from functools import partial
from pathlib import Path
from typing import Optional, Union
import os
import json

from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, PretrainedConfig
import pandas as pd
import torch


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
class GenKwargs:
    num_beams: int = 4
    max_new_tokens: int = 64
    num_return_sequences: int = 1


class LinearLRWithWarmup(LambdaLR):
    """
    Linear learning rate scheduler with linear warmup.
    Adapted from https://github.com/huggingface/transformers/blob/v4.23.0/src/transformers/optimization.py#L75

    Parameters
    ----------
    *args, **kwargs: additionnal arguments are passed to LambdaLR
    warmup_steps: int
    total_steps: int
    """

    def __init__(self, *args, warmup_steps, total_steps, **kwargs):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(*args, **kwargs, lr_lambda=self.lr_lambda)

    def lr_lambda(self, current_step: int):
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))
        return max(
            0.0,
            float(self.total_steps - current_step) / float(max(1, self.total_steps - self.warmup_steps))
        )


def to_latex(metrics):
    table = pd.DataFrame([metrics]) * 100
    return table.to_latex(float_format='%.1f')


def batched_cpu(batch):
    return {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


class Trainee(pl.LightningModule):
    def __init__(self, *args, model_kwargs, gradient_checkpointing=False, warmup_steps=0, lr=2e-5, betas=(0.9, 0.999),
                 eps=1e-08, weight_decay=0.0, output_cpu=False, gen_kwargs: GenKwargs = GenKwargs(), **kwargs):
        super().__init__(*args, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        # scheduling and optimization
        self.warmup_steps = warmup_steps
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.param_groups = self.parameters()
        self.output_cpu = output_cpu
        if gradient_checkpointing:
            self.gradient_checkpointing_enable()
        self.loss_fct = CrossEntropyLoss()

    def step(self, batch, batch_idx):
        prompt_len = batch.pop("prompt_len")
        # there should be at least one token (e.g. <BOS>) in the prompt
        assert prompt_len > 0
        logits = self.model(**batch)
        # FIXME only works with left-padding
        # compute loss on generated tokens only: there's one token shift between input and output (causal LM)
        logits = logits[:, prompt_len - 1:].contiguous()
        labels = batch["input_ids"][:, prompt_len:].contiguous()
        loss = self.loss_fct(logits, labels)
        return dict(loss=loss, logits=logits)

    def eval_step(self, batch, batch_idx):
        # TODO generate
        return self.step(batch, batch_idx)

    def log(self, name, value, **kwargs):
        """Ignores None values."""
        if value is None:
            return None
        super().log(name, value, **kwargs)

    def training_step(self, batch, batch_idx):
        """Step and log training metrics"""
        outputs = self.step(batch, batch_idx)
        self.log("train/loss", outputs['loss'])
        return outputs

    def validation_step(self, batch, batch_idx):
        """Step and log validation metrics"""
        outputs = self.eval_step(batch, batch_idx)
        self.log("eval/loss", outputs['loss'])
        if self.output_cpu:
            return batched_cpu(outputs)
        return outputs

    def test_step(self, batch, batch_idx):
        """Step and log test metrics"""
        outputs = self.eval_step(batch, batch_idx)
        self.log("test/loss", outputs['loss'])
        if self.output_cpu:
            return batched_cpu(outputs)
        return outputs

    def eval_epoch_end(self, eval_outputs):
        warnings.warn("eval_epoch_end is not implemented.")
        return {}

    def validation_epoch_end(self, *args, **kwargs):
        """eval_epoch_end and log"""
        metrics = self.eval_epoch_end(*args, **kwargs)['metrics']
        for k, v in metrics.items():
            self.log(f"eval/{k}", v)

    def test_epoch_end(self, *args, **kwargs):
        """eval_epoch_end and log"""
        metrics = self.eval_epoch_end(*args, **kwargs)['metrics']
        print(to_latex(metrics))
        log_dir = Path(self.trainer.log_dir)
        for k, v in metrics.items():
            self.log(f"test/{k}", v)
        with open(log_dir / 'metrics.json', 'wt') as file:
            json.dump(metrics, file)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer = AdamW(self.param_groups, lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay)

        # FIXME: this will be overwritten when loading state from ckpt_path
        # so if you want to keep training by increasing total_steps,
        # your LR will be 0 if the ckpt reached the previously set total_steps
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = LinearLRWithWarmup(
            optimizer,
            warmup_steps=self.warmup_steps, total_steps=total_steps
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    #####################################################
    # gradient checkpointing: adapted from transformers #
    #####################################################
    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def gradient_checkpointing_enable(self):
        """
        Activates gradient checkpointing for the current model.
        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        self.apply(partial(self._set_gradient_checkpointing, value=True))

    def gradient_checkpointing_disable(self):
        """
        Deactivates gradient checkpointing for the current model.
        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if self.supports_gradient_checkpointing:
            self.apply(partial(self._set_gradient_checkpointing, value=False))

    @property
    def is_gradient_checkpointing(self) -> bool:
        """
        Whether gradient checkpointing is activated for this model or not.
        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        return any(getattr(m, "gradient_checkpointing", False) for m in self.modules())
