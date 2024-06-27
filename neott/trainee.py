from dataclasses import dataclass, asdict
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
import torch

from .metrics import compute_metrics


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


def batched_cpu(batch):
    return {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


class Trainee(pl.LightningModule):
    def __init__(self, *args, model_kwargs, gradient_checkpointing=False, warmup_steps=0, lr=2e-5, betas=(0.9, 0.999),
                 eps=1e-08, weight_decay=0.0, gen_kwargs: GenKwargs = GenKwargs(), **kwargs):
        super().__init__(*args, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        # scheduling and optimization
        self.warmup_steps = warmup_steps
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.param_groups = self.parameters()
        if gradient_checkpointing:
            self.gradient_checkpointing_enable()
        self.loss_fct = CrossEntropyLoss()
        if not isinstance(gen_kwargs, dict):
            gen_kwargs = asdict(gen_kwargs)
        self.gen_kwargs = gen_kwargs

    def training_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        logits = self.model(**batch, return_dict=True).logits
        # there's one token shift between input and output (causal LM)
        logits = logits[:, :-1].contiguous().view(-1, self.model.config.vocab_size)
        labels = labels[:, 1:].contiguous().view(-1)
        loss = self.loss_fct(logits, labels)
        self.log("train/loss", loss)
        return dict(loss=loss)

    def eval_step(self, batch, batch_idx):
        keep = {}
        for k in self.trainer.datamodule.non_tensor_keys:
            keep[k] = batch.pop(k, None)
        batch_size, seq_len = batch["input_ids"].shape
        output = self.model.generate(return_dict_in_generate=True,
                                     pad_token_id=self.trainer.datamodule.tokenizer.pad_token_id,
                                     **batch, **self.gen_kwargs).sequences
        # keep only newly generated tokens
        output = output[:, seq_len:]
        predictions = self.trainer.datamodule.tokenizer.batch_decode(output, skip_special_tokens=True)
        k = self.gen_kwargs["num_return_sequences"]
        assert len(predictions) % k == 0
        predictions_per_input = []
        for i in range(0, len(predictions), k):
            predictions_per_input.append(predictions[i: i + k])
        return dict(predictions=predictions_per_input, **keep)

    def log(self, name, value, **kwargs):
        """Ignores None values."""
        if value is None:
            return None
        super().log(name, value, **kwargs)

    def validation_step(self, batch, batch_idx):
        """Step and log validation metrics"""
        outputs = self.eval_step(batch, batch_idx)
        return outputs

    def test_step(self, batch, batch_idx):
        """Step and log test metrics"""
        outputs = self.eval_step(batch, batch_idx)
        return outputs

    def eval_epoch_end(self, eval_outputs):
        predictions, targets, syns, morphs = [], [], [], []
        for eval_output in eval_outputs:
            predictions.extend(eval_output["predictions"])
            targets.extend(eval_output["text"])
            morphs.extend(eval_output[self.trainer.datamodule.morph_key])
            syns.extend(eval_output["syn"])
        metrics = compute_metrics(predictions, targets, syns, self.trainer.datamodule.preproc, morphs=morphs)
        return {'metrics': metrics, 'predictions': predictions}

    def validation_epoch_end(self, *args, **kwargs):
        """eval_epoch_end and log"""
        metrics = self.eval_epoch_end(*args, **kwargs)['metrics']
        for k, v in metrics.items():
            if isinstance(v, float):
                self.log(f"eval/{k}", v)

    def test_epoch_end(self, *args, **kwargs):
        """eval_epoch_end and log"""
        output = self.eval_epoch_end(*args, **kwargs)
        log_dir = Path(self.trainer.log_dir)
        for k, v in output['metrics'].items():
            if isinstance(v, float):
                self.log(f"test/{k}", v)
        with open(log_dir / 'output.json', 'wt') as file:
            json.dump(output, file)

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
