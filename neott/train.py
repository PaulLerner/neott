import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI


def main():
    cli = LightningCLI(
        trainer_class=pl.Trainer,
        seed_everything_default=0
    )
    return cli


if __name__ == "__main__":
    main()