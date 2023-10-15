import pathlib

from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from src.constants import HYDRA_CONFIG_PATH

## Define own callbacks here


def get_callbacks(cfg):
    callacks = []
    # ckpt_dir = pathlib.Path('/p/project/hai_denovo/Projects/debug/misato-ba/configs/runs')
    ckpt_dir = pathlib.Path(HYDRA_CONFIG_PATH).joinpath("runs")
    if "checkpointing" in cfg.callbacks:
        ckpt_dir.mkdir(exist_ok=True)
        # Saves the top k checkpoints according to the test metric throughout
        # training.
        ckpt = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"{cfg.name}" + "_{epoch}-{val_loss:.4f}",  # {epoch}-{val_loss:.4f}",
            every_n_epochs=cfg.callbacks.checkpointing.checkpoint_freq,
            monitor=f"{cfg.trainer.eval_metrics}",
            save_top_k=cfg.callbacks.checkpointing.save_top_k,
            mode="min",
        )
        callacks.append(ckpt)

    # TODO add your own callbacks here

    return callacks
