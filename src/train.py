"""
Main module to load and train the model. This should be the program entry point.
"""
import warnings

import hydra
import torch
import wandb
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from src import constants
from src.utils.callbacks import get_callbacks
from src.utils.logutils import get_lightning_logger, get_logger

warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

# Logger initialization
logger = get_logger(__name__)


@hydra.main(
    config_path=constants.HYDRA_CONFIG_PATH,
    config_name=constants.HYDRA_CONFIG_NAME,
    version_base=constants.HYDRA_VERSION_BASE,
)
def train(config: DictConfig):
    """
    Train model with PyTorch Lightning and log with Wandb.

    Parameters:
    - config: A configuration object with the training parameters.
    """
    # Set random seeds
    seed_everything(config.seed)
    logger.info("Set random seed to %d", config.seed)
    torch.set_float32_matmul_precision("high")

    # Load data module
    datamodule = hydra.utils.instantiate(config.datamodule)
    datamodule.setup()

    # Validate the loaded data
    train_dataloader = datamodule.train_dataloader()
    logger.info("Loaded datamodule")
    logger.info(f"Number of training samples: {len(train_dataloader.dataset)}")

    # Load the model
    logger.info("Loading model")
    model = hydra.utils.instantiate(config.model)

    # Setup logging and checkpointing
    # pl_logger = get_lightning_logger(config)
    if config.name == "test":
        wandb_logger = None
    else:
        wandb_logger = WandbLogger(
            project=config.logger.project,
            entity=config.logger.entity,
            name=config.name,
            offline=True,
        )

    callbacks = get_callbacks(config)

    # Instantiate and start the Trainer
    trainer = Trainer(
        callbacks=callbacks,
        max_epochs=config.trainer.epochs,
        logger=wandb_logger,
        log_every_n_steps=config.trainer.log_steps,
        val_check_interval=config.trainer.val_interval,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        accelerator="gpu",
        devices=4,
    )
    trainer.fit(model, datamodule)

    # Test the model at the best checkpoint
    #if config.test_model:
    #    logger.info("Testing the model at the best checkpoint")
    #    trainer.test(ckpt_path="best")
    #    logger.info("Train loop completed. Exiting.")


if __name__ == "__main__":
    train()
