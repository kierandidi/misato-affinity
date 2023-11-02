"""
Main module to load and train the model. This should be the program entry point.
"""
import pickle
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
def test(config: DictConfig):
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
    # train_dataloader = datamodule.train_dataloader()
    test_dataloader = datamodule.test_dataloader()
    logger.info("Loaded datamodule")
    logger.info(f"Number of training samples: {len(test_dataloader.dataset)}")

    # Load the model
    logger.info("Loading model")
    model = hydra.utils.instantiate(config.model)

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
        #        log_every_n_steps=config.trainer.log_steps,
        #        val_check_interval=config.trainer.val_interval,
        #        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        accelerator="gpu",
        devices=1,
    )

    logger.info("Train loop completed. Predicting...")
    ckpt_path = config.load_from_checkpoint
    predictions = trainer.predict(
        model=model, ckpt_path="configs/best_ckpts/" + ckpt_path, dataloaders=test_dataloader
    )
    pickle.dump(predictions, open("data/output/predictions_" + config.name + ".pickle", "wb"))
    # return config, predictions


if __name__ == "__main__":
    test()
