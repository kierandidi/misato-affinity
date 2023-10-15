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
    # datamodule = MDDataModule(constants.MD_PATH,
    #                           constants.QM_PATH,
    #                           constants.AFFINITY_PATH,
    #                           constants.PAIR_PATH)
    datamodule.setup()

    # Validate the loaded data
    # train_dataloader = datamodule.train_dataloader()
    test_dataloader = datamodule.test_dataloader()
    logger.info("Loaded datamodule")
    logger.info(f"Number of training samples: {len(test_dataloader.dataset)}")
    print("test dataloader", test_dataloader.dataset)
    print(f"Number of training samples: {len(test_dataloader.dataset)}")
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
    print("\n \n \n RUN name", config.name, "\n \n \n")
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
    print("trainer", trainer)
    print("model", model)
    print("dataloader", test_dataloader)
    print("config", config)
    # trainer.fit(model, datamodule)
    # model = model.load_from_checkpoint("configs/runs/gcn2_logfrac_all_features_3GCN_epoch=6-val_loss=0.1448.ckpt")
    # Test the model at the best checkpoint
    # if config.test_model:
    # logger.info("Testing the model at the best checkpoint")
    # trainer.test(model=model,ckpt_path="configs/runs/gcn2_logfrac_all_features_3GCN_epoch=6-val_loss=0.1448.ckpt", dataloaders=test_dataloader)
    # predictions = []
    logger.info("Train loop completed. Predicting...")
    ckpt_path = config.load_from_checkpoint
    predictions = trainer.predict(
        model=model, ckpt_path="configs/runs/" + ckpt_path, dataloaders=test_dataloader
    )
    # predictions = trainer.predict(model=model,ckpt_path="configs/runs/gcn_ranknorm_uniform_epoch=6-val_loss=0.0498.ckpt", dataloaders=test_dataloader)
    # predictions = trainer.predict(model=model,ckpt_path="configs/runs/gcn_ranknorm_2_epoch=19-val_loss=0.0978.ckpt", dataloaders=test_dataloader)

    # predictions.append(trainer.predict(model=model,ckpt_path="configs/runs/gcn_ranknorm_2_epoch=19-val_loss=0.0978.ckpt", dataloaders=test_dataloader))

    pickle.dump(predictions, open("data/test_out/predictions_" + config.name + ".pickle", "wb"))
    # return config, predictions


if __name__ == "__main__":
    test()
