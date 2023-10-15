from src.models.sample_model import SampleModel
from src.utils.logutils import get_logger

logger = get_logger(__name__)


def get_lightning_model(cfg):
    """Models a Lightning Model giving a config object

    Args:
        cfg (config): Whole experiment config
    """
    # If continuing a previous run:
    if cfg.load_from_checkpoint is not None:
        logger.info("Loading model from checkpoint %s", cfg.load_from_checkpoint)
        model = SampleModel.load_from_checkpoint(cfg.load_from_checkpoint)
        model.config = cfg  # NOTE  will this break things?
    else:
        logger.info("Initialising new model")
        model = SampleModel(cfg)  # TODO

    return model


def get_model(cfg):
    """Models a PyTorch model giving a config object

    Args:
        cfg (config): Whole experiment config
    """

    model_cfg = cfg.model

    if model_cfg.name == "egnn":
        model = EGNN(**model_cfg)

    return model
