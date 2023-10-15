"""Module to validate the hydra config."""
import psutil
import torch
from omegaconf import DictConfig, OmegaConf

from src.utils.logutils import get_logger

logger = get_logger(__name__)


def validate_config(cfg: DictConfig) -> DictConfig:
    """Validate the config and make any necessary alterations to the parameters."""
    if cfg.run_name is None:
        raise TypeError("The `run_name` argument is mandatory.")

    # Make sure num_workers isn't too high.
    core_count = psutil.cpu_count(logical=False)
    if cfg.num_workers > core_count:
        logger.debug(
            (
                "Requested CPUs: %s. Avialable CPUs (physical): %s. "
                "Requested CPU count will therefore be set to maximum number of"
                "available physical cores. NOTE: It is recommended to use N-1"
                "cores or less to avoid memory flush overheads."
            ),
            cfg.num_workers,
            core_count,
        )
        cfg.num_workers = core_count

    # Make sure cuda config is correct
    if not cfg.cuda:
        cfg.gpus = 0
    if cfg.gpus <= 1:
        cfg.parallel_engine = None
    logger.debug("Requested GPUs: %s", cfg.gpus)
    cfg.gpus = min(torch.cuda.device_count(), cfg.gpus)
    logger.debug("GPU count set to: %s", cfg.gpus)

    # Model specific configuration
    ## ADD YOURS HERE

    print("----------------- Options ---------------")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------   End -----------------")
    return cfg
