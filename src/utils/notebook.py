import logging
import os
import pathlib

import dotenv
import hydra
from omegaconf import OmegaConf

from src.constants import CONFIG_PATH
from src.utils.logutils import get_logger

dotenv.load_dotenv(override=True)
try:
    from icecream import ic
except:
    pass

JUPYTER_LOG_FORMATTER = logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s [l:%(lineno)d]", datefmt="%Y-%m-%d %H:%M:%S"
)
log = get_logger("jupyter", log_to_file=False, formatter=JUPYTER_LOG_FORMATTER)

# NOTE: Registering the eval resolver is needed to allow using the ${eval:3*5} interpolation sytax
#  in general this can have security implications if a model is run on a server accessible via the
#  internet, so in the general case we recommend using explicit configs rather than eval statements.
OmegaConf.register_new_resolver("eval", eval)

# Convenient aliases:
instantiate = hydra.utils.instantiate


def init_hydra_singleton(
    path: os.PathLike = CONFIG_PATH, reload: bool = False, version_base="1.2"
) -> None:
    # See: https://stackoverflow.com/questions/60674012/how-to-get-a-hydra-config-without-using-hydra-main
    if reload:
        clear_hydra_singleton()
    try:
        path = pathlib.Path(path)
        # Note: hydra needs to be initialised with a relative path. Since the hydra
        #  singleton is first created here, it needs to be created relative to this
        #  file. The `rel_path` below takes care of that.
        rel_path = os.path.relpath(path, start=pathlib.Path(__file__).parent)
        hydra.initialize(rel_path, version_base=version_base)
        log.info("Hydra initialised at %s." % path.absolute())
    except ValueError:
        log.info("Hydra already initialised.")


def clear_hydra_singleton() -> None:
    if hydra.core.global_hydra.GlobalHydra not in hydra.core.singleton.Singleton._instances:
        return
    hydra_singleton = hydra.core.singleton.Singleton._instances[hydra.core.global_hydra.GlobalHydra]
    hydra_singleton.clear()
    log.info("Hydra singleton cleared and ready to re-initialise.")
