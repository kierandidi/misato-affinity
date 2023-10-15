# Place all your constants here
import logging
import pathlib

import psutil
from dotenv import load_dotenv

# Load environment variables from .env file (secret)
load_dotenv(override=True)

# Note: constants should be UPPER_CASE
# ---------------- PATH CONSTANTS ----------------
SRC_PATH = pathlib.Path(__file__).parent
PROJECT_PATH = SRC_PATH.parent
HYDRA_CONFIG_PATH = (PROJECT_PATH / "configs").as_posix()
HYDRA_CONFIG_NAME = "default"
HYDRA_VERSION_BASE = "1.3"
ATOM_TYPE_MAP = pathlib.Path("data/atom_classes.pickle")
# ---------------- LOGGING CONSTANTS ----------------
DEFAULT_LOG_FORMATTER = logging.Formatter(
    "%(asctime)s %(levelname)s:\n\t%(message)s [in %(funcName)s at %(filename)s:%(lineno)d]"
)
DEFAULT_LOG_FILE = PROJECT_PATH / ".logs" / "default_log.log"
DEFAULT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
DEFAULT_LOG_LEVEL = logging.INFO  # verbose logging per default

# --------------- PROJECT CONSTANTS ----------------
CORE_COUNT = psutil.cpu_count(logical=False)
