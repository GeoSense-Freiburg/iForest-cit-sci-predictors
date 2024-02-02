"""Setup logging for the project."""
import logging


def setup_logger():
    """Setup logging for the project."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %Z",
    )


def subprocess_logger() -> logging.Logger:
    """Setup a logger for a subprocess."""
    setup_logger()
    subp_log = logging.getLogger("__main__")
    subp_log.setLevel(logging.INFO)
    return subp_log
