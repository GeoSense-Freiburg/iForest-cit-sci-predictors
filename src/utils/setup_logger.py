"""Setup logging for the project."""
import logging


def setup_logger():
    """Setup logging for the project."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %Z",
    )
