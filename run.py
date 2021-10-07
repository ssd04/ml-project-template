from src.cli import manage_args
from src.config.logging import setup_logger


if __name__ == "__main__":
    setup_logger()
    args = manage_args()

    # trigger stuff
