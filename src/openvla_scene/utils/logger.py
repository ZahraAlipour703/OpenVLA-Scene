import logging


def get_logger(name: str):

    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s"
    )

    if not logger.handlers:

        console = logging.StreamHandler()

        console.setFormatter(formatter)

        logger.addHandler(console)

    return logger