import logging


def setup_logging():
    logging.basicConfig(level=logging.DEBUG)
    return logging.getLogger(__name__)
