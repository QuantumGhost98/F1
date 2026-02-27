"""
Shared logging configuration for the pipeline package.

All pipeline modules should use:
    from pipeline.log import logger
"""

import logging

logger = logging.getLogger("pipeline")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
