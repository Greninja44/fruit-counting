"""Structured logging setup shared by all scripts."""
from __future__ import annotations

import logging
import sys

_FMT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger, configuring the root handler once."""
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FMT, "%H:%M:%S"))
        root.addHandler(handler)
        root.setLevel(level)
    return logging.getLogger(name)
