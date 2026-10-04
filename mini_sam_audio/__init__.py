"""Mini-SAM-Audio: Model Compression Pipeline for SAM-Audio"""

__version__ = "0.1.0"

from .model import *  # noqa
from .processor import *  # noqa

__all__ = [
    "compression",
]
