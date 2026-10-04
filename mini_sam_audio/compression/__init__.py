"""Model compression pipeline modules."""

from .model_loading import bootstrap_model_and_processor, load_student_model

__all__ = [
    "bootstrap_model_and_processor",
    "load_student_model",
]
