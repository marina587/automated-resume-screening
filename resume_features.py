"""Compatibility alias for older saved models.

Some model artifacts were serialized when the submodule was imported as
`resume_features` instead of `ml_training.resume_features`. Providing this
alias allows joblib/pickle to locate the class during model loading.
"""

import logging

_logger = logging.getLogger(__name__)

try:
    from ml_training.resume_features import ResumeFeatureExtractor
except ImportError as _exc:
    _logger.warning(
        "Could not import ResumeFeatureExtractor from ml_training.resume_features: %s. "
        "This may cause errors when loading older pickled models.",
        _exc,
    )
    ResumeFeatureExtractor = None  # type: ignore

__all__ = ["ResumeFeatureExtractor"]
