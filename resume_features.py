"""Compatibility alias for older saved models.

Some model artifacts were serialized when the submodule was imported as
`resume_features` instead of `ml_training.resume_features`. Providing this
alias allows joblib/pickle to locate the class during model loading.
"""

from ml_training.resume_features import ResumeFeatureExtractor

__all__ = ["ResumeFeatureExtractor"]
