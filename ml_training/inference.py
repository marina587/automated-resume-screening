"""
Unified model loading for API and frontend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

from .embedding_models import (
    CATEGORIZATION_MODEL_ID,
    embed_texts,
    load_model_config,
)
from .resume_ranking import ResumeRanker
from .text_preprocessing import TextPreprocessor

UNKNOWN_CATEGORY = "Unknown"


class ModelBundle:
    """Loaded categorization + ranking models."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.config = load_model_config(self.model_dir)
        self.categorization_backend = self.config["categorization"]["backend"]
        self.categorization_model_id = (
            self.config["categorization"].get("model") or CATEGORIZATION_MODEL_ID
        )

        self.classifier = None
        self.vectorizer = None
        self.label_encoder = None
        self.feature_extractor = None
        self.use_structured_features = False

        # Read preprocessing config to match training-time preprocessing
        preprocessing_config = self.config.get("preprocessing", {})
        use_spacy = preprocessing_config.get("use_spacy", False)

        self.preprocessor = TextPreprocessor(
            use_spacy=use_spacy,
            preserve_technical_terms=True,
            section_aware=True,
        )
        self.ranker = ResumeRanker(preprocessor=self.preprocessor)
        self.is_loaded = False
        self.logger = logging.getLogger(__name__)

    def load(self) -> bool:
        encoder_path = self.model_dir / "label_encoder.pkl"
        if not encoder_path.exists():
            return False

        model_type = self.config.get("categorization", {}).get("model_type")
        if model_type:
            model_path = self.model_dir / f"{model_type}_model.pkl"
        else:
            # Fallback for legacy artifacts: detect any saved classifier model.
            model_path = None
            for candidate in [
                "logistic",
                "random_forest",
                "gradient_boosting",
                "svm",
                "knn",
            ]:
                candidate_path = self.model_dir / f"{candidate}_model.pkl"
                if candidate_path.exists():
                    model_path = candidate_path
                    break
            if model_path is None:
                return False

        if not model_path.exists():
            return False

        self.classifier = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        self.loaded_model_path = str(model_path)
        self.loaded_model_type = model_type or model_path.name.replace("_model.pkl", "")

        vectorizer_path = self.model_dir / "vectorizer.pkl"
        if vectorizer_path.exists() and self.categorization_backend == "tfidf":
            self.vectorizer = joblib.load(vectorizer_path)

        feature_extractor_path = self.model_dir / "feature_extractor.pkl"
        if feature_extractor_path.exists():
            self.feature_extractor = joblib.load(feature_extractor_path)
            self.use_structured_features = True

        self.is_loaded = True
        return True

    def _build_features(self, cleaned_text: str, raw_text: str) -> np.ndarray:
        if self.categorization_backend == "minilm":
            # Truncate to MiniLM's 512-token limit (~3000 chars for English)
            truncated = cleaned_text[:3000] if cleaned_text else ""
            emb = embed_texts([truncated], model_id=self.categorization_model_id)
            parts = [emb]
            if self.use_structured_features and self.feature_extractor is not None:
                struct = self.feature_extractor.transform([raw_text])
                parts.append(struct)
            return np.hstack(parts)

        if self.vectorizer is None:
            raise ValueError("TF-IDF vectorizer not loaded")

        X_text = self.vectorizer.transform([cleaned_text])
        if self.use_structured_features and self.feature_extractor is not None:
            X_struct = csr_matrix(self.feature_extractor.transform([raw_text]))
            return hstack([X_text, X_struct])
        return X_text

    def predict_category(
        self,
        text: str,
        temperature: float = 0.85,
        confidence_threshold: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Predict category with optional temperature scaling to sharpen probabilities.

        Temperature scaling is only applied when the classifier is NOT calibrated
        (i.e., not wrapped in CalibratedClassifierCV). Calibrated models already
        produce well-calibrated probabilities, and applying temperature on top
        would distort the calibration.

        If the maximum predicted probability is below `confidence_threshold`,
        the category is set to "Unknown" to avoid forcing an unrelated resume
        into one of the known categories.

        Args:
            text: Raw resume text to classify.
            temperature: Temperature scaling factor (default 0.85).
                         T < 1 sharpens probabilities (higher peak confidence).
                         T = 1 is standard softmax.
                         T > 1 flattens probabilities.
                         Only applied when the model is NOT calibrated.
            confidence_threshold: Minimum confidence to accept a prediction.
                                  Below this threshold, returns "Unknown"
                                  (default 0.3).

        Raises:
            ValueError: If temperature <= 0 (would cause division by zero or NaN).
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded")

        if temperature <= 0:
            raise ValueError(
                f"Temperature must be > 0 (got {temperature}). "
                "Values <= 0 cause division by zero and NaN scores."
            )

        cleaned_text = self.preprocessor.preprocess(text)
        X = self._build_features(cleaned_text, text)
        prediction = self.classifier.predict(X)[0]

        confidence = 0.0
        try:
            probabilities = self.classifier.predict_proba(X)[0]
            confidence = float(np.max(probabilities))

            # Apply temperature scaling ONLY if the model is NOT calibrated.
            # CalibratedClassifierCV already produces well-calibrated probabilities;
            # applying temperature on top would distort them.
            # We detect calibration by checking if the classifier is a CalibratedClassifierCV
            # instance (which has a 'calibrated_classifiers_' attribute after fitting).
            is_calibrated = hasattr(self.classifier, "calibrated_classifiers_")
            if not is_calibrated and temperature != 1.0:
                # Temperature scaling: sharpen or flatten probabilities
                # Convert probabilities back to logits, scale, then softmax
                eps = 1e-12
                logits = np.log(np.maximum(probabilities, eps))
                scaled_logits = logits / temperature
                scaled = np.exp(scaled_logits - np.max(scaled_logits))
                scaled = scaled / np.sum(scaled)
                confidence = float(np.max(scaled))
        except Exception as e:
            self.logger.warning(
                "predict_proba failed for %s: %s",
                type(self.classifier).__name__,
                e,
            )
            try:
                decision = self.classifier.decision_function(X)
                decision = np.asarray(decision)
                if decision.ndim == 1:
                    confidence = float(1 / (1 + np.exp(-decision[0])))
                else:
                    raw = decision[0].astype(float)
                    exp = np.exp(raw - np.max(raw))
                    confidence = float(np.max(exp / np.sum(exp)))
            except Exception as exc:
                self.logger.debug(
                    "decision_function fallback failed for %s: %s",
                    type(self.classifier).__name__,
                    exc,
                )

        # Two-tier Unknown detection:
        # 1. If the model itself predicted "Unknown", respect that directly.
        # 2. Otherwise, fall back to confidence threshold for low-confidence predictions.
        predicted_category = self.label_encoder.inverse_transform([prediction])[0]
        if predicted_category == UNKNOWN_CATEGORY:
            category = UNKNOWN_CATEGORY
        elif confidence < confidence_threshold:
            category = UNKNOWN_CATEGORY
        else:
            category = predicted_category

        return {
            "category": category,
            "confidence": confidence,
            "cleaned_text": cleaned_text,
            "categorization_backend": self.categorization_backend,
        }

    def rank_resumes(
        self,
        job_description: str,
        resumes: List[Dict],
        top_n: int = 10,
    ) -> List[Dict]:
        if not self.is_loaded:
            raise ValueError("Models not loaded")

        for resume in resumes:
            if "cleaned_text" not in resume:
                raw = resume.get("text", resume.get("resume_text", ""))
                resume["cleaned_text"] = self.preprocessor.preprocess(raw)

        return self.ranker.rank_with_skills(job_description, resumes, top_n=top_n)
