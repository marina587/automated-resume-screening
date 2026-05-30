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
        model_path = self.model_dir / "logistic_model.pkl"
        encoder_path = self.model_dir / "label_encoder.pkl"

        if not model_path.exists() or not encoder_path.exists():
            return False

        self.classifier = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)

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
            emb = embed_texts([cleaned_text], model_id=self.categorization_model_id)
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

        If the maximum predicted probability is below `confidence_threshold`,
        the category is set to "Unknown" to avoid forcing an unrelated resume
        into one of the known categories.

        Args:
            text: Raw resume text to classify.
            temperature: Temperature scaling factor (default 0.85).
                         T < 1 sharpens probabilities (higher peak confidence).
                         T = 1 is standard softmax.
                         T > 1 flattens probabilities.
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
            # Try temperature-scaled softmax first for sharper probabilities
            if hasattr(self.classifier, "predict_log_proba"):
                log_probs = self.classifier.predict_log_proba(X)[0]
                # Apply temperature scaling: divide logits by T
                scaled_log_probs = log_probs / temperature
                # Convert back to probabilities via softmax
                probs = np.exp(scaled_log_probs - np.max(scaled_log_probs))
                probs = probs / np.sum(probs)
                confidence = float(np.max(probs))
            else:
                probabilities = self.classifier.predict_proba(X)[0]
                confidence = float(np.max(probabilities))
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

        # Apply confidence threshold — return "Unknown" if confidence is too low
        if confidence < confidence_threshold:
            category = "Unknown"
        else:
            category = self.label_encoder.inverse_transform([prediction])[0]

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
