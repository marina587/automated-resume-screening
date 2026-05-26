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

        self.preprocessor = TextPreprocessor(
            use_spacy=True,
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

    def predict_category(self, text: str) -> Dict[str, Any]:
        if not self.is_loaded:
            raise ValueError("Models not loaded")

        cleaned_text = self.preprocessor.preprocess(text)
        X = self._build_features(cleaned_text, text)
        prediction = self.classifier.predict(X)[0]

        confidence = 0.0
        try:
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
