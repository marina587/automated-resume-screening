"""
Shared sentence-transformer models for categorization and ranking.
Ranking: sentence-transformers/all-MiniLM-L6-v2 (~90MB)
Categorization: same MiniLM embeddings + sklearn classifier, or TF-IDF fallback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

RANKING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CATEGORIZATION_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SPACY_MODEL_ID = "en_core_web_sm"

_encoder_cache: dict = {}


def get_sentence_encoder(model_id: Optional[str] = None):
    """Lazy-load a SentenceTransformer model (cached per model id)."""
    model_id = model_id or RANKING_MODEL_ID
    if model_id not in _encoder_cache:
        from sentence_transformers import SentenceTransformer
        _encoder_cache[model_id] = SentenceTransformer(model_id)
    return _encoder_cache[model_id]


def embed_texts(
    texts: List[str],
    model_id: Optional[str] = None,
    batch_size: int = 32,
    show_progress: bool = False,
) -> np.ndarray:
    """Encode texts to normalized embedding vectors."""
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    encoder = get_sentence_encoder(model_id)
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows of a (1 x d) and b (n x d); vectors assumed normalized."""
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return np.dot(a, b.T)


def save_model_config(
    model_dir: Union[str, Path],
    categorization_backend: str = "minilm",
    categorization_model: str = CATEGORIZATION_MODEL_ID,
    ranking_model: str = RANKING_MODEL_ID,
) -> Path:
    config = {
        "parsing": {"pdf": "pymupdf", "nlp": SPACY_MODEL_ID},
        "categorization": {
            "backend": categorization_backend,
            "model": categorization_model,
        },
        "ranking": {"model": ranking_model},
    }
    path = Path(model_dir) / "model_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def load_model_config(model_dir: Union[str, Path]) -> dict:
    path = Path(model_dir) / "model_config.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    # Legacy TF-IDF artifacts without config
    if (Path(model_dir) / "vectorizer.pkl").exists():
        return {
            "parsing": {"pdf": "pymupdf", "nlp": SPACY_MODEL_ID},
            "categorization": {"backend": "tfidf", "model": None},
            "ranking": {"model": RANKING_MODEL_ID},
        }
    return {
        "parsing": {"pdf": "pymupdf", "nlp": SPACY_MODEL_ID},
        "categorization": {"backend": "minilm", "model": CATEGORIZATION_MODEL_ID},
        "ranking": {"model": RANKING_MODEL_ID},
    }
