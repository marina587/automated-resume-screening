"""
Resume-specific structured feature extraction.
Complements TF-IDF with years of experience, education, skill metrics, and recency.
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .text_preprocessing import extract_skills_from_text, normalize_skill_aliases

_CURRENT_YEAR = datetime.now().year


EDUCATION_LEVELS = {
    "phd": 5,
    "doctorate": 5,
    "doctoral": 5,
    "master": 4,
    "mba": 4,
    "ms": 4,
    "m.s": 4,
    "m.sc": 4,
    "bachelor": 3,
    "bs": 3,
    "b.s": 3,
    "b.sc": 3,
    "ba": 3,
    "b.a": 3,
    "associate": 2,
    "diploma": 2,
    "certificate": 1,
    "certification": 1,
    "high school": 1,
}

STRUCTURED_FEATURE_COLUMNS = [
    "years_experience",
    "education_level",
    "skill_count",
    "skill_density",
    "recency_score",
    "experience_section_ratio",
    "skills_section_ratio",
    "job_title_engineer",
    "job_title_data",
    "job_title_frontend",
    "job_title_backend",
    "job_title_devops",
    "job_title_manager",
    "job_title_designer",
]


def extract_years_of_experience(text: str) -> float:
    """Estimate total years of experience from phrases like '5+ years' or date ranges."""
    if not text:
        return 0.0

    text_lower = text.lower()
    years = []

    for match in re.finditer(
        r"(\d{1,2})\s*\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp)",
        text_lower,
    ):
        years.append(float(match.group(1)))

    for match in re.finditer(
        r"(?:over|more than|at least)\s+(\d{1,2})\s+(?:years?|yrs?)",
        text_lower,
    ):
        years.append(float(match.group(1)))

    current_year = _CURRENT_YEAR
    for match in re.finditer(
        r"\b((?:19|20)\d{2})\s*[-–—to]+\s*((?:19|20)\d{2}|present|current|now)\b",
        text_lower,
    ):
        start_year = int(match.group(1))
        end_token = match.group(2)
        if end_token in ("present", "current", "now"):
            end_year = current_year
        else:
            end_year = int(end_token)
        if end_year >= start_year:
            years.append(float(end_year - start_year))

    return float(max(years)) if years else 0.0


def extract_education_level(text: str) -> int:
    """Map highest detected education credential to an ordinal level."""
    if not text:
        return 0

    text_lower = text.lower()
    level = 0
    for keyword, value in EDUCATION_LEVELS.items():
        if re.search(rf"\b{re.escape(keyword)}\b", text_lower):
            level = max(level, value)
    return level


def compute_skill_metrics(
    text: str, word_count: Optional[int] = None
) -> Dict[str, float]:
    """Skill count and density (skills per 100 words)."""
    skills = extract_skills_from_text(text)
    words = word_count if word_count is not None else max(len(text.split()), 1)
    count = len(skills)
    density = (count / words) * 100.0
    return {"skill_count": float(count), "skill_density": float(density)}


def compute_recency_score(text: str, current_year: Optional[int] = None) -> float:
    """
    Weight recent experience higher based on latest year mentioned in the document.
    Returns 0-1 score (1 = mentions current/recent year).
    """
    if not text:
        return 0.0

    current_year = current_year or _CURRENT_YEAR
    years = [int(y) for y in re.findall(r"\b(20\d{2}|19\d{2})\b", text)]
    if not years:
        return 0.0

    latest = max(years)
    gap = max(0, current_year - latest)
    return float(max(0.0, 1.0 - (gap / 15.0)))


JOB_TITLE_PATTERNS = {
    "engineer": ["engineer", "developer", "architect", "programmer"],
    "data": ["data scientist", "data analyst", "machine learning"],
    "frontend": ["frontend", "ui", "ux", "front-end"],
    "backend": ["backend", "back-end", "infrastructure"],
    "devops": ["devops", "dev-ops", "sre", "infrastructure"],
    "manager": ["manager", "lead", "director", "head of"],
    "designer": ["designer", "ux", "ui", "creative"],
}


def extract_job_title_signals(text: str) -> Dict[str, float]:
    """
    Extract job role signals as binary/count features.
    Returns dict with score for each major job category.
    """
    if not text:
        return {k: 0.0 for k in JOB_TITLE_PATTERNS.keys()}

    text_lower = text.lower()
    signals = {}

    for category, keywords in JOB_TITLE_PATTERNS.items():
        count = sum(
            1
            for keyword in keywords
            if re.search(rf"\b{re.escape(keyword)}\b", text_lower)
        )
        signals[category] = float(min(count, 1))  # Binary: 0 or 1

    return signals


def extract_structured_features(
    text: str,
    sections: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """
    Extract numeric resume features from raw or lightly cleaned text,
    including job title signals to improve category prediction.
    """
    sections = sections or {}
    word_count = max(len(text.split()), 1)

    skill_metrics = compute_skill_metrics(text, word_count)
    total_section_len = sum(len(s.split()) for s in sections.values()) or word_count
    exp_len = len(sections.get("experience", "").split())
    skills_len = len(sections.get("skills", "").split())

    features = {
        "years_experience": extract_years_of_experience(text),
        "education_level": float(extract_education_level(text)),
        "skill_count": skill_metrics["skill_count"],
        "skill_density": skill_metrics["skill_density"],
        "recency_score": compute_recency_score(text),
        "experience_section_ratio": float(exp_len / total_section_len),
        "skills_section_ratio": float(skills_len / total_section_len),
    }

    # Add job title signals
    job_signals = extract_job_title_signals(text)
    for key, value in job_signals.items():
        features[f"job_title_{key}"] = value

    return features


def structured_features_dataframe(texts: List[str]) -> pd.DataFrame:
    """Build a DataFrame of structured features for a list of resume texts."""
    rows = [extract_structured_features(t) for t in texts]
    return pd.DataFrame(rows, columns=STRUCTURED_FEATURE_COLUMNS)


class ResumeFeatureExtractor:
    """Fits a scaler on structured features and transforms texts to numeric matrices."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, texts: List[str]) -> "ResumeFeatureExtractor":
        features = structured_features_dataframe(texts)
        self.scaler.fit(features.values)
        self.is_fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("ResumeFeatureExtractor not fitted. Call fit() first.")
        features = structured_features_dataframe(texts)
        return self.scaler.transform(features.values)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)
