"""
Load and adapt AzharAli05/Resume-Screening-Dataset from Hugging Face.

Dataset: https://huggingface.co/datasets/AzharAli05/Resume-Screening-Dataset
~10.2k rows, columns: Role, Resume, Decision, Reason_for_decision, Job_Description
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

HF_DATASET_ID = "AzharAli05/Resume-Screening-Dataset"
DEFAULT_OUTPUT = "data/hf_resume_screening.csv"

# Map HF column names (handle casing variants from parquet/csv export)
COLUMN_ALIASES = {
    "role": "category",
    "resume": "resume_text",
    "decision": "screening_decision",
    "reason_for_decision": "reason_for_decision",
    "job_description": "job_description",
}


RESUME_PREAMBLE_PATTERNS = [
    r"^Here'?s a (?:professional |sample )?resume for[^:]+:\s*",
    r"^Here is a (?:professional |sample )?resume for[^:]+:\s*",
    r"^Here'?s a sample professional resume for[^:]+:\s*",
]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "_")
        if key in COLUMN_ALIASES:
            rename[col] = COLUMN_ALIASES[key]
    return df.rename(columns=rename)


ROLE_ACRONYM_FIXES = {
    "Ux": "UX",
    "Ui": "UI",
    "Ai": "AI",
    "Ml": "ML",
    "Hr": "HR",
    "Qa": "QA",
    "Devops": "DevOps",
}


def normalize_role_name(role: str) -> str:
    """Normalize role name with proper casing and acronym handling."""
    role = re.sub(r"\s+", " ", role.strip())
    # Split on whitespace and hyphens, title-case each part, then rejoin
    parts = re.split(r"(\s+|-)", role)
    normalized_parts = []
    for part in parts:
        if part in (" ", "-"):
            normalized_parts.append(part)
        else:
            titled = part.title()
            normalized_parts.append(ROLE_ACRONYM_FIXES.get(titled, titled))
    return "".join(normalized_parts)


def clean_resume_text(text: str) -> str:
    """Strip synthetic preamble and normalize whitespace."""
    if not text or not isinstance(text, str):
        return ""

    text = text.strip()
    for pattern in RESUME_PREAMBLE_PATTERNS:
        text = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)

    # Unwrap markdown links: [name](mailto:email) -> name
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_hf_resume_dataset(
    split: str = "train",
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Download the dataset from Hugging Face Hub and return a pandas DataFrame.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Install the datasets package: pip install datasets"
        ) from exc

    print(f"Loading {HF_DATASET_ID} (split={split})...")
    dataset = load_dataset(HF_DATASET_ID, split=split)
    df = dataset.to_pandas()

    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows).copy()

    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    return df


def adapt_hf_resume_dataset(
    df: Optional[pd.DataFrame] = None,
    output_path: Union[str, Path] = DEFAULT_OUTPUT,
    holdout_fraction: float = 0.2,
    random_state: int = 42,
    max_rows: Optional[int] = None,
    min_category_count: int = 5,
) -> str:
    """
    Adapt HF dataset for role classification training and resume screening.

    - category: job Role (45 classes)
    - resume_text: cleaned Resume body
    - screening_decision: select | reject
    - job_description, reason_for_decision: kept for ranking / screening eval
    - data_source: train | holdout (stratified by category)
    """
    if df is None:
        df = load_hf_resume_dataset(max_rows=max_rows)

    df = _normalize_columns(df)

    required = {"resume_text", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset missing columns {missing}. Found: {list(df.columns)}"
        )

    df["resume_text"] = df["resume_text"].map(clean_resume_text)
    df["category"] = df["category"].astype(str).map(normalize_role_name)

    if "screening_decision" in df.columns:
        df["screening_decision"] = (
            df["screening_decision"].astype(str).str.strip().str.lower()
        )

    df = df[df["resume_text"].str.len() >= 50]
    df = df[df["category"].str.len() > 0]
    df = df.drop_duplicates(subset=["resume_text"])

    # Drop very rare roles so stratified splits stay stable
    counts = df["category"].value_counts()
    valid_cats = counts[counts >= min_category_count].index
    dropped = len(counts) - len(valid_cats)
    if dropped:
        print(f"Dropping {dropped} role(s) with fewer than {min_category_count} samples")
    df = df[df["category"].isin(valid_cats)].copy()

    rng = np.random.default_rng(random_state)
    n_holdout = max(1, int(len(df) * holdout_fraction))
    holdout_idx = set()

    for category in df["category"].unique():
        cat_idx = df.index[df["category"] == category].tolist()
        n_cat_holdout = max(1, int(len(cat_idx) * holdout_fraction))
        chosen = rng.choice(cat_idx, size=min(n_cat_holdout, len(cat_idx)), replace=False)
        holdout_idx.update(chosen)

    if len(holdout_idx) < n_holdout:
        remaining = list(set(df.index) - holdout_idx)
        extra = rng.choice(
            remaining,
            size=min(n_holdout - len(holdout_idx), len(remaining)),
            replace=False,
        )
        holdout_idx.update(extra)

    df["data_source"] = ["holdout" if i in holdout_idx else "train" for i in df.index]

    keep_cols = [
        "resume_text",
        "category",
        "data_source",
        "screening_decision",
        "reason_for_decision",
        "job_description",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"Saved adapted dataset to {output_path}")
    print(f"  Rows: {len(out)}")
    print(f"  Roles: {out['category'].nunique()}")
    print(f"  Holdout: {(out['data_source'] == 'holdout').sum()}")
    print(f"  Category distribution (top 10):\n{out['category'].value_counts().head(10)}")
    if "screening_decision" in out.columns:
        print(f"  Screening decisions:\n{out['screening_decision'].value_counts()}")

    return str(output_path)


def download_and_prepare(
    output_path: Union[str, Path] = DEFAULT_OUTPUT,
    max_rows: Optional[int] = None,
    **kwargs,
) -> str:
    """Convenience: load from Hub, adapt, and save CSV."""
    return adapt_hf_resume_dataset(
        output_path=output_path,
        max_rows=max_rows,
        **kwargs,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download HF Resume Screening Dataset")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--holdout", type=float, default=0.2)
    args = parser.parse_args()

    download_and_prepare(
        output_path=args.output,
        max_rows=args.max_rows,
        holdout_fraction=args.holdout,
    )
