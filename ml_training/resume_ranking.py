"""
Resume Ranking Module
Semantic similarity via sentence-transformers/all-MiniLM-L6-v2 (~90MB).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from .text_preprocessing import TextPreprocessor, extract_skills_from_text
from .embedding_models import RANKING_MODEL_ID, embed_texts, cosine_similarity_matrix


class ResumeRanker:
    """Rank resumes by semantic similarity to a job description (all-MiniLM-L6-v2)."""

    def __init__(
        self,
        preprocessor: Optional[TextPreprocessor] = None,
        ranking_model_id: str = RANKING_MODEL_ID,
    ):
        self.preprocessor = preprocessor or TextPreprocessor(use_spacy=True)
        self.ranking_model_id = ranking_model_id
        # Legacy: categorization TF-IDF vectorizer must not be used for ranking
        self.vectorizer = None

    def set_vectorizer(self, vectorizer):
        """No-op for backward compatibility; ranking uses MiniLM embeddings."""
        self.vectorizer = vectorizer

    def _texts_for_embedding(self, job_description: str, resume_texts: List[str]) -> Tuple[str, List[str]]:
        """Use lightly cleaned raw text for semantic similarity (not stemmed TF-IDF)."""
        job = job_description.strip()
        resumes = [
            t.strip() if t else ""
            for t in resume_texts
        ]
        return job, resumes

    def compute_similarity(self, job_description: str, resume_texts: List[str]) -> np.ndarray:
        job, resumes = self._texts_for_embedding(job_description, resume_texts)
        if not job or not resumes:
            return np.zeros(len(resume_texts))

        all_texts = [job] + resumes
        embeddings = embed_texts(all_texts, model_id=self.ranking_model_id)
        job_vec = embeddings[0:1]
        resume_vecs = embeddings[1:]
        return cosine_similarity_matrix(job_vec, resume_vecs).flatten()

    def rank_resumes(
        self,
        job_description: str,
        resume_data: List[Dict],
        top_n: int = 10,
    ) -> List[Dict]:
        texts = [
            r.get("cleaned_text") or r.get("text", r.get("resume_text", ""))
            for r in resume_data
        ]
        similarities = self.compute_similarity(job_description, texts)

        results = []
        for i, resume in enumerate(resume_data):
            result = resume.copy()
            result["similarity_score"] = float(similarities[i])
            results.append(result)

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_n]

    def rank_with_skills(
        self,
        job_description: str,
        resume_data: List[Dict],
        skill_weight: float = 0.5,
        similarity_weight: float = 0.5,
        top_n: int = 10,
    ) -> List[Dict]:
        job_skills = set(extract_skills_from_text(job_description))
        results = []

        for resume in resume_data:
            raw_text = resume.get("text", resume.get("resume_text", ""))
            cleaned_text = resume.get(
                "cleaned_text",
                self.preprocessor.preprocess(raw_text) if raw_text else "",
            )

            embed_text = raw_text.strip() or cleaned_text
            sim_score = float(
                self.compute_similarity(job_description, [embed_text])[0]
            )

            resume_skills = set(extract_skills_from_text(raw_text or embed_text))
            skill_match = (
                len(job_skills & resume_skills) / len(job_skills) if job_skills else 0.0
            )

            combined_score = (
                similarity_weight * sim_score + skill_weight * skill_match
            )

            result = resume.copy()
            result["cleaned_text"] = cleaned_text
            result["similarity_score"] = sim_score
            result["skill_match_score"] = float(skill_match)
            result["combined_score"] = float(combined_score)
            result["matched_skills"] = list(job_skills & resume_skills)
            result["missing_skills"] = list(job_skills - resume_skills)
            result["job_skills"] = list(job_skills)
            result["resume_skills"] = list(resume_skills)
            results.append(result)

        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results[:top_n]


class ApprovalWorkflow:
    """Handles approval/rejection decisions based on ranking scores."""

    def __init__(self, shortlist_threshold: float = 0.6, review_threshold: float = 0.4):
        self.shortlist_threshold = shortlist_threshold
        self.review_threshold = review_threshold

    def make_decision(self, score: float) -> str:
        if score >= self.shortlist_threshold:
            return "Shortlist"
        if score >= self.review_threshold:
            return "Further Review"
        return "Reject"

    def apply_to_rankings(
        self,
        ranked_resumes: List[Dict],
        score_field: str = "combined_score",
    ) -> List[Dict]:
        for resume in ranked_resumes:
            resume["decision"] = self.make_decision(resume.get(score_field, 0.0))
        return ranked_resumes

    def generate_shortlist_report(
        self,
        ranked_resumes: List[Dict],
        output_path: str = "data/shortlist_report.csv",
        top_n: int = 10,
    ) -> str:
        top_candidates = ranked_resumes[:top_n]
        report_data = []
        for candidate in top_candidates:
            report_data.append({
                "candidate_name": candidate.get("filename", "Unknown"),
                "similarity_score": round(candidate.get("similarity_score", 0.0), 4),
                "skill_match_score": round(candidate.get("skill_match_score", 0.0), 4),
                "combined_score": round(candidate.get("combined_score", 0.0), 4),
                "decision": candidate.get("decision", "Unknown"),
                "matched_skills": ", ".join(candidate.get("matched_skills", [])),
                "missing_skills": ", ".join(candidate.get("missing_skills", [])),
                "all_resume_skills": ", ".join(candidate.get("resume_skills", [])),
            })

        df = pd.DataFrame(report_data)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Shortlist report saved to {output_path}")
        return output_path


def rank_resumes(
    job_desc: str,
    resumes: List[str],
    vectorizer=None,
    top_n: int = 10,
) -> List[Tuple[str, float]]:
    ranker = ResumeRanker()
    resume_data = [{"text": text} for text in resumes]
    ranked = ranker.rank_resumes(job_desc, resume_data, top_n=top_n)
    return [(r.get("text", ""), r["similarity_score"]) for r in ranked]


if __name__ == "__main__":
    job_description = """
    Software Engineer with Python, machine learning, AWS, Docker, agile.
    """
    sample_resumes = [
        "Data scientist: machine learning, Python, TensorFlow, PyTorch",
        "Software engineer: Java, Spring Boot, AWS, Docker, microservices",
        "Product manager: agile, user research, roadmaps",
    ]

    ranker = ResumeRanker()
    resume_data = [{"text": r, "filename": f"resume_{i}.txt"} for i, r in enumerate(sample_resumes)]
    ranked = ranker.rank_with_skills(job_description, resume_data, top_n=3)

    for i, r in enumerate(ranked, 1):
        print(f"{i}. {r['filename']} — similarity={r['similarity_score']:.3f}")
