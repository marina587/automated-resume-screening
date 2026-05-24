"""
Resume Ranking Module
Implements similarity scoring and ranking of resumes against job descriptions.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity

# Use relative imports for package usage
try:
    from .text_preprocessing import TextPreprocessor, extract_skills_from_text
except ImportError:
    from text_preprocessing import TextPreprocessor, extract_skills_from_text


class ResumeRanker:
    """Handles ranking of resumes based on job description similarity."""
    
    def __init__(self, vectorizer=None, preprocessor: TextPreprocessor = None):
        """
        Initialize the ranker.
        
        Args:
            vectorizer: Pre-fitted TfidfVectorizer (from trained model)
            preprocessor: TextPreprocessor instance for cleaning texts
        """
        self.vectorizer = vectorizer
        self.preprocessor = preprocessor or TextPreprocessor()
    
    def set_vectorizer(self, vectorizer):
        """Set the TF-IDF vectorizer for transforming texts."""
        self.vectorizer = vectorizer
    
    def compute_similarity(self, job_description: str, resume_texts: List[str]) -> np.ndarray:
        """
        Compute cosine similarity between job description and resumes.
        
        Args:
            job_description: Job description text
            resume_texts: List of preprocessed resume texts
            
        Returns:
            Array of similarity scores
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not set. Call set_vectorizer() first.")
        
        # Transform job description
        job_vector = self.vectorizer.transform([job_description])
        
        # Transform resumes
        resume_vectors = self.vectorizer.transform(resume_texts)
        
        # Compute cosine similarity
        similarities = cosine_similarity(job_vector, resume_vectors).flatten()
        
        return similarities
    
    def rank_resumes(self, job_description: str, resume_data: List[Dict], 
                     top_n: int = 10) -> List[Dict]:
        """
        Rank resumes based on similarity to job description.
        
        Args:
            job_description: Job description text
            resume_data: List of dicts with 'text' (preprocessed) and optional metadata
            top_n: Number of top resumes to return
            
        Returns:
            List of ranked resumes with scores
        """
        # Extract texts
        texts = [r.get('cleaned_text', r.get('text', '')) for r in resume_data]
        
        # Compute similarities
        similarities = self.compute_similarity(job_description, texts)
        
        # Create results with scores
        results = []
        for i, resume in enumerate(resume_data):
            result = resume.copy()
            result['similarity_score'] = float(similarities[i])
            results.append(result)
        
        # Sort by score (descending)
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Return top N
        return results[:top_n]
    
    def rank_with_skills(self, job_description: str, resume_data: List[Dict],
                         skill_weight: float = 0.5, 
                         similarity_weight: float = 0.5,
                         top_n: int = 10) -> List[Dict]:
        """
        Rank resumes using both similarity and skill matching.
        
        Args:
            job_description: Job description text
            resume_data: List of dicts with 'text' (raw) and optional metadata
            skill_weight: Weight for skill matching (0-1)
            similarity_weight: Weight for cosine similarity (0-1)
            top_n: Number of top resumes to return
            
        Returns:
            List of ranked resumes with combined scores
        """
        # Extract job skills
        job_skills = set(extract_skills_from_text(job_description))
        
        results = []
        
        for resume in resume_data:
            raw_text = resume.get('text', resume.get('resume_text', ''))
            cleaned_text = resume.get('cleaned_text', self.preprocessor.preprocess(raw_text))
            
            # Compute similarity score
            if self.vectorizer is not None:
                sim_score = self.compute_similarity(job_description, [cleaned_text])[0]
            else:
                sim_score = 0.0
            
            # Extract resume skills and compute skill match
            resume_skills = set(extract_skills_from_text(raw_text))
            if job_skills:
                skill_match = len(job_skills & resume_skills) / len(job_skills)
            else:
                skill_match = 0.0
            
            # Combined score
            combined_score = (similarity_weight * sim_score + 
                            skill_weight * skill_match)
            
            result = resume.copy()
            result['similarity_score'] = float(sim_score)
            result['skill_match_score'] = float(skill_match)
            result['combined_score'] = float(combined_score)
            result['matched_skills'] = list(job_skills & resume_skills)
            result['missing_skills'] = list(job_skills - resume_skills)
            result['job_skills'] = list(job_skills)
            result['resume_skills'] = list(resume_skills)
            
            results.append(result)
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return results[:top_n]


class ApprovalWorkflow:
    """Handles approval/rejection decisions based on ranking scores."""
    
    def __init__(self, shortlist_threshold: float = 0.6, 
                 review_threshold: float = 0.4):
        """
        Initialize the approval workflow.
        
        Args:
            shortlist_threshold: Score above which to shortlist
            review_threshold: Score above which to mark for further review
        """
        self.shortlist_threshold = shortlist_threshold
        self.review_threshold = review_threshold
    
    def make_decision(self, score: float) -> str:
        """
        Make approval decision based on score.
        
        Args:
            score: Similarity or combined score
            
        Returns:
            Decision string: 'Shortlist', 'Further Review', or 'Reject'
        """
        if score >= self.shortlist_threshold:
            return 'Shortlist'
        elif score >= self.review_threshold:
            return 'Further Review'
        else:
            return 'Reject'
    
    def apply_to_rankings(self, ranked_resumes: List[Dict], 
                          score_field: str = 'combined_score') -> List[Dict]:
        """
        Apply approval decisions to ranked resumes.
        
        Args:
            ranked_resumes: List of ranked resume dicts
            score_field: Field name containing the score to use
            
        Returns:
            Updated list with decision field added
        """
        for resume in ranked_resumes:
            score = resume.get(score_field, 0.0)
            resume['decision'] = self.make_decision(score)
        
        return ranked_resumes
    
    def generate_shortlist_report(self, ranked_resumes: List[Dict],
                                   output_path: str = 'data/shortlist_report.csv',
                                   top_n: int = 10) -> str:
        """
        Generate a shortlist report CSV.
        
        Args:
            ranked_resumes: List of ranked resume dicts
            output_path: Path to save the report
            top_n: Number of top candidates to include
            
        Returns:
            Path to saved report
        """
        import pandas as pd
        from pathlib import Path
        
        # Take top N
        top_candidates = ranked_resumes[:top_n]
        
        # Prepare report data
        report_data = []
        for candidate in top_candidates:
            report_data.append({
                'candidate_name': candidate.get('filename', 'Unknown'),
                'similarity_score': round(candidate.get('similarity_score', 0.0), 4),
                'skill_match_score': round(candidate.get('skill_match_score', 0.0), 4),
                'combined_score': round(candidate.get('combined_score', 0.0), 4),
                'decision': candidate.get('decision', 'Unknown'),
                'matched_skills': ', '.join(candidate.get('matched_skills', [])),
                'missing_skills': ', '.join(candidate.get('missing_skills', [])),
                'all_resume_skills': ', '.join(candidate.get('resume_skills', []))
            })
        
        df = pd.DataFrame(report_data)
        
        # Save to CSV
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"Shortlist report saved to {output_path}")
        return output_path


def rank_resumes(job_desc: str, resumes: List[str], 
                 vectorizer=None, top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Convenience function to rank resumes against a job description.
    
    Args:
        job_desc: Job description text
        resumes: List of resume texts
        vectorizer: Pre-fitted TfidfVectorizer
        top_n: Number of top resumes to return
        
    Returns:
        List of (resume_text, score) tuples
    """
    ranker = ResumeRanker(vectorizer=vectorizer)
    
    # Prepare resume data
    resume_data = [{'text': text} for text in resumes]
    
    # Rank
    ranked = ranker.rank_resumes(job_desc, resume_data, top_n=top_n)
    
    # Convert to simple format
    return [(r['text'], r['similarity_score']) for r in ranked]


if __name__ == "__main__":
    # Example usage
    from ml_training.model_training import ResumeClassifier
    from ml_training.text_preprocessing import TextPreprocessor
    
    # Sample data
    job_description = """
    We are looking for a Software Engineer with experience in Python, 
    machine learning, cloud technologies (AWS), and containerization (Docker).
    The ideal candidate should have strong problem-solving skills and 
    experience with agile development methodologies.
    """
    
    sample_resumes = [
        "Experienced data scientist with expertise in machine learning, Python, TensorFlow",
        "Software engineer specializing in backend development with Java, Spring Boot, AWS, Docker",
        "Product manager with 5+ years experience in agile methodologies, user research",
        "DevOps engineer skilled in CI/CD pipelines, Kubernetes, Terraform, Python",
        "Full stack developer proficient in React, Node.js, MongoDB, GraphQL"
    ]
    
    # Initialize components
    preprocessor = TextPreprocessor()
    ranker = ResumeRanker(preprocessor=preprocessor)
    approval = ApprovalWorkflow(shortlist_threshold=0.5, review_threshold=0.3)
    
    # Clean resumes
    cleaned_resumes = [preprocessor.preprocess(r) for r in sample_resumes]
    
    # For demonstration, create a simple vectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    vectorizer.fit(cleaned_resumes)
    ranker.set_vectorizer(vectorizer)
    
    # Prepare resume data
    resume_data = [
        {'text': r, 'cleaned_text': c, 'filename': f'resume_{i}.txt'}
        for i, (r, c) in enumerate(zip(sample_resumes, cleaned_resumes))
    ]
    
    # Rank with skills
    ranked = ranker.rank_with_skills(job_description, resume_data, top_n=5)
    
    # Apply approval decisions
    ranked = approval.apply_to_rankings(ranked, score_field='combined_score')
    
    # Print results
    print("Ranked Resumes:")
    print("=" * 80)
    for i, resume in enumerate(ranked, 1):
        print(f"\n{i}. {resume['filename']}")
        print(f"   Similarity Score: {resume['similarity_score']:.3f}")
        print(f"   Skill Match: {resume['skill_match_score']:.3f}")
        print(f"   Combined Score: {resume['combined_score']:.3f}")
        print(f"   Decision: {resume['decision']}")
        print(f"   Matched Skills: {', '.join(resume['matched_skills'])}")
    
    # Generate report
    approval.generate_shortlist_report(ranked, top_n=5)
