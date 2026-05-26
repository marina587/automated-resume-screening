# ML Training Package
"""
AI Resume Screening - Machine Learning Modules

This package contains all ML-related modules for the resume screening system.
"""

from .data_preparation import DataPreparator, create_sample_dataset
from .text_preprocessing import TextPreprocessor, extract_skills_from_text
from .resume_parser import ResumeParser
from .model_training import ResumeClassifier, compare_models
from .resume_ranking import ResumeRanker, ApprovalWorkflow
from .resume_features import extract_structured_features, ResumeFeatureExtractor
from .embedding_models import embed_texts, RANKING_MODEL_ID, CATEGORIZATION_MODEL_ID
from .inference import ModelBundle
from .load_hf_dataset import download_and_prepare, load_hf_resume_dataset

__version__ = '1.2.0'
__all__ = [
    'DataPreparator',
    'create_sample_dataset',
    'TextPreprocessor',
    'extract_skills_from_text',
    'ResumeParser',
    'ResumeClassifier',
    'compare_models',
    'ResumeRanker',
    'ApprovalWorkflow',
    'extract_structured_features',
    'ResumeFeatureExtractor',
    'embed_texts',
    'RANKING_MODEL_ID',
    'CATEGORIZATION_MODEL_ID',
    'ModelBundle',
    'download_and_prepare',
    'load_hf_resume_dataset',
]
