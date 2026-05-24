"""
Backend API for Resume Screening System
FastAPI-based REST API for resume processing, classification, and ranking.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import os
import tempfile
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Resume Screening API",
    description="API for AI-powered resume screening and approval system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models
model_manager = None


class ModelManager:
    """Manages loading and access to ML models."""
    
    def __init__(self):
        self.classifier = None
        self.vectorizer = None
        self.label_encoder = None
        self.preprocessor = None
        self.ranker = None
        self.is_loaded = False
    
    def load_models(self, model_dir: str = "models"):
        """Load all required models from disk."""
        try:
            import joblib
            from ml_training.text_preprocessing import TextPreprocessor
            from ml_training.resume_ranking import ResumeRanker
            
            model_path = Path(model_dir) / "logistic_model.pkl"
            vectorizer_path = Path(model_dir) / "vectorizer.pkl"
            encoder_path = Path(model_dir) / "label_encoder.pkl"
            
            if not model_path.exists():
                logger.warning(f"Model not found at {model_path}. Models not loaded.")
                return False
            
            self.classifier = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.label_encoder = joblib.load(encoder_path)
            self.preprocessor = TextPreprocessor()
            self.ranker = ResumeRanker(vectorizer=self.vectorizer, preprocessor=self.preprocessor)
            
            self.is_loaded = True
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def predict_category(self, text: str) -> Dict[str, Any]:
        """Predict category for a resume text."""
        if not self.is_loaded:
            raise ValueError("Models not loaded")
        
        cleaned_text = self.preprocessor.preprocess(text)
        category, confidence = self.classifier.predict_category(cleaned_text)
        
        return {
            "category": category,
            "confidence": float(confidence),
            "cleaned_text": cleaned_text
        }
    
    def rank_resumes(self, job_description: str, resumes: List[Dict], 
                     top_n: int = 10) -> List[Dict]:
        """Rank resumes against a job description."""
        if not self.is_loaded:
            raise ValueError("Models not loaded")
        
        # Preprocess resumes
        for resume in resumes:
            if 'cleaned_text' not in resume:
                resume['cleaned_text'] = self.preprocessor.preprocess(
                    resume.get('text', resume.get('resume_text', ''))
                )
        
        ranked = self.ranker.rank_with_skills(job_description, resumes, top_n=top_n)
        return ranked


class JobDescription(BaseModel):
    """Job description input model."""
    text: str
    required_skills: Optional[List[str]] = None


class ResumeScreeningRequest(BaseModel):
    """Request model for resume screening."""
    job_description: str
    shortlist_threshold: float = 0.6
    review_threshold: float = 0.4
    top_n: int = 10


class ScreeningResult(BaseModel):
    """Result model for screening."""
    filename: str
    category: Optional[str] = None
    category_confidence: Optional[float] = None
    similarity_score: float
    skill_match_score: float
    combined_score: float
    decision: str
    matched_skills: List[str]
    missing_skills: List[str]
    resume_skills: List[str]


@app.on_event("startup")
async def startup_event():
    """Load models on application startup."""
    global model_manager
    import sys
    from pathlib import Path
    # Add workspace to path for ml_training imports
    workspace_dir = Path(__file__).parent.parent
    if str(workspace_dir) not in sys.path:
        sys.path.insert(0, str(workspace_dir))
    
    model_manager = ModelManager()
    
    # Try to load models (will fail gracefully if not present)
    model_manager.load_models()
    
    logger.info("Application started")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI Resume Screening API",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": model_manager.is_loaded if model_manager else False
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": model_manager.is_loaded if model_manager else False
    }


@app.post("/upload-resumes")
async def upload_resumes(files: List[UploadFile] = File(...)):
    """
    Upload and extract text from resume files.
    
    Returns extracted text for each uploaded file.
    """
    try:
        from ml_training.resume_parser import ResumeParser
        
        parser = ResumeParser()
        results = []
        
        for file in files:
            # Save temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name
            
            try:
                text = parser.extract_text(tmp_path)
                if text:
                    results.append({
                        "filename": file.filename,
                        "text": text,
                        "file_type": Path(file.filename).suffix[1:].upper()
                    })
            finally:
                os.unlink(tmp_path)
        
        return {
            "success": True,
            "count": len(results),
            "resumes": results
        }
        
    except Exception as e:
        logger.error(f"Error uploading resumes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/screen-resumes")
async def screen_resumes(
    job_description: str = Form(...),
    shortlist_threshold: float = Form(0.6),
    review_threshold: float = Form(0.4),
    top_n: int = Form(10),
    files: List[UploadFile] = File(...)
):
    """
    Screen uploaded resumes against a job description.
    
    This endpoint:
    1. Extracts text from uploaded resume files
    2. Ranks resumes based on similarity to job description
    3. Applies approval workflow
    4. Returns ranked results with decisions
    """
    try:
        from ml_training.resume_parser import ResumeParser
        from ml_training.resume_ranking import ApprovalWorkflow, extract_skills_from_text
        
        if not model_manager or not model_manager.is_loaded:
            raise HTTPException(
                status_code=503, 
                detail="Models not loaded. Please train models first."
            )
        
        # Extract text from resumes
        parser = ResumeParser()
        resumes = []
        
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name
            
            try:
                text = parser.extract_text(tmp_path)
                if text:
                    resumes.append({
                        "filename": file.filename,
                        "text": text
                    })
            finally:
                os.unlink(tmp_path)
        
        if not resumes:
            raise HTTPException(status_code=400, detail="No valid resumes uploaded")
        
        # Rank resumes
        ranked = model_manager.rank_resumes(
            job_description, 
            resumes, 
            top_n=top_n
        )
        
        # Apply approval workflow
        approval = ApprovalWorkflow(
            shortlist_threshold=shortlist_threshold,
            review_threshold=review_threshold
        )
        ranked = approval.apply_to_rankings(ranked, score_field='combined_score')
        
        # Add category predictions
        for resume in ranked:
            try:
                prediction = model_manager.predict_category(resume['cleaned_text'])
                resume['predicted_category'] = prediction['category']
                resume['category_confidence'] = prediction['confidence']
            except Exception:
                resume['predicted_category'] = None
                resume['category_confidence'] = None
        
        return {
            "success": True,
            "job_description": job_description,
            "total_resumes": len(resumes),
            "ranked_count": len(ranked),
            "thresholds": {
                "shortlist": shortlist_threshold,
                "review": review_threshold
            },
            "results": ranked
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error screening resumes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-category")
async def predict_category(text: str = Form(...)):
    """
    Predict job category for a resume text.
    """
    try:
        if not model_manager or not model_manager.is_loaded:
            raise HTTPException(
                status_code=503, 
                detail="Models not loaded. Please train models first."
            )
        
        result = model_manager.predict_category(text)
        
        return {
            "success": True,
            **result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting category: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-skills")
async def extract_skills(text: str = Form(...)):
    """
    Extract skills from resume or job description text.
    """
    try:
        from ml_training.resume_ranking import extract_skills_from_text
        
        skills = extract_skills_from_text(text)
        
        return {
            "success": True,
            "skills": skills,
            "count": len(skills)
        }
        
    except Exception as e:
        logger.error(f"Error extracting skills: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/available-categories")
async def get_available_categories():
    """Get list of available job categories from trained model."""
    try:
        if not model_manager or not model_manager.is_loaded:
            return {
                "success": False,
                "message": "Models not loaded",
                "categories": []
            }
        
        categories = list(model_manager.label_encoder.classes_)
        
        return {
            "success": True,
            "categories": categories,
            "count": len(categories)
        }
        
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
