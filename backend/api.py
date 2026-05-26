"""
Backend API for Resume Screening System
FastAPI-based REST API for resume processing, classification, and ranking.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import tempfile
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Resume Screening API",
    description="API for AI-powered resume screening and approval system",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager = None


class ModelManager:
    """Manages MiniLM categorization + all-MiniLM-L6-v2 ranking models."""

    def __init__(self):
        self.bundle = None
        self.is_loaded = False

    def load_models(self, model_dir: str = "models") -> bool:
        try:
            from ml_training.inference import ModelBundle

            bundle = ModelBundle(model_dir)
            if not bundle.load():
                logger.warning("Model artifacts not found in %s", model_dir)
                return False

            self.bundle = bundle
            self.is_loaded = True
            logger.info(
                "Models loaded (categorization=%s, ranking=%s)",
                bundle.categorization_backend,
                bundle.config.get("ranking", {}).get("model"),
            )
            return True
        except Exception as e:
            logger.error("Error loading models: %s", e)
            return False

    def predict_category(self, text: str) -> Dict[str, Any]:
        if not self.is_loaded:
            raise ValueError("Models not loaded")
        return self.bundle.predict_category(text)

    def rank_resumes(
        self,
        job_description: str,
        resumes: List[Dict],
        top_n: int = 10,
    ) -> List[Dict]:
        if not self.is_loaded:
            raise ValueError("Models not loaded")
        return self.bundle.rank_resumes(job_description, resumes, top_n=top_n)


class JobDescription(BaseModel):
    text: str
    required_skills: Optional[List[str]] = None


class ResumeScreeningRequest(BaseModel):
    job_description: str
    shortlist_threshold: float = 0.6
    review_threshold: float = 0.4
    top_n: int = 10


class ScreeningResult(BaseModel):
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
    global model_manager
    import sys

    workspace_dir = Path(__file__).parent.parent
    if str(workspace_dir) not in sys.path:
        sys.path.insert(0, str(workspace_dir))

    model_manager = ModelManager()
    model_manager.load_models()
    logger.info("Application started")


@app.get("/")
async def root():
    config = {}
    if model_manager and model_manager.is_loaded:
        config = model_manager.bundle.config
    return {
        "message": "AI Resume Screening API",
        "version": "1.1.0",
        "status": "running",
        "models_loaded": model_manager.is_loaded if model_manager else False,
        "model_stack": config,
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": model_manager.is_loaded if model_manager else False,
    }


@app.post("/upload-resumes")
async def upload_resumes(files: List[UploadFile] = File(...)):
    try:
        from ml_training.resume_parser import ResumeParser

        parser = ResumeParser(use_spacy=True)
        results = []

        for file in files:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(file.filename).suffix
            ) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            try:
                text = parser.extract_text(tmp_path)
                if text:
                    results.append({
                        "filename": file.filename,
                        "text": text,
                        "file_type": Path(file.filename).suffix[1:].upper(),
                    })
            finally:
                os.unlink(tmp_path)

        return {"success": True, "count": len(results), "resumes": results}

    except Exception as e:
        logger.error("Error uploading resumes: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/screen-resumes")
async def screen_resumes(
    job_description: str = Form(...),
    shortlist_threshold: float = Form(0.6),
    review_threshold: float = Form(0.4),
    top_n: int = Form(10),
    files: List[UploadFile] = File(...),
):
    try:
        from ml_training.resume_parser import ResumeParser
        from ml_training.resume_ranking import ApprovalWorkflow

        if not model_manager or not model_manager.is_loaded:
            raise HTTPException(
                status_code=503,
                detail="Models not loaded. Train models first.",
            )

        parser = ResumeParser(use_spacy=True)
        resumes = []

        for file in files:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(file.filename).suffix
            ) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            try:
                text = parser.extract_text(tmp_path)
                if text:
                    resumes.append({"filename": file.filename, "text": text})
            finally:
                os.unlink(tmp_path)

        if not resumes:
            raise HTTPException(status_code=400, detail="No valid resumes uploaded")

        ranked = model_manager.rank_resumes(
            job_description, resumes, top_n=top_n
        )

        approval = ApprovalWorkflow(
            shortlist_threshold=shortlist_threshold,
            review_threshold=review_threshold,
        )
        ranked = approval.apply_to_rankings(ranked, score_field="combined_score")

        for resume in ranked:
            try:
                raw = resume.get("text", resume.get("cleaned_text", ""))
                prediction = model_manager.predict_category(raw)
                resume["predicted_category"] = prediction["category"]
                resume["category_confidence"] = prediction["confidence"]
            except Exception:
                resume["predicted_category"] = None
                resume["category_confidence"] = None

        return {
            "success": True,
            "job_description": job_description,
            "total_resumes": len(resumes),
            "ranked_count": len(ranked),
            "thresholds": {
                "shortlist": shortlist_threshold,
                "review": review_threshold,
            },
            "results": ranked,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error screening resumes: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-category")
async def predict_category(text: str = Form(...)):
    try:
        if not model_manager or not model_manager.is_loaded:
            raise HTTPException(
                status_code=503,
                detail="Models not loaded. Train models first.",
            )

        result = model_manager.predict_category(text)
        return {"success": True, **result}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error predicting category: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-skills")
async def extract_skills(text: str = Form(...)):
    try:
        from ml_training.text_preprocessing import extract_skills_from_text

        skills = extract_skills_from_text(text)
        return {"success": True, "skills": skills, "count": len(skills)}

    except Exception as e:
        logger.error("Error extracting skills: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/available-categories")
async def get_available_categories():
    try:
        if not model_manager or not model_manager.is_loaded:
            return {
                "success": False,
                "message": "Models not loaded",
                "categories": [],
            }

        categories = list(model_manager.bundle.label_encoder.classes_)

        return {
            "success": True,
            "categories": categories,
            "count": len(categories),
        }

    except Exception as e:
        logger.error("Error getting categories: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
