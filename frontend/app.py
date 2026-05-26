"""
Streamlit Frontend for AI Resume Screening System
Interactive dashboard for recruiters to screen and approve resumes.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from typing import List, Dict

# Page configuration
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #1E88E5;
    }
    .score-high {
        color: #4CAF50;
        font-weight: bold;
    }
    .score-medium {
        color: #FF9800;
        font-weight: bold;
    }
    .score-low {
        color: #F44336;
        font-weight: bold;
    }
    .decision-shortlist {
        background-color: #E8F5E9;
        padding: 5px 10px;
        border-radius: 5px;
        color: #2E7D32;
        font-weight: bold;
    }
    .decision-review {
        background-color: #FFF3E0;
        padding: 5px 10px;
        border-radius: 5px;
        color: #EF6C00;
        font-weight: bold;
    }
    .decision-reject {
        background-color: #FFEBEE;
        padding: 5px 10px;
        border-radius: 5px;
        color: #C62828;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def _load_cached_models(model_dir: str = None):
    """Cached loader: MiniLM categorization + all-MiniLM-L6-v2 ranking."""
    try:
        from ml_training.inference import ModelBundle

        if model_dir is None:
            possible_paths = [
                Path("models"),
                Path(__file__).parent.parent / "models",
                Path("/app/models"),
            ]
            model_path = Path("models")
            for p in possible_paths:
                if (p / "logistic_model.pkl").exists():
                    model_path = p
                    break
        else:
            model_path = Path(model_dir)

        bundle = ModelBundle(str(model_path))
        if not bundle.load():
            return None
        return bundle

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None


class ModelLoader:
    """Handles loading of ML models for the frontend."""

    def __init__(self):
        self.bundle = None
        self.is_loaded = False

    @property
    def ranker(self):
        return self.bundle.ranker if self.bundle else None

    @property
    def preprocessor(self):
        return self.bundle.preprocessor if self.bundle else None

    @property
    def label_encoder(self):
        return self.bundle.label_encoder if self.bundle else None

    def load_models(self, model_dir: str = None):
        bundle = _load_cached_models(model_dir)
        if bundle is None:
            return False
        self.bundle = bundle
        self.is_loaded = True
        return True

    def predict_category(self, text: str):
        return self.bundle.predict_category(text)


def extract_skills(text: str) -> List[str]:
    """Extract skills from text."""
    from ml_training.text_preprocessing import extract_skills_from_text
    return extract_skills_from_text(text)


def process_uploaded_files(uploaded_files) -> List[Dict]:
    """Process uploaded resume files and extract text."""
    from ml_training.resume_parser import ResumeParser
    
    parser = ResumeParser()
    resumes = []
    
    for uploaded_file in uploaded_files:
        try:
            # Save temporarily
            temp_path = f"/tmp/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Extract text
            text = parser.extract_text(temp_path)
            
            if text:
                resumes.append({
                    'filename': uploaded_file.name,
                    'text': text,
                    'file_type': Path(uploaded_file.name).suffix[1:].upper()
                })
            
            # Clean up
            os.remove(temp_path)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
    
    return resumes


def make_decision(score: float, shortlist_threshold: float, review_threshold: float) -> str:
    """Make approval decision based on score."""
    if score >= shortlist_threshold:
        return 'Shortlist'
    elif score >= review_threshold:
        return 'Further Review'
    else:
        return 'Reject'


def render_decision(decision: str) -> str:
    """Render decision with appropriate styling."""
    if decision == 'Shortlist':
        return f'<span class="decision-shortlist">✓ {decision}</span>'
    elif decision == 'Further Review':
        return f'<span class="decision-review">⚠ {decision}</span>'
    else:
        return f'<span class="decision-reject">✗ {decision}</span>'


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">📄 AI-Powered Resume Screening & Approval System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Automatically screen, rank, and approve resumes using machine learning</p>', unsafe_allow_html=True)
    
    # Initialize model loader
    model_loader = ModelLoader()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model status
        # Model Status Section
        st.subheader("Model Status")
        models_loaded = model_loader.load_models()
        
        categories = []
        if models_loaded:
            if model_loader.label_encoder is not None:
                try:
                    if hasattr(model_loader.label_encoder, 'classes_'):
                        categories = list(model_loader.label_encoder.classes_)
                        st.success("✅ Models loaded successfully")
                        st.info(f"Available categories: {len(categories)}")
                    else:
                        st.warning("⚠️ Label encoder missing classes attribute")
                except Exception as e:
                    st.error(f"Error accessing label encoder: {e}")
                    categories = []
            else:
                st.error("❌ Models loaded but label encoder is None")
                st.info("Try restarting the application or retraining models.")
        else:
            st.error("❌ Models not loaded. Please train models first.")
            st.info("Run training script to create models: `python ml_training/train_pipeline.py --sample --sample-size 1000`")
        
        st.divider()
        
        # Threshold settings
        st.subheader("Approval Thresholds")
        shortlist_threshold = st.slider(
            "Shortlist Threshold",
            min_value=0.3,
            max_value=0.9,
            value=0.6,
            step=0.05,
            help="Resumes above this score will be shortlisted"
        )
        review_threshold = st.slider(
            "Further Review Threshold",
            min_value=0.2,
            max_value=0.7,
            value=0.4,
            step=0.05,
            help="Resumes between this and shortlist threshold need further review"
        )
        
        st.divider()
        
        # Ranking settings
        st.subheader("Ranking Settings")
        top_n = st.slider("Top N Candidates", 5, 50, 10)
        skill_weight = st.slider("Skill Match Weight", 0.0, 1.0, 0.5, 0.1)
        similarity_weight = 1.0 - skill_weight
        
        st.divider()
        
        # Info
        st.info("""
        ### How to Use:
        1. Enter a job description
        2. Upload resume files (PDF/DOCX)
        3. Click 'Screen Resumes'
        4. Review ranked results
        5. Download shortlist report
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Job Description")
        job_description = st.text_area(
            "Enter the job description",
            height=200,
            placeholder="Paste the full job description here...",
            help="The system will match resumes against this description"
        )
        
        # Extract and display skills from job description
        if job_description:
            jd_skills = extract_skills(job_description)
            if jd_skills:
                st.caption(f"🎯 Detected skills: {', '.join(jd_skills)}")
    
    with col2:
        st.subheader("📎 Upload Resumes")
        uploaded_files = st.file_uploader(
            "Upload resume files",
            type=['pdf', 'docx'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
    
    # Screen button
    col1, col2, col3 = st.columns([3, 1, 3])
    with col2:
        screen_button = st.button(
            "🔍 Screen Resumes",
            type="primary",
            use_container_width=True
        )
    
    # Results section
    if screen_button:
        if not job_description:
            st.warning("⚠️ Please enter a job description")
        elif not uploaded_files:
            st.warning("⚠️ Please upload at least one resume")
        elif not model_loader.is_loaded:
            st.error("❌ Models not loaded. Please train models first.")
        else:
            with st.spinner("🔄 Processing resumes..."):
                try:
                    # Process uploaded files
                    resumes = process_uploaded_files(uploaded_files)
                    
                    if not resumes:
                        st.error("No valid resumes could be processed")
                    else:
                        # Rank resumes
                        ranked = model_loader.ranker.rank_with_skills(
                            job_description,
                            resumes,
                            skill_weight=skill_weight,
                            similarity_weight=similarity_weight,
                            top_n=top_n
                        )
                        
                        # Apply approval decisions
                        for resume in ranked:
                            resume['decision'] = make_decision(
                                resume['combined_score'],
                                shortlist_threshold,
                                review_threshold
                            )
                            
                            try:
                                raw = resume.get('text', resume.get('cleaned_text', ''))
                                pred = model_loader.predict_category(raw)
                                resume['predicted_category'] = pred['category']
                                resume['category_confidence'] = pred['confidence']
                            except Exception:
                                resume['predicted_category'] = 'Unknown'
                                resume['category_confidence'] = 0.0
                        
                        # Display summary
                        st.divider()
                        st.subheader("📊 Screening Results")
                        
                        # Summary metrics
                        shortlist_count = sum(1 for r in ranked if r['decision'] == 'Shortlist')
                        review_count = sum(1 for r in ranked if r['decision'] == 'Further Review')
                        reject_count = sum(1 for r in ranked if r['decision'] == 'Reject')
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Resumes", len(resumes))
                        col2.metric("Shortlisted", shortlist_count, delta_color="normal")
                        col3.metric("Further Review", review_count)
                        col4.metric("Rejected", reject_count)
                        
                        st.divider()
                        
                        # Detailed results
                        st.subheader("🏆 Ranked Candidates")
                        
                        for i, resume in enumerate(ranked, 1):
                            score_color = "score-high" if resume['combined_score'] >= 0.6 else \
                                         "score-medium" if resume['combined_score'] >= 0.4 else "score-low"
                            
                            with st.container():
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.markdown(f"**#{i} - {resume['filename']}**")
                                    st.caption(f"File Type: {resume.get('file_type', 'Unknown')}")
                                    
                                    # Skills
                                    if resume.get('matched_skills'):
                                        st.markdown(f"""
                                        **Matched Skills:**  
                                        {', '.join(resume['matched_skills'])}
                                        """)
                                    
                                    if resume.get('missing_skills'):
                                        st.markdown(f"""
                                        **Missing Skills:**  
                                        {', '.join(resume['missing_skills'])}
                                        """)
                                    
                                    # Category prediction
                                    if resume.get('predicted_category'):
                                        conf_pct = resume['category_confidence'] * 100
                                        st.caption(f"Predicted Category: {resume['predicted_category']} ({conf_pct:.1f}% confidence)")
                                
                                with col2:
                                    st.markdown(f"""
                                    <div style="text-align: right;">
                                        <p class="{score_color}">{resume['combined_score']:.2%}</p>
                                        {render_decision(resume['decision'])}
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    with st.expander("Details"):
                                        st.write(f"Similarity: {resume['similarity_score']:.2%}")
                                        st.write(f"Skill Match: {resume['skill_match_score']:.2%}")
                                
                                st.divider()
                        
                        # Export functionality
                        st.subheader("💾 Export Results")
                        
                        # Prepare DataFrame for export
                        export_data = []
                        for resume in ranked:
                            export_data.append({
                                'Rank': ranked.index(resume) + 1,
                                'Filename': resume['filename'],
                                'Decision': resume['decision'],
                                'Combined Score': round(resume['combined_score'], 4),
                                'Similarity Score': round(resume['similarity_score'], 4),
                                'Skill Match': round(resume['skill_match_score'], 4),
                                'Predicted Category': resume.get('predicted_category', 'Unknown'),
                                'Category Confidence': round(resume.get('category_confidence', 0.0), 4),
                                'Matched Skills': ', '.join(resume.get('matched_skills', [])),
                                'Missing Skills': ', '.join(resume.get('missing_skills', []))
                            })
                        
                        df_export = pd.DataFrame(export_data)
                        
                        # CSV download
                        csv = df_export.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"resume_screening_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                except Exception as e:
                    st.error(f"Error during screening: {e}")
    
    # Footer
    st.divider()
    st.caption("AI Resume Screening System v1.0 | Powered by Machine Learning")


if __name__ == "__main__":
    main()
