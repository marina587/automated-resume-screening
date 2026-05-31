# AI-Powered Resume Screening & Approval System

A complete machine learning system for automated resume screening, category prediction, resume ranking, and hiring decisions.

## 📋 Key Features

- **Resume Parsing**: PDF and DOCX text extraction via PyMuPDF + spaCy
- **Text Preprocessing**: Section-aware cleaning, lemmatization, and technical-term preservation
- **Category Prediction**: MiniLM embeddings + classifier, with optional TF-IDF fallback
- **Resume Ranking**: Semantic similarity scoring using sentence-transformers/all-MiniLM-L6-v2
- **Skill Matching**: Extracts and compares skills from resumes and job queries
- **Approval Workflow**: Shortlist / Further Review / Reject decisions
- **Web UI**: Streamlit dashboard for recruiters
- **API Integration**: FastAPI backend for programmatic screening
- **Docker Support**: Containerized deployment and training options

## 🏗️ Project Structure

```
.
├── ml_training/                # Machine learning modules
│   ├── __init__.py
│   ├── data_preparation.py      # Load, clean, and prepare training data
│   ├── text_preprocessing.py    # Resume cleaning and skill extraction
│   ├── resume_parser.py         # PDF/DOCX text extraction
│   ├── resume_features.py       # Structured resume signals
│   ├── embedding_models.py      # Sentence-transformer helpers and config
│   ├── model_training.py        # Training, evaluation, and saving models
│   ├── resume_ranking.py        # Similarity scoring and ranking logic
│   ├── inference.py             # ModelBundle for API/frontend inference
│   ├── train_pipeline.py        # End-to-end training script
│   ├── load_hf_dataset.py       # Hugging Face dataset download/preparation
│   ├── inspect_dataset.py       # Dataset inspection utilities
│   ├── data/                    # Data artifacts and cleaned CSVs
│   └── models/                  # Saved model artifacts
│
├── backend/                    # FastAPI REST API
│   └── api.py
├── frontend/                   # Streamlit user interface
│   └── app.py
├── docker/                     # Docker setup and Compose
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── README.md
├── scripts/                    # Helper scripts
│   └── list_categories.py
├── data/                       # Dataset files and exports
├── models/                     # Trained models and config
├── logs/                       # Application logs
├── resume_features.py          # Compatibility alias
└── requirements.txt            # Python dependencies
```

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Train the models

```bash
# Train on the Hugging Face dataset
python ml_training/train_pipeline.py --hf

# Quick test with limited rows
python ml_training/train_pipeline.py --hf --hf-max-rows 2000

# Train with sample data for fast iteration
python ml_training/train_pipeline.py --sample --sample-size 1000

# Use TF-IDF instead of MiniLM features
python ml_training/train_pipeline.py --sample --feature-backend tfidf

# Train with your own dataset
python ml_training/train_pipeline.py --data path/to/resumes.csv --model-type logistic
```

### 3. Run the stack locally

```bash
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000
streamlit run frontend/app.py --server.port 8501
```

### 4. Run with Docker Compose

```bash
cd docker
docker compose up -d
```

To run training via Docker:

```bash
docker compose --profile train up training
```

## 🌐 Access Points

- Streamlit UI: http://localhost:8501
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## 📌 What This Project Does

This repository builds an automated resume screening system that can:

- extract text from resumes
- preprocess and clean resume content
- classify resumes into job categories
- rank resumes by relevance to a job description or structured hiring criteria
- extract and match skills
- apply automated shortlist/review/reject decisions
- expose a user interface and an API for integration

## 🧠 How It Works

### Resume processing

- Extract resume text from PDF/DOCX files
- Clean and normalize text with spaCy-based preprocessing
- Preserve technology terms and section structure

### Category prediction

- Uses sentence-transformer embeddings or TF-IDF features
- Trains a classifier to predict job categories
- Calibrates probabilities and supports fallback to `Unknown`

### Resume ranking

- Computes semantic similarity between the query and resumes
- Extracts skills from both query and resume text
- Produces a combined score from similarity + skill match
- Ranks candidates by final combined score

### Approval workflow

- Shortlist when score >= shortlist threshold
- Further Review when score >= review threshold
- Reject otherwise

## 🖥️ Frontend Usage

### Candidate Search Modes

The Streamlit UI supports two ranking entry modes:

- **Job Description**: paste a full job description into the text area
- **Search Parameters**: choose structured criteria such as:
  - job title
  - seniority
  - years of experience
  - industry
  - location
  - education level
  - remote preference
  - required skills
  - preferred skills

The selected search prompt is used to rank resumes.

### Ranking controls

- Shortlist Threshold: score above this value becomes Shortlist
- Further Review Threshold: score above this value becomes Further Review
- Skill Match Weight: balances semantic similarity vs. skill coverage
- Top N Candidates: number of resumes shown

### Results

The app displays:

- ranked resumes
- similarity score
- skill match score
- combined score
- decision label
- predicted category and confidence
- matched and missing skills
- CSV export of results

## 🔧 Training Workflow

`ml_training/train_pipeline.py` is the main orchestration script.

Key training options:

- `--data`: path to a CSV dataset
- `--output`: model output directory
- `--max-features`: TF-IDF feature size
- `--model-type`: classifier type (`logistic`, `random_forest`, `gradient_boosting`, `svm`, `knn`)
- `--hf`: use the Hugging Face dataset
- `--hf-max-rows`: limit HF rows for testing
- `--sample`: create a sample dataset
- `--sample-size`: number of sample rows
- `--no-balance`: disable oversampling
- `--balance`: force oversampling even on large datasets
- `--feature-backend`: `minilm` or `tfidf`

## 📂 Data Format

Expected CSV columns:

- `resume_text`
- `category`

Optional columns can include dataset metadata such as `data_source`, `candidate_name`, or `source_url`.

## 🔎 Model Files

Saved artifacts in `models/` include:

- `*_model.pkl` (e.g. `gradient_boosting_model.pkl`)
- `label_encoder.pkl`
- `vectorizer.pkl` (when using TF-IDF)
- `feature_extractor.pkl`
- `model_config.json`

## 🛠️ Troubleshooting

### Models not loading

- Ensure `models/label_encoder.pkl` exists.
- Ensure a classifier file exists: `*_model.pkl`.
- Check that `model_config.json` is valid and points to the correct model type.

### Training slow or hanging

- Use `--hf-max-rows` to reduce dataset size for testing.
- Use `--feature-backend tfidf` for a lighter feature pipeline.
- Confirm the model training process is still running with `Get-Process python` or a task manager.

### Frontend issues

- Restart Streamlit after training finishes.
- Verify `models/` contains trained artifacts.
- If using Docker, confirm volume mappings for `models/`, `data/`, and `logs/`.

## 📦 Docker Notes

`docker/docker-compose.yml` defines:

- `api`: FastAPI backend
- `frontend`: Streamlit app
- `training`: one-time training job under the `train` profile

Shared volumes:

- `models`
- `data`
- `logs`

## 📈 Evaluation Metrics

Training reports:

- accuracy
- macro F1 score
- weighted F1 score
- classification report
- confusion matrix
- expected calibration error
- optional holdout metrics

## ✅ Summary

This repository implements a full resume screening pipeline:

- document parsing
- preprocessing
- classification
- semantic ranking
- skill matching
- recruiter UI
- API integration
- Docker deployment

Train with your own labeled resumes and tune thresholds for the best hiring results.
