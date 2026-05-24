# AI-Powered Resume Screening & Approval System

A comprehensive machine learning system for automated resume screening, categorization, and ranking against job descriptions.

## 📋 Features

- **Resume Parsing**: Extract text from PDF and DOCX files
- **Text Preprocessing**: Clean, normalize, and tokenize resume text
- **Job Category Classification**: Automatically categorize resumes into job roles
- **Similarity Ranking**: Rank resumes based on job description match
- **Skill Extraction**: Identify and match technical skills
- **Approval Workflow**: Automatic shortlist/review/reject decisions
- **Web Interface**: Interactive Streamlit dashboard for recruiters
- **REST API**: FastAPI backend for integration
- **Docker Support**: Containerized deployment

## 🏗️ Project Structure

```
/workspace
├── ml_training/           # Machine Learning modules
│   ├── data_preparation.py    # Data loading and cleaning
│   ├── text_preprocessing.py  # Text cleaning and NLP
│   ├── resume_parser.py       # PDF/DOCX text extraction
│   ├── model_training.py      # Model training and evaluation
│   ├── resume_ranking.py      # Similarity scoring and ranking
│   └── train_pipeline.py      # End-to-end training script
│
├── backend/               # REST API
│   └── api.py             # FastAPI application
│
├── frontend/              # Web UI
│   └── app.py             # Streamlit application
│
├── docker/                # Docker configuration
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── data/                  # Data directory
│   └── resumes_cleaned.csv
│
├── models/                # Trained models
│   ├── logistic_model.pkl
│   ├── vectorizer.pkl
│   └── label_encoder.pkl
│
└── requirements.txt       # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Train Models

```bash
# Train with sample data (for testing)
python ml_training/train_pipeline.py --sample --sample-size 1000

# Train with your own dataset
python ml_training/train_pipeline.py --data path/to/resumes.csv --model-type logistic
```

### 3. Run the Application

#### Option A: Run Separately

```bash
# Start the API backend
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000

# In another terminal, start the Streamlit frontend
streamlit run frontend/app.py --server.port 8501
```

#### Option B: Use Docker Compose

```bash
cd docker

# Build and run all services
docker-compose up -d

# Or run training first, then services
docker-compose --profile train up training
docker-compose up -d
```

### 4. Access the Application

- **Streamlit UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/health

## 📊 Usage Guide

### Using the Web Interface

1. **Enter Job Description**: Paste the full job description in the text area
2. **Upload Resumes**: Select PDF or DOCX resume files
3. **Adjust Settings** (optional):
   - Shortlist threshold (default: 0.6)
   - Review threshold (default: 0.4)
   - Skill match weight (default: 0.5)
4. **Click "Screen Resumes"**: Wait for processing
5. **Review Results**: See ranked candidates with scores and decisions
6. **Download Report**: Export results as CSV

### Using the API

```python
import requests

# Screen resumes
files = [('files', open('resume1.pdf', 'rb')), ('files', open('resume2.pdf', 'rb'))]
data = {'job_description': 'Software Engineer with Python experience...'}

response = requests.post('http://localhost:8000/screen-resumes', files=files, data=data)
results = response.json()

# Get ranked results
for candidate in results['results']:
    print(f"{candidate['filename']}: {candidate['combined_score']:.2%} - {candidate['decision']}")
```

## 🧠 Machine Learning Pipeline

### Phase 1: Data Preparation
- Load labeled resume dataset
- Handle missing values and duplicates
- Analyze class distribution

### Phase 2: Text Extraction
- Parse PDF files using pdfminer.six
- Parse DOCX files using python-docx
- Batch processing support

### Phase 3: Text Preprocessing
- Remove special characters, URLs, emails
- Convert to lowercase
- Remove stopwords
- Lemmatization/stemming

### Phase 4: Feature Engineering
- TF-IDF vectorization (max 5000 features)
- Label encoding for categories

### Phase 5: Model Training
- Logistic Regression (default)
- Random Forest, SVM, KNN alternatives
- Stratified train/test split
- Accuracy, F1-score evaluation

### Phase 6: Ranking
- Cosine similarity with job description
- Skill-based matching
- Weighted combined scoring

### Phase 7: Approval Workflow
- Shortlist: score ≥ 0.6
- Further Review: 0.4 ≤ score < 0.6
- Reject: score < 0.4

## 📁 Data Format

The system expects a CSV file with at least these columns:

```csv
resume_text,category
"Experienced software engineer with Python, Java...", "Software Engineer"
"Data scientist specializing in ML, TensorFlow...", "Data Scientist"
```

### Sample Dataset
For testing, use the `--sample` flag to generate synthetic data:
```bash
python ml_training/train_pipeline.py --sample --sample-size 1000
```

### Real Dataset
Download from Kaggle: [Resume Screening Dataset](https://www.kaggle.com/datasets/rhythmghai/resume-screening-dataset-200k-candidates/data)

## 🔧 Configuration

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| max_features | 5000 | TF-IDF feature limit |
| model_type | logistic | Model algorithm |
| shortlist_threshold | 0.6 | Score for shortlisting |
| review_threshold | 0.4 | Score for further review |
| skill_weight | 0.5 | Weight for skill matching |

### Threshold Tuning

Adjust thresholds based on your hiring needs:
- **Higher thresholds**: More selective, fewer false positives
- **Lower thresholds**: More inclusive, catch more potential candidates

## 📈 Evaluation Metrics

The training pipeline reports:
- **Accuracy**: Overall classification accuracy
- **F1 Score (macro)**: Balanced measure across classes
- **Classification Report**: Precision, recall, F1 per class
- **Confusion Matrix**: Detailed prediction breakdown

## 🔐 Privacy & Security

- All file processing happens locally
- Temporary files are cleaned up immediately
- No data is sent to external services
- For production: implement proper authentication and HTTPS

## 🛠️ Troubleshooting

### Models Not Loading
```bash
# Retrain models
python ml_training/train_pipeline.py --sample
```

### Missing Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Port Already in Use
```bash
# Change ports in docker-compose.yml or command line args
streamlit run frontend/app.py --server.port 8502
```

## 📝 License

MIT License - Feel free to use and modify for your needs.

## 🤝 Contributing

Contributions welcome! Please submit issues and pull requests.

## 📞 Support

For questions or issues, please open a GitHub issue.
