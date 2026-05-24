# AI Resume Screening System - Docker Commands

## Prerequisites
- Docker installed (v20.10+)
- Docker Compose installed (v2.0+)

## Quick Start

### 1. Train the ML Model (First Time Only)

Run the training service to create sample data and train the model:

```bash
cd /workspace/docker
docker compose --profile train up training
```

This will:
- Create a sample dataset with 1000 resumes
- Preprocess the text data
- Train a Logistic Regression classifier
- Save models to the `../models/` directory

### 2. Start the Application

After training completes, start the API and frontend:

```bash
docker compose up -d
```

This starts:
- **API Backend** on http://localhost:8000
- **Streamlit Frontend** on http://localhost:8501

### 3. Access the Services

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Frontend UI**: http://localhost:8501

### 4. View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f api
docker compose logs -f frontend
```

### 5. Stop Services

```bash
docker compose down
```

## Advanced Usage

### Rebuild Images

```bash
docker compose build
```

### Run Training with Custom Parameters

Edit the `command` section in the `training` service in `docker-compose.yml`:

```yaml
training:
  command: >
    python ml_training/train_pipeline.py 
    --data /app/data/your_dataset.csv
    --sample-size 2000
    --model-type random_forest
    --max-features 10000
```

Then run:
```bash
docker compose --profile train up training
```

### Compare Different Models

```bash
docker compose --profile train run training \
  python ml_training/train_pipeline.py --compare --sample
```

### Scale or Update

```bash
# Pull latest changes and rebuild
docker compose up -d --build

# Force recreate containers
docker compose up -d --force-recreate
```

## Troubleshooting

### Health Check Failing

The API health check has a 40-second start period to allow for model loading. If it still fails:
```bash
docker compose logs api
```

### Port Already in Use

Change the port mapping in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Instead of "8000:8000"
```

### Volume Permissions

If you encounter permission issues with mounted volumes:
```bash
sudo chown -R $USER:$USER ../models ../data ../logs
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   API Backend   │
│   (Streamlit)   │     │   (FastAPI)     │
│   :8501         │     │   :8000         │
└─────────────────┘     └────────┬────────┘
                                 │
                          ┌──────▼────────┐
                          │   Models      │
                          │   - vectorizer│
                          │   - classifier│
                          │   - encoder   │
                          └───────────────┘
```

## Services

| Service   | Port | Description                          |
|-----------|------|--------------------------------------|
| api       | 8000 | FastAPI backend for predictions      |
| frontend  | 8501 | Streamlit UI for resume screening    |
| training  | -    | One-time ML model training service   |

## Data Flow

1. **Training**: Sample data → Preprocessing → Model Training → Saved Models
2. **Prediction**: Upload Resume → Text Extraction → Preprocessing → Prediction → Results
