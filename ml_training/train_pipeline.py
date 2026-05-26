"""
Training Pipeline Script
End-to-end training pipeline for the resume screening system.
"""

# %% GPU check for Colab
import torch
print(torch.cuda.is_available())

import argparse
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_training_pipeline(
    data_path: str = None,
    output_dir: str = 'models',
    max_features: int = 8000,
    model_type: str = 'logistic',
    create_sample: bool = False,
    use_hf_dataset: bool = False,
    hf_max_rows: int = None,
    sample_size: int = 1000,
    balance_classes: bool = True,
    force_balance: bool = False,
    feature_backend: str = 'minilm',
):
    """
    Run the complete training pipeline.
    """
    logger.info("=" * 60)
    logger.info("Starting Resume Screening Training Pipeline")
    logger.info("=" * 60)

    from ml_training.data_preparation import DataPreparator, create_sample_dataset
    from ml_training.text_preprocessing import TextPreprocessor
    from ml_training.model_training import ResumeClassifier

    logger.info("\n📊 STEP 1: Data Preparation")
    logger.info("-" * 40)

    if use_hf_dataset:
        from ml_training.load_hf_dataset import download_and_prepare
        logger.info("Downloading Hugging Face Resume-Screening-Dataset...")
        data_path = download_and_prepare(max_rows=hf_max_rows)
    elif create_sample or data_path is None:
        logger.info("Creating varied sample dataset with holdout split...")
        data_path = create_sample_dataset(n_samples=sample_size)

    preparator = DataPreparator(data_path)
    df = preparator.load_data()

    explorations = preparator.explore_data()
    logger.info(f"Dataset shape: {explorations['shape']}")
    if explorations.get('imbalance_ratio'):
        logger.info(f"Class imbalance ratio: {explorations['imbalance_ratio']:.2f}")

    df = preparator.clean_data()

    if 'data_source' not in df.columns:
        df = preparator.assign_data_source(holdout_fraction=0.2)
    else:
        preparator.df = df

    n_rows = len(df)
    if balance_classes and n_rows > 5000 and not force_balance:
        logger.info(
            f"Skipping oversampling balance on large dataset ({n_rows} rows). "
            "Use --balance to force class balancing."
        )
        balance_classes = False

    if balance_classes:
        logger.info("Balancing train split only (preserving holdout)...")
        df = preparator.balance_classes(strategy='oversample', only_train=True)
        n_rows = len(df)
        logger.info(f"Rows after balancing: {n_rows}")

    logger.info("\n🧹 STEP 2: Text Preprocessing")
    logger.info("-" * 40)

    use_spacy_preprocess = n_rows <= 3000
    if not use_spacy_preprocess:
        logger.info(f"Fast preprocessing (no spaCy) for {n_rows} resumes")

    preprocessor = TextPreprocessor(
        use_spacy=use_spacy_preprocess,
        use_lemmatization=True,
        preserve_technical_terms=True,
        section_aware=True,
    )
    df['cleaned_text'] = df['resume_text'].apply(preprocessor.preprocess)
    logger.info(f"Preprocessed {len(df)} resumes")
    logger.info(f"Sample cleaned text:\n{df['cleaned_text'].iloc[0][:200]}...")

    cleaned_path = Path('data/resumes_cleaned.csv')
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cleaned_path, index=False)
    logger.info(f"Saved cleaned data to {cleaned_path}")

    logger.info("\n🤖 STEP 3: Model Training")
    logger.info("-" * 40)

    logger.info(f"Categorization backend: {feature_backend}")
    classifier = ResumeClassifier(
        max_features=max_features,
        model_type=model_type,
        use_structured_features=True,
        feature_backend=feature_backend,
    )
    metrics = classifier.train(
        df,
        text_column='cleaned_text',
        raw_text_column='resume_text',
        label_column='category',
    )

    logger.info("\n💾 STEP 4: Saving Models")
    logger.info("-" * 40)

    saved_paths = classifier.save(model_dir=output_dir)
    for name, path in saved_paths.items():
        logger.info(f"  - {name}: {path}")

    logger.info("\n✅ STEP 5: Validation")
    logger.info("-" * 40)

    test_cases = [
        (
            "Senior Software Engineer with 8+ years in C++, C#, .NET, Python 3.11, "
            "AWS, Docker, Kubernetes, and CI/CD.",
            None,
        ),
        (
            "Data scientist — TensorFlow 2.x, PyTorch, ML, NLP, statistical modeling, 5 years.",
            None,
        ),
        (
            "UX designer: Figma, user research, wireframes, UI/UX, 4 years experience.",
            None,
        ),
    ]

    for raw_text, _ in test_cases:
        cleaned = preprocessor.preprocess(raw_text)
        prediction, confidence = classifier.predict_category(cleaned, raw_text=raw_text)
        logger.info(f"  Input: {raw_text[:55]}...")
        logger.info(f"  Predicted: {prediction} (confidence: {confidence:.2%})")

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"  - Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  - F1 Score (macro): {metrics['f1_macro']:.4f}")
    if metrics.get('holdout_metrics'):
        hm = metrics['holdout_metrics']
        logger.info(f"  - Holdout accuracy: {hm['accuracy']:.4f} (n={hm['size']})")

    return {
        'metrics': metrics,
        'model_paths': saved_paths,
        'cleaned_data_path': str(cleaned_path),
    }


def compare_all_models(data_path: str = None, sample_size: int = 500):
    from ml_training.data_preparation import create_sample_dataset, DataPreparator
    from ml_training.text_preprocessing import TextPreprocessor
    from ml_training.model_training import compare_models

    if data_path is None:
        data_path = create_sample_dataset(n_samples=sample_size)

    preparator = DataPreparator(data_path)
    df = preparator.load_data()
    df = preparator.clean_data()
    df = preparator.balance_classes()

    preprocessor = TextPreprocessor(section_aware=True)
    df['cleaned_text'] = df['resume_text'].apply(preprocessor.preprocess)

    return compare_models(df, text_column='cleaned_text', label_column='category')


def main():
    parser = argparse.ArgumentParser(description='Train Resume Screening Models')
    parser.add_argument('--data', '-d', type=str, default=None)
    parser.add_argument('--output', '-o', type=str, default='models')
    parser.add_argument('--max-features', '-f', type=int, default=8000)
    parser.add_argument('--model-type', '-m', type=str, default='logistic',
                        choices=['logistic', 'random_forest', 'svm', 'knn'])
    parser.add_argument('--compare', '-c', action='store_true')
    parser.add_argument('--sample', '-s', action='store_true')
    parser.add_argument(
        '--hf',
        action='store_true',
        help='Train on AzharAli05/Resume-Screening-Dataset from Hugging Face',
    )
    parser.add_argument(
        '--hf-max-rows',
        type=int,
        default=None,
        help='Limit HF rows (for quick tests); default uses full ~10k dataset',
    )
    parser.add_argument('--sample-size', type=int, default=1000)
    parser.add_argument('--no-balance', action='store_true',
                        help='Skip class balancing oversampling')
    parser.add_argument(
        '--balance',
        action='store_true',
        help='Force oversampling balance even on large datasets (>5000 rows)',
    )
    parser.add_argument(
        '--feature-backend',
        type=str,
        default='minilm',
        choices=['minilm', 'tfidf'],
        help='Categorization: minilm (MiniLM embeddings) or tfidf',
    )

    args = parser.parse_args()

    if args.compare:
        compare_all_models(data_path=args.data, sample_size=args.sample_size)
    else:
        run_training_pipeline(
            data_path=args.data,
            output_dir=args.output,
            max_features=args.max_features,
            model_type=args.model_type,
            create_sample=args.sample,
            use_hf_dataset=args.hf,
            hf_max_rows=args.hf_max_rows,
            sample_size=args.sample_size,
            balance_classes=not args.no_balance,
            force_balance=args.balance,
            feature_backend=args.feature_backend,
        )


if __name__ == "__main__":
    main()
