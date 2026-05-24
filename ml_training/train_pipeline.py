"""
Training Pipeline Script
End-to-end training pipeline for the resume screening system.
"""

import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_training_pipeline(
    data_path: str = None,
    output_dir: str = 'models',
    max_features: int = 5000,
    model_type: str = 'logistic',
    create_sample: bool = False,
    sample_size: int = 1000
):
    """
    Run the complete training pipeline.
    
    Args:
        data_path: Path to CSV dataset
        output_dir: Directory to save trained models
        max_features: Maximum TF-IDF features
        model_type: Type of model to train
        create_sample: Whether to create sample dataset
        sample_size: Size of sample dataset if created
    """
    logger.info("=" * 60)
    logger.info("Starting Resume Screening Training Pipeline")
    logger.info("=" * 60)
    
    # Import modules
    from ml_training.data_preparation import DataPreparator, create_sample_dataset
    from ml_training.text_preprocessing import TextPreprocessor
    from ml_training.model_training import ResumeClassifier, compare_models
    
    # Step 1: Prepare Data
    logger.info("\n📊 STEP 1: Data Preparation")
    logger.info("-" * 40)
    
    if create_sample or data_path is None:
        logger.info("Creating sample dataset...")
        data_path = create_sample_dataset(n_samples=sample_size)
    
    preparator = DataPreparator(data_path)
    df = preparator.load_data()
    
    # Explore data
    explorations = preparator.explore_data()
    logger.info(f"Dataset shape: {explorations['shape']}")
    logger.info(f"Missing values: {explorations['missing_values']}")
    logger.info(f"Duplicates: {explorations['duplicates']}")
    
    # Clean data
    df = preparator.clean_data()
    
    # Step 2: Text Preprocessing
    logger.info("\n🧹 STEP 2: Text Preprocessing")
    logger.info("-" * 40)
    
    preprocessor = TextPreprocessor(use_lemmatization=True)
    logger.info("Preprocessing resume texts...")
    
    df['cleaned_text'] = df['resume_text'].apply(preprocessor.preprocess)
    logger.info(f"Preprocessed {len(df)} resumes")
    
    # Sample of cleaned text
    logger.info(f"Sample cleaned text:\n{df['cleaned_text'].iloc[0][:200]}...")
    
    # Save cleaned data
    cleaned_path = Path('data/resumes_cleaned.csv')
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cleaned_path, index=False)
    logger.info(f"Saved cleaned data to {cleaned_path}")
    
    # Step 3: Model Training
    logger.info("\n🤖 STEP 3: Model Training")
    logger.info("-" * 40)
    
    classifier = ResumeClassifier(max_features=max_features, model_type=model_type)
    metrics = classifier.train(df, text_column='cleaned_text', label_column='category')
    
    # Step 4: Save Models
    logger.info("\n💾 STEP 4: Saving Models")
    logger.info("-" * 40)
    
    saved_paths = classifier.save(model_dir=output_dir)
    logger.info(f"Model files saved:")
    for name, path in saved_paths.items():
        logger.info(f"  - {name}: {path}")
    
    # Step 5: Validation
    logger.info("\n✅ STEP 5: Validation")
    logger.info("-" * 40)
    
    # Test prediction
    test_texts = [
        "experienced software engineer python java machine learning aws docker kubernetes",
        "data scientist with expertise in tensorflow pytorch statistical analysis deep learning",
        "product manager agile methodologies user research product strategy leadership"
    ]
    
    logger.info("Testing predictions:")
    for text in test_texts:
        prediction, confidence = classifier.predict_category(text)
        logger.info(f"  Input: {text[:50]}...")
        logger.info(f"  Predicted: {prediction} (confidence: {confidence:.2%})")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"\nFinal Metrics:")
    logger.info(f"  - Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  - F1 Score (macro): {metrics['f1_macro']:.4f}")
    logger.info(f"  - Number of classes: {len(metrics['classes'])}")
    logger.info(f"  - Features: {metrics['n_features']}")
    logger.info(f"\nModels saved to: {output_dir}/")
    logger.info(f"Cleaned data saved to: {cleaned_path}")
    
    return {
        'metrics': metrics,
        'model_paths': saved_paths,
        'cleaned_data_path': str(cleaned_path)
    }


def compare_all_models(data_path: str = None, sample_size: int = 500):
    """Compare different models and find the best one."""
    from ml_training.data_preparation import create_sample_dataset, DataPreparator
    from ml_training.text_preprocessing import TextPreprocessor
    from ml_training.model_training import compare_models
    
    logger.info("\n🔬 MODEL COMPARISON")
    logger.info("=" * 60)
    
    # Prepare data
    if data_path is None:
        data_path = create_sample_dataset(n_samples=sample_size)
    
    preparator = DataPreparator(data_path)
    df = preparator.load_data()
    df = preparator.clean_data()
    
    # Preprocess
    preprocessor = TextPreprocessor(use_lemmatization=True)
    df['cleaned_text'] = df['resume_text'].apply(preprocessor.preprocess)
    
    # Compare models
    results = compare_models(df, text_column='cleaned_text', label_column='category')
    
    # Print comparison summary
    logger.info("\n📊 MODEL COMPARISON SUMMARY")
    logger.info("-" * 40)
    logger.info(f"{'Model':<20} {'Accuracy':<12} {'F1 Macro':<12}")
    logger.info("-" * 40)
    
    for model_name, metrics in results.items():
        logger.info(f"{model_name.upper():<20} {metrics['accuracy']:<12.4f} {metrics['f1_macro']:<12.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train Resume Screening Models')
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        default=None,
        help='Path to training data CSV'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='models',
        help='Output directory for models'
    )
    parser.add_argument(
        '--max-features', '-f',
        type=int,
        default=5000,
        help='Maximum number of TF-IDF features'
    )
    parser.add_argument(
        '--model-type', '-m',
        type=str,
        default='logistic',
        choices=['logistic', 'random_forest', 'svm', 'knn'],
        help='Type of model to train'
    )
    parser.add_argument(
        '--compare', '-c',
        action='store_true',
        help='Compare all models instead of training one'
    )
    parser.add_argument(
        '--sample', '-s',
        action='store_true',
        help='Create sample dataset for testing'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=1000,
        help='Size of sample dataset'
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
            sample_size=args.sample_size
        )


if __name__ == "__main__":
    main()
