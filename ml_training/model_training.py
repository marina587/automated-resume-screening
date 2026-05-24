"""
Model Training Module
Handles feature engineering, model training, evaluation, and saving.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score
)


class ResumeClassifier:
    """Handles training and evaluation of resume classification models."""
    
    def __init__(self, max_features: int = 5000, model_type: str = 'logistic'):
        """
        Initialize the classifier.
        
        Args:
            max_features: Maximum number of TF-IDF features
            model_type: Type of model to use ('logistic', 'random_forest', 'svm', 'knn')
        """
        self.max_features = max_features
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=2,
            max_df=0.95
        )
        self.label_encoder = LabelEncoder()
        self.model = None
        self.is_trained = False
    
    def _create_model(self):
        """Create the classification model based on model_type."""
        if self.model_type == 'logistic':
            return LogisticRegression(
                max_iter=1000,
                C=1.0,
                class_weight='balanced',
                random_state=42
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'svm':
            return SVC(
                kernel='rbf',
                C=1.0,
                class_weight='balanced',
                random_state=42,
                probability=True
            )
        elif self.model_type == 'knn':
            return KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def prepare_features(self, texts: list) -> np.ndarray:
        """
        Convert text to TF-IDF features.
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            TF-IDF feature matrix
        """
        return self.vectorizer.transform(texts)
    
    def encode_labels(self, labels: list) -> np.ndarray:
        """
        Encode categorical labels to integers.
        
        Args:
            labels: List of category labels
            
        Returns:
            Encoded labels
        """
        return self.label_encoder.fit_transform(labels)
    
    def train(self, df: pd.DataFrame, text_column: str = 'cleaned_text', 
              label_column: str = 'category', test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the classification model.
        
        Args:
            df: DataFrame with resume data
            text_column: Name of column containing preprocessed text
            label_column: Name of column containing labels
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with training results and metrics
        """
        # Prepare data
        texts = df[text_column].fillna('').astype(str).tolist()
        labels = df[label_column].tolist()
        
        # Fit vectorizer on all data (will split later)
        X = self.vectorizer.fit_transform(texts)
        y = self.encode_labels(labels)
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=42, 
            stratify=y
        )
        
        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'classification_report': classification_report(
                y_test, y_pred, 
                target_names=self.label_encoder.classes_
            ),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'train_size': X_train.shape[0],
            'test_size': X_test.shape[0],
            'n_features': X.shape[1],
            'classes': list(self.label_encoder.classes_)
        }
        
        self.is_trained = True
        
        print(f"Training completed!")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score (macro): {metrics['f1_macro']:.4f}")
        print(f"Number of features: {metrics['n_features']}")
        print(f"\nClassification Report:\n{metrics['classification_report']}")
        
        return metrics
    
    def predict(self, texts: list) -> Tuple[list, list]:
        """
        Predict categories for new texts.
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            Tuple of (predicted labels, predicted probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X = self.prepare_features(texts)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Convert encoded predictions back to labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return predicted_labels, probabilities
    
    def predict_category(self, text: str) -> Tuple[str, float]:
        """
        Predict category for a single text.
        
        Args:
            text: Preprocessed resume text
            
        Returns:
            Tuple of (predicted category, confidence score)
        """
        labels, probs = self.predict([text])
        predicted_label = labels[0]
        confidence = float(probs[0].max())
        
        return predicted_label, confidence
    
    def save(self, model_dir: str = 'models'):
        """
        Save the trained model and vectorizer.
        
        Args:
            model_dir: Directory to save models
        """
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = Path(model_dir) / f'{self.model_type}_model.pkl'
        joblib.dump(self.model, model_path)
        
        # Save vectorizer
        vectorizer_path = Path(model_dir) / 'vectorizer.pkl'
        joblib.dump(self.vectorizer, vectorizer_path)
        
        # Save label encoder
        encoder_path = Path(model_dir) / 'label_encoder.pkl'
        joblib.dump(self.label_encoder, encoder_path)
        
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
        print(f"Label encoder saved to {encoder_path}")
        
        return {
            'model': str(model_path),
            'vectorizer': str(vectorizer_path),
            'label_encoder': str(encoder_path)
        }
    
    def load(self, model_dir: str = 'models', model_type: str = None):
        """
        Load a trained model and vectorizer.
        
        Args:
            model_dir: Directory containing saved models
            model_type: Type of model to load (optional)
        """
        model_type = model_type or self.model_type
        
        model_path = Path(model_dir) / f'{model_type}_model.pkl'
        vectorizer_path = Path(model_dir) / 'vectorizer.pkl'
        encoder_path = Path(model_dir) / 'label_encoder.pkl'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.label_encoder = joblib.load(encoder_path)
        self.is_trained = True
        
        print(f"Loaded model from {model_path}")
        print(f"Loaded vectorizer from {vectorizer_path}")
        print(f"Loaded label encoder from {encoder_path}")


def compare_models(df: pd.DataFrame, text_column: str = 'cleaned_text',
                   label_column: str = 'category') -> Dict[str, Dict]:
    """
    Compare different models and return their performance metrics.
    
    Args:
        df: DataFrame with resume data
        text_column: Column with preprocessed text
        label_column: Column with labels
        
    Returns:
        Dictionary with metrics for each model
    """
    models = ['logistic', 'random_forest', 'svm', 'knn']
    results = {}
    
    for model_type in models:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()} model...")
        print('='*50)
        
        classifier = ResumeClassifier(max_features=5000, model_type=model_type)
        metrics = classifier.train(df, text_column, label_column)
        results[model_type] = metrics
    
    # Find best model
    best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
    print(f"\n{'='*50}")
    print(f"Best model: {best_model.upper()} with accuracy {results[best_model]['accuracy']:.4f}")
    print('='*50)
    
    return results


if __name__ == "__main__":
    # Example usage with sample data
    from data_preparation import create_sample_dataset, DataPreparator
    from text_preprocessing import TextPreprocessor
    
    # Create sample dataset
    sample_path = create_sample_dataset(n_samples=500)
    
    # Load and prepare data
    preparator = DataPreparator(sample_path)
    df = preparator.load_data()
    df = preparator.clean_data()
    
    # Preprocess text
    preprocessor = TextPreprocessor(use_lemmatization=True)
    df['cleaned_text'] = df['resume_text'].apply(preprocessor.preprocess)
    
    # Train model
    classifier = ResumeClassifier(max_features=3000, model_type='logistic')
    metrics = classifier.train(df, text_column='cleaned_text', label_column='category')
    
    # Save model
    classifier.save()
    
    # Test prediction
    test_text = "experienced software engineer python java machine learning aws docker"
    prediction, confidence = classifier.predict_category(test_text)
    print(f"\nTest prediction: '{test_text}'")
    print(f"Predicted category: {prediction} (confidence: {confidence:.2%})")
