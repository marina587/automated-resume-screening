"""
Model Training Module
Handles feature engineering, model training, evaluation, and saving.
Combines TF-IDF text features with resume-specific structured features.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import joblib
from scipy.sparse import hstack, csr_matrix
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
    f1_score,
)

from .resume_features import ResumeFeatureExtractor
from .embedding_models import (
    CATEGORIZATION_MODEL_ID,
    embed_texts,
    save_model_config,
    load_model_config,
)


class ResumeClassifier:
    """Handles training and evaluation of resume classification models."""

    def __init__(
        self,
        max_features: int = 8000,
        model_type: str = 'logistic',
        use_structured_features: bool = True,
        feature_backend: str = 'minilm',
        categorization_model_id: str = CATEGORIZATION_MODEL_ID,
    ):
        self.max_features = max_features
        self.model_type = model_type
        self.use_structured_features = use_structured_features
        self.feature_backend = feature_backend  # 'minilm' | 'tfidf'
        self.categorization_model_id = categorization_model_id
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            token_pattern=r'(?u)\b[a-z][a-z0-9]+\b',
        )
        self.feature_extractor = ResumeFeatureExtractor()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.is_trained = False

    def _create_model(self):
        if self.model_type == 'logistic':
            return LogisticRegression(
                max_iter=2000,
                C=1.0,
                class_weight='balanced',
                random_state=42,
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
            )
        elif self.model_type == 'svm':
            return SVC(
                kernel='rbf',
                C=1.0,
                class_weight='balanced',
                random_state=42,
                probability=True,
            )
        elif self.model_type == 'knn':
            return KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _build_feature_matrix(
        self,
        texts: List[str],
        raw_texts: List[str],
        fit: bool = False,
        show_progress: bool = False,
    ):
        if self.feature_backend == 'minilm':
            # Truncate long resumes for faster embedding (MiniLM max ~512 tokens)
            truncated = [t[:3000] if t else "" for t in texts]
            X_emb = embed_texts(
                truncated,
                model_id=self.categorization_model_id,
                show_progress=show_progress,
            )
            if self.use_structured_features:
                if fit:
                    X_struct = self.feature_extractor.fit_transform(raw_texts)
                else:
                    X_struct = self.feature_extractor.transform(raw_texts)
                return np.hstack([X_emb, X_struct])
            return X_emb

        if fit:
            X_text = self.vectorizer.fit_transform(texts)
            if self.use_structured_features:
                X_struct = self.feature_extractor.fit_transform(raw_texts)
        else:
            X_text = self.vectorizer.transform(texts)
            if self.use_structured_features:
                X_struct = self.feature_extractor.transform(raw_texts)

        if self.use_structured_features:
            return hstack([X_text, csr_matrix(X_struct)])
        return X_text

    def encode_labels(self, labels: list, fit: bool = True) -> np.ndarray:
        if fit:
            return self.label_encoder.fit_transform(labels)
        return self.label_encoder.transform(labels)

    def train(
        self,
        df: pd.DataFrame,
        text_column: str = 'cleaned_text',
        raw_text_column: str = 'resume_text',
        label_column: str = 'category',
        test_size: float = 0.2,
        data_source_column: str = 'data_source',
        calibrate: bool = True,
    ) -> Dict[str, Any]:
        """
        Train with proper train/test split (no leakage on vectorizer).
        Optionally evaluate on holdout rows tagged in data_source_column.
        """
        texts = df[text_column].fillna('').astype(str).tolist()
        raw_texts = (
            df[raw_text_column].fillna('').astype(str).tolist()
            if raw_text_column in df.columns
            else texts
        )
        labels = df[label_column].tolist()

        # Holdout set from different distribution (e.g. alternate wording)
        holdout_mask = None
        if data_source_column in df.columns:
            holdout_mask = df[data_source_column] == 'holdout'
            train_df = df[~holdout_mask].copy()
            holdout_df = df[holdout_mask].copy()
        else:
            train_df = df
            holdout_df = pd.DataFrame()

        train_texts = train_df[text_column].fillna('').astype(str).tolist()
        train_raw = (
            train_df[raw_text_column].fillna('').astype(str).tolist()
            if raw_text_column in train_df.columns
            else train_texts
        )
        train_labels = train_df[label_column].tolist()

        # Fit label encoding on the full dataset so holdout classes are supported
        self.encode_labels(df[label_column].tolist(), fit=True)
        y = self.encode_labels(train_labels, fit=False)

        indices = np.arange(len(train_texts))
        try:
            train_idx, test_idx = train_test_split(
                indices,
                test_size=test_size,
                random_state=42,
                stratify=y,
            )
        except ValueError as exc:
            print(
                "Warning: could not stratify train/test split due to small or "
                "imbalanced class counts. Falling back to random split."
            )
            train_idx, test_idx = train_test_split(
                indices,
                test_size=test_size,
                random_state=42,
                stratify=None,
            )

        if self.feature_backend == 'minilm':
            print(f"Encoding {len(train_texts)} training resumes with MiniLM (one pass)...")
            X_all = self._build_feature_matrix(
                train_texts, train_raw, fit=True, show_progress=True
            )
            X_train = X_all[train_idx]
            X_test = X_all[test_idx]
        else:
            print(f"Extracting TF-IDF features for {len(train_texts)} training resumes...")
            X_train = self._build_feature_matrix(
                [train_texts[i] for i in train_idx],
                [train_raw[i] for i in train_idx],
                fit=True,
            )
            X_test = self._build_feature_matrix(
                [train_texts[i] for i in test_idx],
                [train_raw[i] for i in test_idx],
                fit=False,
            )
        y_train, y_test = y[train_idx], y[test_idx]

        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        
        # Optionally calibrate to improve confidence accuracy
        if calibrate and len(y_train) >= 30:  # Need sufficient data for calibration
            from sklearn.calibration import CalibratedClassifierCV
            self.model = CalibratedClassifierCV(
                self.model, 
                method='sigmoid',  # Works with any classifier
                cv=5
            )
            self.model.fit(X_train, y_train)
            print("✓ Model calibrated with CalibratedClassifierCV (sigmoid method)")
        
        y_pred = self.model.predict(X_test)

        all_label_ids = list(range(len(self.label_encoder.classes_)))
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'classification_report': classification_report(
                y_test,
                y_pred,
                labels=all_label_ids,
                target_names=self.label_encoder.classes_,
                zero_division=0,
            ),
            'confusion_matrix': confusion_matrix(y_test, y_pred, labels=all_label_ids),
            'train_size': X_train.shape[0],
            'test_size': X_test.shape[0],
            'n_features': X_train.shape[1],
            'classes': list(self.label_encoder.classes_),
            'holdout_metrics': None,
        }

        if not holdout_df.empty and len(holdout_df) > 0:
            holdout_texts = holdout_df[text_column].fillna('').astype(str).tolist()
            holdout_raw = (
                holdout_df[raw_text_column].fillna('').astype(str).tolist()
                if raw_text_column in holdout_df.columns
                else holdout_texts
            )
            holdout_labels = self.encode_labels(
                holdout_df[label_column].tolist(), fit=False
            )
            if self.feature_backend == 'minilm':
                print(f"Encoding {len(holdout_texts)} holdout resumes with MiniLM...")
            else:
                print(f"Transforming {len(holdout_texts)} holdout resumes with TF-IDF...")
            X_holdout = self._build_feature_matrix(
                holdout_texts, holdout_raw, fit=False, show_progress=True
            )
            y_holdout_pred = self.model.predict(X_holdout)
            metrics['holdout_metrics'] = {
                'size': len(holdout_df),
                'accuracy': accuracy_score(holdout_labels, y_holdout_pred),
                'f1_macro': f1_score(
                    holdout_labels, y_holdout_pred, average='macro'
                ),
            }

        self.is_trained = True

        print("Training completed!")
        print(f"Accuracy (stratified test): {metrics['accuracy']:.4f}")
        print(f"F1 Score (macro): {metrics['f1_macro']:.4f}")
        print(f"Number of features: {metrics['n_features']}")
        if metrics['holdout_metrics']:
            hm = metrics['holdout_metrics']
            print(
                f"Holdout evaluation ({hm['size']} samples): "
                f"accuracy={hm['accuracy']:.4f}, f1_macro={hm['f1_macro']:.4f}"
            )
        print(f"\nClassification Report:\n{metrics['classification_report']}")

        return metrics

    def _transform_input(self, text: str, raw_text: Optional[str] = None) -> csr_matrix:
        raw = raw_text if raw_text is not None else text
        return self._build_feature_matrix([text], [raw], fit=False)

    def predict(
        self,
        texts: list,
        raw_texts: Optional[list] = None,
    ) -> Tuple[list, list]:
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        raw_texts = raw_texts or texts
        predictions = []
        probabilities = []
        for text, raw in zip(texts, raw_texts):
            X = self._transform_input(text, raw)
            pred = self.model.predict(X)[0]
            prob = self.model.predict_proba(X)[0]
            predictions.append(pred)
            probabilities.append(prob)

        predicted_labels = self.label_encoder.inverse_transform(predictions)
        return predicted_labels, probabilities

    def predict_category(
        self,
        text: str,
        raw_text: Optional[str] = None,
    ) -> Tuple[str, float]:
        labels, probs = self.predict([text], [raw_text or text])
        return labels[0], float(probs[0].max())

    def save(self, model_dir: str = 'models'):
        if not self.is_trained:
            raise ValueError("No trained model to save")

        Path(model_dir).mkdir(parents=True, exist_ok=True)

        model_path = Path(model_dir) / f'{self.model_type}_model.pkl'
        vectorizer_path = Path(model_dir) / 'vectorizer.pkl'
        encoder_path = Path(model_dir) / 'label_encoder.pkl'
        feature_extractor_path = Path(model_dir) / 'feature_extractor.pkl'

        joblib.dump(self.model, model_path)
        joblib.dump(self.label_encoder, encoder_path)
        if self.feature_backend == 'tfidf':
            joblib.dump(self.vectorizer, vectorizer_path)
        if self.use_structured_features:
            joblib.dump(self.feature_extractor, feature_extractor_path)

        config_path = save_model_config(
            model_dir,
            categorization_backend=self.feature_backend,
            categorization_model=self.categorization_model_id,
        )

        print(f"Model saved to {model_path}")
        print(f"Label encoder saved to {encoder_path}")
        print(f"Config saved to {config_path} (backend={self.feature_backend})")
        if self.feature_backend == 'tfidf':
            print(f"Vectorizer saved to {vectorizer_path}")
        if self.use_structured_features:
            print(f"Feature extractor saved to {feature_extractor_path}")

        result = {
            'model': str(model_path),
            'label_encoder': str(encoder_path),
            'model_config': str(config_path),
        }
        if self.feature_backend == 'tfidf':
            result['vectorizer'] = str(vectorizer_path)
        if self.use_structured_features:
            result['feature_extractor'] = str(feature_extractor_path)
        return result

    def load(self, model_dir: str = 'models', model_type: str = None):
        model_type = model_type or self.model_type

        model_path = Path(model_dir) / f'{model_type}_model.pkl'
        vectorizer_path = Path(model_dir) / 'vectorizer.pkl'
        encoder_path = Path(model_dir) / 'label_encoder.pkl'
        feature_extractor_path = Path(model_dir) / 'feature_extractor.pkl'

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        config = load_model_config(model_dir)
        self.feature_backend = config['categorization']['backend']
        self.categorization_model_id = (
            config['categorization'].get('model') or CATEGORIZATION_MODEL_ID
        )

        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        if vectorizer_path.exists() and self.feature_backend == 'tfidf':
            self.vectorizer = joblib.load(vectorizer_path)
        if feature_extractor_path.exists():
            self.feature_extractor = joblib.load(feature_extractor_path)
            self.use_structured_features = True
        else:
            self.use_structured_features = False

        self.is_trained = True
        print(f"Loaded model from {model_path} (backend={self.feature_backend})")


def compare_models(
    df: pd.DataFrame,
    text_column: str = 'cleaned_text',
    label_column: str = 'category',
) -> Dict[str, Dict]:
    models = ['logistic', 'random_forest', 'svm', 'knn']
    results = {}

    for model_type in models:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()} model...")
        print('='*50)

        classifier = ResumeClassifier(
            max_features=8000,
            model_type=model_type,
            feature_backend='minilm',
        )
        metrics = classifier.train(df, text_column, label_column=label_column)
        results[model_type] = metrics

    best_model = max(results.keys(), key=lambda x: results[x]['f1_macro'])
    print(f"\n{'='*50}")
    print(
        f"Best model (F1 macro): {best_model.upper()} "
        f"with accuracy {results[best_model]['accuracy']:.4f}"
    )
    print('='*50)

    return results


if __name__ == "__main__":
    from data_preparation import create_sample_dataset, DataPreparator
    from text_preprocessing import TextPreprocessor

    sample_path = create_sample_dataset(n_samples=500)
    preparator = DataPreparator(sample_path)
    df = preparator.load_data()
    df = preparator.clean_data()
    df = preparator.balance_classes()

    preprocessor = TextPreprocessor(use_lemmatization=True)
    df['cleaned_text'] = df['resume_text'].apply(preprocessor.preprocess)

    classifier = ResumeClassifier(
        max_features=8000,
        model_type='logistic',
        feature_backend='minilm',
    )
    metrics = classifier.train(df)
    classifier.save()

    test_text = "experienced software engineer python java machine learning aws docker"
    prediction, confidence = classifier.predict_category(
        preprocessor.preprocess(test_text), raw_text=test_text
    )
    print(f"\nTest prediction: {prediction} (confidence: {confidence:.2%})")
