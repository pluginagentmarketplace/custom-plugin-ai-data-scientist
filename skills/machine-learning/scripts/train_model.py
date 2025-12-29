#!/usr/bin/env python3
"""
Machine Learning Training Script
Trains and evaluates ML models with cross-validation.
"""

import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler


def load_config(config_path: str = "assets/model_config.yaml") -> dict:
    """Load model configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_data(X, y, config: dict):
    """Split data into train/test sets."""
    return train_test_split(
        X, y,
        test_size=config['data']['test_size'],
        random_state=config['model']['random_state'],
        stratify=y if config['data']['stratify'] else None
    )


def train_model(X_train, y_train, config: dict):
    """Train the ML model based on configuration."""
    model_type = config['model']['name']
    params = config['hyperparameters']
    params['random_state'] = config['model']['random_state']

    if model_type == "random_forest_classifier":
        model = RandomForestClassifier(**params)
    elif model_type == "gradient_boosting_classifier":
        model = GradientBoostingClassifier(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, config: dict) -> dict:
    """Evaluate the trained model."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    results = {
        'classification_report': classification_report(y_test, predictions, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
    }

    if probabilities is not None:
        results['roc_auc'] = roc_auc_score(y_test, probabilities)

    return results


def cross_validate(model, X, y, config: dict) -> dict:
    """Perform cross-validation."""
    cv_config = config['training']['cross_validation']
    if not cv_config['enabled']:
        return {}

    scores = cross_val_score(
        model, X, y,
        cv=cv_config['folds'],
        scoring=cv_config['scoring']
    )

    return {
        'cv_scores': scores.tolist(),
        'cv_mean': float(np.mean(scores)),
        'cv_std': float(np.std(scores))
    }


def save_model(model, config: dict, metrics: dict):
    """Save trained model and metrics."""
    output_dir = Path(config['output']['model_path'])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{config['model']['name']}.joblib"
    joblib.dump(model, model_path)

    print(f"Model saved to: {model_path}")
    return model_path


def main():
    """Main training pipeline."""
    config = load_config()

    # Example usage - replace with actual data loading
    print("Load your data and call train_model()")
    print("Example:")
    print("  X_train, X_test, y_train, y_test = prepare_data(X, y, config)")
    print("  model = train_model(X_train, y_train, config)")
    print("  results = evaluate_model(model, X_test, y_test, config)")


if __name__ == "__main__":
    main()
