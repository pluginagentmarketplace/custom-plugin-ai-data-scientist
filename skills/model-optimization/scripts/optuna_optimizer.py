#!/usr/bin/env python3
"""
Hyperparameter Optimization with Optuna
Automated ML model tuning framework
"""

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from typing import Dict, Any, Optional, Callable
import logging
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptunaOptimizer:
    """Hyperparameter optimization using Optuna."""

    def __init__(
        self,
        model_class: type,
        search_space: Dict[str, Dict],
        metric: str = 'f1',
        direction: str = 'maximize',
        cv: int = 5,
        n_jobs: int = -1
    ):
        self.model_class = model_class
        self.search_space = search_space
        self.metric = metric
        self.direction = direction
        self.cv = cv
        self.n_jobs = n_jobs
        self.study = None
        self.best_model = None

    def _sample_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample hyperparameters from search space."""
        params = {}

        for param_name, config in self.search_space.items():
            param_type = config['type']

            if param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    config['low'],
                    config['high'],
                    step=config.get('step', 1)
                )
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    config['low'],
                    config['high'],
                    log=config.get('log', False)
                )
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    config['choices']
                )

        return params

    def _objective(self, trial: optuna.Trial, X, y) -> float:
        """Objective function for optimization."""
        params = self._sample_params(trial)

        try:
            model = self.model_class(**params)

            # Define scoring based on metric
            if self.metric == 'f1':
                scorer = make_scorer(f1_score, average='weighted')
            elif self.metric == 'accuracy':
                scorer = 'accuracy'
            elif self.metric == 'roc_auc':
                scorer = 'roc_auc'
            else:
                scorer = self.metric

            scores = cross_val_score(
                model, X, y,
                cv=self.cv,
                scoring=scorer,
                n_jobs=self.n_jobs
            )

            return scores.mean()

        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return float('-inf') if self.direction == 'maximize' else float('inf')

    def optimize(
        self,
        X,
        y,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        callbacks: Optional[list] = None
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization."""

        logger.info(f"Starting optimization with {n_trials} trials...")

        # Create study
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=TPESampler(seed=42),
            pruner=HyperbandPruner()
        )

        # Run optimization
        self.study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=n_trials,
            timeout=timeout,
            callbacks=callbacks,
            show_progress_bar=True
        )

        # Get best parameters
        best_params = self.study.best_params
        best_score = self.study.best_value

        logger.info(f"Best score: {best_score:.4f}")
        logger.info(f"Best params: {best_params}")

        # Train final model with best params
        self.best_model = self.model_class(**best_params)
        self.best_model.fit(X, y)

        return {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': len(self.study.trials),
            'model': self.best_model
        }

    def get_importance(self) -> Dict[str, float]:
        """Get hyperparameter importance."""
        if self.study is None:
            raise ValueError("Run optimize() first")

        try:
            importance = optuna.importance.get_param_importances(self.study)
            return dict(importance)
        except Exception as e:
            logger.warning(f"Could not compute importance: {e}")
            return {}

    def get_trials_dataframe(self):
        """Get trials as DataFrame."""
        if self.study is None:
            raise ValueError("Run optimize() first")

        return self.study.trials_dataframe()


def create_search_space(model_type: str) -> Dict[str, Dict]:
    """Create search space for common model types."""

    spaces = {
        'random_forest': {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 300, 'step': 50},
            'max_depth': {'type': 'int', 'low': 3, 'high': 15},
            'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
            'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
            'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]}
        },
        'xgboost': {
            'n_estimators': {'type': 'int', 'low': 100, 'high': 500},
            'max_depth': {'type': 'int', 'low': 3, 'high': 10},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
            'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0}
        },
        'lightgbm': {
            'n_estimators': {'type': 'int', 'low': 100, 'high': 500},
            'num_leaves': {'type': 'int', 'low': 20, 'high': 100},
            'max_depth': {'type': 'int', 'low': 3, 'high': 10},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
            'feature_fraction': {'type': 'float', 'low': 0.6, 'high': 1.0}
        }
    }

    return spaces.get(model_type, spaces['random_forest'])


def main():
    """Demo hyperparameter optimization."""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    print("Hyperparameter Optimization Demo")
    print("=" * 50)

    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create optimizer
    search_space = create_search_space('random_forest')
    optimizer = OptunaOptimizer(
        model_class=RandomForestClassifier,
        search_space=search_space,
        metric='f1',
        direction='maximize',
        cv=3
    )

    # Run optimization (fewer trials for demo)
    results = optimizer.optimize(X_train, y_train, n_trials=20)

    print(f"\nBest Parameters: {results['best_params']}")
    print(f"Best CV Score: {results['best_score']:.4f}")

    # Evaluate on test set
    y_pred = results['model'].predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Test F1 Score: {test_f1:.4f}")

    # Parameter importance
    importance = optimizer.get_importance()
    print("\nParameter Importance:")
    for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {param}: {imp:.4f}")

    print("\n[SUCCESS] Optimization complete!")


if __name__ == '__main__':
    main()
