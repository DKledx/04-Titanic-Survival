"""
Titanic Survival Prediction - Source Code Package

This package contains utilities for:
- Data preprocessing and feature engineering
- Model training and evaluation
- Ensemble methods
- Submission preparation
"""

__version__ = "1.0.0"
__author__ = "ML Portfolio"

# Import main modules
from .data_preprocessing import (
    load_data, 
    preprocess_data, 
    prepare_features, 
    get_feature_columns
)

from .models import (
    ModelTrainer, 
    EnsembleModel, 
    evaluate_model, 
    create_submission
)

from .evaluation import (
    evaluate_classification_model,
    plot_confusion_matrix,
    plot_roc_curve,
    compare_models,
    generate_evaluation_report
)

__all__ = [
    'load_data',
    'preprocess_data', 
    'prepare_features',
    'get_feature_columns',
    'ModelTrainer',
    'EnsembleModel',
    'evaluate_model',
    'create_submission',
    'evaluate_classification_model',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'compare_models',
    'generate_evaluation_report'
]
