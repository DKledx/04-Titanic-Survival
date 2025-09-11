"""
Model evaluation utilities for Titanic Survival Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.model_selection import learning_curve, validation_curve


def evaluate_classification_model(y_true, y_pred, y_pred_proba=None, model_name="Model"):
    """
    Comprehensive evaluation of classification model
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities (optional)
        model_name: Name of the model for display
    
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    print(f"\n{'='*50}")
    print(f"📊 {model_name} - Model Evaluation")
    print(f"{'='*50}")
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"🎯 Accuracy:  {accuracy:.4f}")
    print(f"🎯 Precision: {precision:.4f}")
    print(f"🎯 Recall:    {recall:.4f}")
    print(f"🎯 F1-Score:  {f1:.4f}")
    
    # AUC Score
    if y_pred_proba is not None:
        auc = roc_auc_score(y_true, y_pred_proba)
        print(f"🎯 AUC Score: {auc:.4f}")
    else:
        auc = None
    
    # Classification Report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_true, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n🔢 Confusion Matrix:")
    print(cm)
    
    # Store results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_score': auc,
        'confusion_matrix': cm
    }
    
    return results


def plot_confusion_matrix(y_true, y_pred, model_name="Model", figsize=(8, 6)):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Died', 'Survived'],
                yticklabels=['Died', 'Survived'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, model_name="Model", figsize=(8, 6)):
    """Plot ROC curve"""
    if y_pred_proba is None:
        print("❌ Cannot plot ROC curve: No probability predictions provided")
        return
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


def plot_precision_recall_curve(y_true, y_pred_proba, model_name="Model", figsize=(8, 6)):
    """Plot Precision-Recall curve"""
    if y_pred_proba is None:
        print("❌ Cannot plot PR curve: No probability predictions provided")
        return
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, color='darkorange', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} - Precision-Recall Curve')
    plt.show()


def plot_feature_importance(model, feature_names, model_name="Model", top_n=15, figsize=(10, 8)):
    """Plot feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print("❌ Model does not have feature importance or coefficients")
        return
    
    # Create feature importance dataframe
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=figsize)
    sns.barplot(data=feature_importance_df, x='importance', y='feature')
    plt.title(f'{model_name} - Feature Importance (Top {top_n})')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()
    
    return feature_importance_df


def compare_models(results_dict, metric='accuracy', figsize=(12, 6)):
    """Compare multiple models"""
    model_names = list(results_dict.keys())
    metrics = [results_dict[name][metric] for name in model_names]
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Model': model_names,
        metric.title(): metrics
    }).sort_values(metric.title(), ascending=False)
    
    # Plot comparison
    plt.figure(figsize=figsize)
    sns.barplot(data=comparison_df, x=metric.title(), y='Model')
    plt.title(f'Model Comparison - {metric.title()}')
    plt.xlabel(metric.title())
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Model Comparison - {metric.title()}:")
    print("=" * 40)
    for _, row in comparison_df.iterrows():
        print(f"{row['Model']:20}: {row[metric.title()]:.4f}")
    
    return comparison_df


def plot_learning_curve(model, X, y, model_name="Model", cv=5, figsize=(10, 6)):
    """Plot learning curve"""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=figsize)
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title(f'{model_name} - Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_validation_curve(model, X, y, param_name, param_range, model_name="Model", cv=5, figsize=(10, 6)):
    """Plot validation curve"""
    train_scores, val_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range, cv=cv
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=figsize)
    plt.plot(param_range, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(param_range, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.title(f'{model_name} - Validation Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


def generate_evaluation_report(results_dict, save_path=None):
    """Generate comprehensive evaluation report"""
    report = []
    report.append("🚢 TITANIC SURVIVAL PREDICTION - MODEL EVALUATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Model comparison
    report.append("📊 MODEL COMPARISON:")
    report.append("-" * 30)
    
    comparison_data = []
    for model_name, results in results_dict.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': results.get('accuracy', 0),
            'Precision': results.get('precision', 0),
            'Recall': results.get('recall', 0),
            'F1-Score': results.get('f1_score', 0),
            'AUC': results.get('auc_score', 0)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    
    report.append(comparison_df.to_string(index=False))
    report.append("")
    
    # Best model
    best_model = comparison_df.iloc[0]
    report.append(f"🏆 BEST MODEL: {best_model['Model']}")
    report.append(f"   Accuracy: {best_model['Accuracy']:.4f}")
    report.append(f"   F1-Score: {best_model['F1-Score']:.4f}")
    report.append(f"   AUC: {best_model['AUC']:.4f}")
    report.append("")
    
    # Recommendations
    report.append("💡 RECOMMENDATIONS:")
    report.append("-" * 20)
    report.append("1. Use the best performing model for final predictions")
    report.append("2. Consider ensemble methods for improved performance")
    report.append("3. Perform hyperparameter tuning on top models")
    report.append("4. Validate results with cross-validation")
    
    # Join report
    report_text = "\n".join(report)
    
    # Save report
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"📄 Evaluation report saved to: {save_path}")
    
    return report_text
