"""
Model definitions and training utilities for Titanic Survival Prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import joblib
import os


class ModelTrainer:
    """Model training and evaluation class"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def initialize_models(self):
        """Initialize all models to train"""
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=self.random_state, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=self.random_state),
            'SVM': SVC(random_state=self.random_state, probability=True),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(random_state=self.random_state),
            'XGBoost': xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=self.random_state, verbose=-1)
        }
        
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train all models and evaluate performance"""
        print("🤖 Training Models...")
        print("=" * 50)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            # Store results
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"✅ {name}: Accuracy = {accuracy:.4f}, AUC = {auc_score:.4f if auc_score else 'N/A'}")
        
        return self.results
    
    def cross_validate_models(self, X, y, cv=5):
        """Perform cross-validation for all models"""
        print("🔄 Cross-Validation Analysis...")
        print("=" * 50)
        
        cv_results = {}
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models.items():
            print(f"Cross-validating {name}...")
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
            
            cv_results[name] = {
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'scores': cv_scores
            }
            
            print(f"✅ {name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_results
    
    def get_best_model(self, metric='accuracy'):
        """Get the best performing model"""
        if metric == 'accuracy':
            best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        elif metric == 'auc':
            best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['auc'] or 0)
        else:
            raise ValueError("Metric must be 'accuracy' or 'auc'")
        
        return best_model_name, self.results[best_model_name]['model']
    
    def save_models(self, save_dir='models/trained_models'):
        """Save all trained models"""
        os.makedirs(save_dir, exist_ok=True)
        
        for name, result in self.results.items():
            model_path = os.path.join(save_dir, f"{name.lower().replace(' ', '_')}.pkl")
            joblib.dump(result['model'], model_path)
            print(f"💾 Saved {name} to {model_path}")
    
    def load_models(self, save_dir='models/trained_models'):
        """Load saved models"""
        for name in self.models.keys():
            model_path = os.path.join(save_dir, f"{name.lower().replace(' ', '_')}.pkl")
            if os.path.exists(model_path):
                self.models[name] = joblib.load(model_path)
                print(f"📂 Loaded {name} from {model_path}")


class EnsembleModel:
    """Ensemble model class"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.ensemble = None
        
    def create_voting_classifier(self, models, voting='soft'):
        """Create voting classifier"""
        self.ensemble = VotingClassifier(
            estimators=list(models.items()),
            voting=voting
        )
        return self.ensemble
    
    def train_ensemble(self, X_train, y_train):
        """Train ensemble model"""
        if self.ensemble is None:
            raise ValueError("Ensemble not created. Call create_voting_classifier first.")
        
        self.ensemble.fit(X_train, y_train)
        return self.ensemble
    
    def predict(self, X):
        """Make predictions with ensemble"""
        if self.ensemble is None:
            raise ValueError("Ensemble not trained.")
        
        return self.ensemble.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.ensemble is None:
            raise ValueError("Ensemble not trained.")
        
        return self.ensemble.predict_proba(X)


def evaluate_model(y_true, y_pred, model_name):
    """Comprehensive model evaluation"""
    print(f"\n=== {model_name} Evaluation ===")
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return accuracy


def create_submission(predictions, passenger_ids, filename='submission.csv'):
    """Create Kaggle submission file"""
    submission_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions
    })
    
    submission_df.to_csv(filename, index=False)
    print(f"📄 Submission file created: {filename}")
    
    return submission_df
