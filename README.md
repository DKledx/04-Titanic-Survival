# 🚢 Titanic Survival Prediction

## 🎯 Mục Tiêu Dự Án

Dự án **Titanic Survival Prediction** là bài toán classification kinh điển trên Kaggle. Bạn sẽ dự đoán hành khách nào có khả năng sống sót sau thảm họa Titanic dựa trên thông tin cá nhân, vé, và cabin. Đây là dự án hoàn hảo để học feature engineering và classification techniques.

## 🎓 Kiến Thức Sẽ Học Được

### 📚 Core ML Concepts
- **Binary Classification**: Phân loại nhị phân
- **Feature Engineering**: Tạo đặc trưng từ dữ liệu thô
- **Data Preprocessing**: Xử lý dữ liệu thiếu, outliers
- **Model Selection**: Chọn mô hình phù hợp
- **Hyperparameter Tuning**: Tối ưu tham số

### 🔧 Techniques & Algorithms
- **Logistic Regression**: Hồi quy logistic
- **Random Forest**: Rừng ngẫu nhiên
- **Gradient Boosting**: XGBoost, LightGBM
- **Support Vector Machine**: SVM
- **Neural Networks**: MLP

### 📊 Data Science Skills
- **Exploratory Data Analysis**: Phân tích khám phá
- **Feature Selection**: Lựa chọn đặc trưng
- **Cross-validation**: Kiểm tra chéo
- **Ensemble Methods**: Phương pháp kết hợp

## 📁 Cấu Trúc Dự Án

```
04-Titanic-Survival/
├── README.md
├── notebooks/
│   ├── 01-eda-analysis.ipynb
│   ├── 02-feature-engineering.ipynb
│   ├── 03-model-training.ipynb
│   ├── 04-hyperparameter-tuning.ipynb
│   ├── 05-ensemble-methods.ipynb
│   └── 06-submission-preparation.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── evaluation.py
│   └── submission.py
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed/
├── models/
│   └── trained_models/
├── submissions/
├── requirements.txt
└── .gitignore
```

## 📊 Dataset Overview

### 🚢 Titanic Dataset
- **Nguồn**: [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- **Training Set**: 891 passengers
- **Test Set**: 418 passengers
- **Target**: Survived (0 = No, 1 = Yes)

### 🔍 Features
- **PassengerId**: Unique identifier
- **Pclass**: Ticket class (1, 2, 3)
- **Name**: Passenger name
- **Sex**: Gender (male, female)
- **Age**: Age in years
- **SibSp**: Siblings/spouses aboard
- **Parch**: Parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (C, Q, S)

## 🚀 Cách Bắt Đầu

### 1. Download Dataset
```bash
# Tạo thư mục data
mkdir -p data/raw

# Download từ Kaggle (cần API key)
kaggle competitions download -c titanic -p data/raw/
unzip data/raw/titanic.zip -d data/raw/
```

### 2. Cài Đặt Môi Trường
```bash
python -m venv titanic_env
source titanic_env/bin/activate
pip install -r requirements.txt
```

### 3. Bắt Đầu EDA
Mở `notebooks/01-eda-analysis.ipynb` để khám phá dữ liệu!

## 📋 Roadmap Học Tập

### ✅ Phase 1: Exploratory Data Analysis
- [ ] Load và khám phá dataset
- [ ] Phân tích missing values
- [ ] Visualization survival patterns
- [ ] Correlation analysis
- [ ] Outlier detection

### ✅ Phase 2: Feature Engineering
- [ ] Extract title from Name
- [ ] Create family size features
- [ ] Age grouping
- [ ] Fare binning
- [ ] Cabin deck extraction
- [ ] One-hot encoding

### ✅ Phase 3: Model Training
- [ ] Train-test split
- [ ] Baseline models
- [ ] Logistic Regression
- [ ] Random Forest
- [ ] Gradient Boosting
- [ ] Neural Networks

### ✅ Phase 4: Model Optimization
- [ ] Cross-validation
- [ ] Hyperparameter tuning
- [ ] Feature selection
- [ ] Model comparison
- [ ] Performance analysis

### ✅ Phase 5: Ensemble & Submission
- [ ] Voting classifiers
- [ ] Stacking methods
- [ ] Final model selection
- [ ] Test set predictions
- [ ] Kaggle submission

## 🔧 Feature Engineering Techniques

### 📝 Title Extraction
```python
def extract_title(name):
    """Extract title from passenger name"""
    title = name.split(',')[1].split('.')[0].strip()
    # Group rare titles
    if title in ['Mr']:
        return 'Mr'
    elif title in ['Miss', 'Mlle']:
        return 'Miss'
    elif title in ['Mrs', 'Mme']:
        return 'Mrs'
    elif title in ['Master']:
        return 'Master'
    else:
        return 'Rare'

df['Title'] = df['Name'].apply(extract_title)
```

### 👨‍👩‍👧‍👦 Family Features
```python
def create_family_features(df):
    """Create family-related features"""
    # Family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Is alone
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Family size groups
    df['FamilySizeGroup'] = pd.cut(df['FamilySize'], 
                                  bins=[0, 1, 4, 7, 20], 
                                  labels=['Alone', 'Small', 'Medium', 'Large'])
    
    return df
```

### 🎂 Age Grouping
```python
def create_age_groups(df):
    """Create age groups"""
    df['AgeGroup'] = pd.cut(df['Age'], 
                           bins=[0, 12, 18, 35, 60, 100], 
                           labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    
    # Fill missing ages with median by title
    for title in df['Title'].unique():
        median_age = df[df['Title'] == title]['Age'].median()
        df.loc[(df['Age'].isna()) & (df['Title'] == title), 'Age'] = median_age
    
    return df
```

## 🤖 Model Implementation

### 📊 Baseline Models
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'XGBoost': XGBClassifier(random_state=42)
}

# Train and evaluate
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name}: {accuracy:.4f}")
```

### 🎯 Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

# Random Forest tuning
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)
print(f"Best RF params: {rf_grid.best_params_}")
print(f"Best RF score: {rf_grid.best_score_:.4f}")
```

## 📊 Evaluation Metrics

### 🎯 Classification Metrics
```python
from sklearn.metrics import classification_report, confusion_matrix

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
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return accuracy
```

## 🎯 Ensemble Methods

### 🗳️ Voting Classifier
```python
from sklearn.ensemble import VotingClassifier

# Create ensemble
ensemble = VotingClassifier([
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(random_state=42)),
    ('lr', LogisticRegression(random_state=42))
], voting='soft')

# Train ensemble
ensemble.fit(X_train, y_train)
ensemble_pred = ensemble.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
```

### 📚 Stacking
```python
from sklearn.ensemble import StackingClassifier

# Create stacking classifier
stacking = StackingClassifier([
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(random_state=42))
], 
final_estimator=LogisticRegression(random_state=42),
cv=5)

# Train stacking
stacking.fit(X_train, y_train)
stacking_pred = stacking.predict(X_test)
stacking_accuracy = accuracy_score(y_test, stacking_pred)
print(f"Stacking Accuracy: {stacking_accuracy:.4f}")
```

## 🎯 Expected Results

Sau khi hoàn thành dự án này, bạn sẽ có:

1. **High Accuracy Model**: >80% accuracy trên test set
2. **Feature Engineering Skills**: Tạo features từ dữ liệu thô
3. **Model Comparison**: So sánh nhiều algorithms
4. **Ensemble Knowledge**: Kết hợp models hiệu quả
5. **Kaggle Submission**: Submit kết quả lên Kaggle

## 🔍 Key Insights to Discover

### 📊 Data Insights
- **Women và children** có tỷ lệ sống sót cao hơn
- **Higher class passengers** ưu tiên trong rescue
- **Family size** ảnh hưởng đến survival
- **Age** có correlation với survival

### 🎯 Model Insights
- **Feature engineering** quan trọng hơn algorithm choice
- **Ensemble methods** thường cho kết quả tốt hơn
- **Cross-validation** cần thiết để tránh overfitting
- **Hyperparameter tuning** cải thiện performance đáng kể

## 📚 Tài Liệu Tham Khảo

- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [Feature Engineering Guide](https://www.kaggle.com/learn/feature-engineering)
- [Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## 🏆 Next Steps

Sau khi hoàn thành Titanic Survival Prediction, bạn có thể:
- Thử advanced feature engineering
- Chuyển sang dự án 5: Housing Price Predictor
- Phát triển web app với Streamlit
- Tham gia Kaggle competitions khác

## 🎨 Bonus: Streamlit Dashboard

Tạo web app để demo model:
- Input passenger information
- Predict survival probability
- Show feature importance
- Visualize decision path

---

**Happy Predicting! 🚢**

*Hãy bắt đầu với EDA và khám phá những câu chuyện ẩn sau dữ liệu Titanic!*
