# 🚢 Titanic Dataset Setup Guide

## 📊 Dataset Information

Dự án này sử dụng **dataset Titanic thật** từ Kaggle Competition:
- **Nguồn**: [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/data)
- **Training Set**: 891 passengers với target variable (Survived)
- **Test Set**: 418 passengers (không có target variable)
- **Features**: 12 columns trong training set, 11 columns trong test set

## 🔧 Setup Instructions

### 1. Tải Dataset
```bash
# Chạy script tự động tải dataset
python3 download_data_direct.py
```

### 2. Kiểm tra Dataset
```bash
# Test việc load dữ liệu
python3 test_data_loading.py
```

### 3. Chạy EDA với dữ liệu thật
```bash
# Phân tích EDA cơ bản
python3 run_eda_with_real_data.py
```

## 📁 File Structure

```
04-Titanic-Survival/
├── data/
│   └── raw/
│       ├── train.csv          # Training data (891 rows)
│       ├── test.csv           # Test data (418 rows)
│       └── gender_submission.csv  # Sample submission
├── notebooks/
│   ├── 01-eda-analysis.ipynb  # EDA với dữ liệu thật
│   ├── 02-feature-engineering.ipynb
│   ├── 03-model-training.ipynb
│   ├── 04-hyperparameter-tuning.ipynb
│   ├── 05-ensemble-methods.ipynb
│   └── 06-submission-preparation.ipynb
├── src/                       # Source code
├── download_data_direct.py    # Script tải dataset
├── test_data_loading.py       # Script test dữ liệu
└── run_eda_with_real_data.py  # Script EDA cơ bản
```

## 📊 Dataset Overview

### Training Set (891 passengers)
- **Survived**: 0 = No, 1 = Yes (target variable)
- **Pclass**: Ticket class (1, 2, 3)
- **Name**: Passenger name
- **Sex**: Gender (male, female)
- **Age**: Age in years (177 missing values)
- **SibSp**: Siblings/spouses aboard
- **Parch**: Parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number (687 missing values)
- **Embarked**: Port of embarkation (C, Q, S) (2 missing values)

### Test Set (418 passengers)
- Tương tự training set nhưng **không có cột Survived**
- Có thêm 1 missing value trong Fare

## 🔍 Key Insights từ Real Data

### Survival Patterns
- **Overall survival rate**: 38.4%
- **Female survival rate**: 74.2%
- **Male survival rate**: 18.9%
- **1st class survival rate**: 63.0%
- **2nd class survival rate**: 47.3%
- **3rd class survival rate**: 24.2%

### Missing Values
- **Age**: 177 missing (19.9%) trong training set
- **Cabin**: 687 missing (77.1%) trong training set
- **Embarked**: 2 missing (0.2%) trong training set

### Age Distribution
- **Mean age**: 29.7 years
- **Age range**: 0.42 - 80 years
- **Median age**: 28 years

### Fare Distribution
- **Mean fare**: $32.20
- **Fare range**: $0 - $512.33
- **Median fare**: $14.45

## 🚀 Next Steps

1. **Chạy EDA Notebook**: `01-eda-analysis.ipynb`
2. **Feature Engineering**: `02-feature-engineering.ipynb`
3. **Model Training**: `03-model-training.ipynb`
4. **Hyperparameter Tuning**: `04-hyperparameter-tuning.ipynb`
5. **Ensemble Methods**: `05-ensemble-methods.ipynb`
6. **Submission**: `06-submission-preparation.ipynb`

## ⚠️ Important Notes

- Dataset đã được tải từ Kaggle và sẵn sàng sử dụng
- Không cần tải lại dataset trừ khi có lỗi
- Tất cả notebooks đã được cập nhật để sử dụng dữ liệu thật
- Scripts test đã được tạo để kiểm tra tính toàn vẹn dữ liệu

## 🎯 Expected Results

Với dữ liệu thật, bạn có thể mong đợi:
- **Accuracy**: 80-85% với models tốt
- **Feature importance**: Sex, Pclass, Age, Fare là quan trọng nhất
- **Missing values**: Cần xử lý Age, Cabin, Embarked
- **Feature engineering**: Title extraction, family features rất quan trọng

---

**Happy Predicting! 🚢**
