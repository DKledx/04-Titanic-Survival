#!/usr/bin/env python3
"""
Script để chạy EDA với dữ liệu Titanic thật
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def run_eda_analysis():
    """Chạy phân tích EDA với dữ liệu thật"""
    print("🚢 TITANIC EDA ANALYSIS WITH REAL DATA")
    print("=" * 50)
    
    # Load dữ liệu
    train_df = pd.read_csv('data/raw/train.csv')
    test_df = pd.read_csv('data/raw/test.csv')
    
    print(f"📊 Training set shape: {train_df.shape}")
    print(f"📊 Test set shape: {test_df.shape}")
    
    # Basic info
    print("\n📋 Dataset Info:")
    print("Training set info:")
    train_df.info()
    
    print("\nTest set info:")
    test_df.info()
    
    # Missing values analysis
    print("\n🔍 Missing Values Analysis:")
    print("Training set missing values:")
    train_missing = train_df.isnull().sum()
    print(train_missing[train_missing > 0])
    
    print("\nTest set missing values:")
    test_missing = test_df.isnull().sum()
    print(test_missing[test_missing > 0])
    
    # Survival analysis
    print("\n🎯 Survival Analysis:")
    survival_rate = train_df['Survived'].mean()
    print(f"Overall survival rate: {survival_rate:.3f}")
    
    # Survival by gender
    survival_by_sex = train_df.groupby('Sex')['Survived'].agg(['count', 'sum', 'mean'])
    survival_by_sex.columns = ['Total', 'Survived', 'Survival_Rate']
    print("\nSurvival by Gender:")
    print(survival_by_sex)
    
    # Survival by class
    survival_by_class = train_df.groupby('Pclass')['Survived'].agg(['count', 'sum', 'mean'])
    survival_by_class.columns = ['Total', 'Survived', 'Survival_Rate']
    print("\nSurvival by Class:")
    print(survival_by_class)
    
    # Age analysis
    print("\n🎂 Age Analysis:")
    print(f"Age statistics:")
    print(train_df['Age'].describe())
    
    # Fare analysis
    print("\n💰 Fare Analysis:")
    print(f"Fare statistics:")
    print(train_df['Fare'].describe())
    
    # Family size analysis
    train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
    print("\n👨‍👩‍👧‍👦 Family Size Analysis:")
    print(f"Family size statistics:")
    print(train_df['FamilySize'].describe())
    
    # Title extraction
    def extract_title(name):
        title = name.split(',')[1].split('.')[0].strip()
        return title
    
    train_df['Title'] = train_df['Name'].apply(extract_title)
    print("\n📝 Title Analysis:")
    print("Title distribution:")
    print(train_df['Title'].value_counts())
    
    print("\n✅ EDA Analysis Complete!")
    print("🚀 Dữ liệu sẵn sàng cho feature engineering và model training!")

if __name__ == "__main__":
    run_eda_analysis()
