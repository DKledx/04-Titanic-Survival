#!/usr/bin/env python3
"""
Script để test việc load dữ liệu Titanic
"""

import pandas as pd
import numpy as np
import os

def test_data_loading():
    """Test việc load dữ liệu Titanic"""
    print("🧪 TESTING TITANIC DATA LOADING")
    print("=" * 40)
    
    # Kiểm tra file tồn tại
    train_path = 'data/raw/train.csv'
    test_path = 'data/raw/test.csv'
    
    if not os.path.exists(train_path):
        print(f"❌ Không tìm thấy file: {train_path}")
        return False
    
    if not os.path.exists(test_path):
        print(f"❌ Không tìm thấy file: {test_path}")
        return False
    
    print("✅ Các file dữ liệu tồn tại")
    
    # Load dữ liệu
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print("✅ Load dữ liệu thành công")
    except Exception as e:
        print(f"❌ Lỗi khi load dữ liệu: {e}")
        return False
    
    # Kiểm tra shape
    print(f"\n📊 Training set shape: {train_df.shape}")
    print(f"📊 Test set shape: {test_df.shape}")
    
    # Kiểm tra columns
    expected_columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    train_columns = list(train_df.columns)
    test_columns = list(test_df.columns)
    
    print(f"\n🔍 Training columns: {train_columns}")
    print(f"🔍 Test columns: {test_columns}")
    
    if train_columns == expected_columns:
        print("✅ Training set có đúng columns")
    else:
        print("❌ Training set thiếu hoặc thừa columns")
        return False
    
    expected_test_columns = expected_columns.copy()
    expected_test_columns.remove('Survived')
    
    if test_columns == expected_test_columns:
        print("✅ Test set có đúng columns")
    else:
        print("❌ Test set thiếu hoặc thừa columns")
        return False
    
    # Kiểm tra missing values
    print(f"\n🔍 Missing values trong training set:")
    train_missing = train_df.isnull().sum()
    print(train_missing[train_missing > 0])
    
    print(f"\n🔍 Missing values trong test set:")
    test_missing = test_df.isnull().sum()
    print(test_missing[test_missing > 0])
    
    # Kiểm tra target variable
    print(f"\n🎯 Target variable (Survived) distribution:")
    print(train_df['Survived'].value_counts())
    print(f"Survival rate: {train_df['Survived'].mean():.3f}")
    
    # Kiểm tra dữ liệu cơ bản
    print(f"\n📈 Basic statistics:")
    print("Age range:", train_df['Age'].min(), "-", train_df['Age'].max())
    print("Fare range:", train_df['Fare'].min(), "-", train_df['Fare'].max())
    print("Pclass distribution:", train_df['Pclass'].value_counts().to_dict())
    print("Sex distribution:", train_df['Sex'].value_counts().to_dict())
    
    print("\n✅ TẤT CẢ TESTS PASSED!")
    print("🚀 Dữ liệu sẵn sàng để sử dụng trong notebooks!")
    
    return True

if __name__ == "__main__":
    test_data_loading()
