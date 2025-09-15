#!/usr/bin/env python3
"""
Script để verify toàn bộ setup của dự án Titanic
"""

import os
import pandas as pd
import sys
from pathlib import Path

def verify_setup():
    """Verify toàn bộ setup của dự án"""
    print("🔍 TITANIC PROJECT SETUP VERIFICATION")
    print("=" * 50)
    
    # 1. Kiểm tra cấu trúc thư mục
    print("\n📁 Checking directory structure...")
    required_dirs = [
        'data/raw',
        'notebooks',
        'src',
        'models',
        'reports'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - Missing!")
            return False
    
    # 2. Kiểm tra file dữ liệu
    print("\n📊 Checking data files...")
    data_files = [
        'data/raw/train.csv',
        'data/raw/test.csv'
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing!")
            return False
    
    # 3. Kiểm tra notebooks
    print("\n📓 Checking notebooks...")
    notebook_files = [
        'notebooks/01-eda-analysis.ipynb',
        'notebooks/02-feature-engineering.ipynb',
        'notebooks/03-model-training.ipynb',
        'notebooks/04-hyperparameter-tuning.ipynb',
        'notebooks/05-ensemble-methods.ipynb',
        'notebooks/06-submission-preparation.ipynb'
    ]
    
    for file_path in notebook_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing!")
            return False
    
    # 4. Kiểm tra source code
    print("\n🔧 Checking source code...")
    src_files = [
        'src/__init__.py',
        'src/data_preprocessing.py',
        'src/models.py',
        'src/evaluation.py'
    ]
    
    for file_path in src_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing!")
            return False
    
    # 5. Kiểm tra dữ liệu
    print("\n🧪 Testing data loading...")
    try:
        train_df = pd.read_csv('data/raw/train.csv')
        test_df = pd.read_csv('data/raw/test.csv')
        
        print(f"✅ Training set: {train_df.shape}")
        print(f"✅ Test set: {test_df.shape}")
        
        # Kiểm tra columns
        expected_train_cols = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
        expected_test_cols = expected_train_cols.copy()
        expected_test_cols.remove('Survived')
        
        if list(train_df.columns) == expected_train_cols:
            print("✅ Training set columns correct")
        else:
            print("❌ Training set columns incorrect")
            return False
        
        if list(test_df.columns) == expected_test_cols:
            print("✅ Test set columns correct")
        else:
            print("❌ Test set columns incorrect")
            return False
        
        # Kiểm tra missing values
        train_missing = train_df.isnull().sum()
        test_missing = test_df.isnull().sum()
        
        print(f"✅ Training missing values: {train_missing[train_missing > 0].to_dict()}")
        print(f"✅ Test missing values: {test_missing[test_missing > 0].to_dict()}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
    
    # 6. Kiểm tra scripts
    print("\n📜 Checking utility scripts...")
    script_files = [
        'download_data_direct.py',
        'test_data_loading.py',
        'run_eda_with_real_data.py',
        'verify_setup.py'
    ]
    
    for file_path in script_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing!")
            return False
    
    # 7. Kiểm tra requirements
    print("\n📦 Checking requirements...")
    if os.path.exists('requirements.txt'):
        print("✅ requirements.txt")
    else:
        print("❌ requirements.txt - Missing!")
        return False
    
    # 8. Kiểm tra documentation
    print("\n📚 Checking documentation...")
    doc_files = [
        'README.md',
        'DATA_SETUP.md'
    ]
    
    for file_path in doc_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing!")
            return False
    
    print("\n🎉 SETUP VERIFICATION COMPLETE!")
    print("=" * 50)
    print("✅ All checks passed!")
    print("🚀 Project is ready to use!")
    print("\n📋 Next steps:")
    print("1. Run: jupyter notebook")
    print("2. Open: notebooks/01-eda-analysis.ipynb")
    print("3. Start exploring the real Titanic dataset!")
    
    return True

if __name__ == "__main__":
    success = verify_setup()
    sys.exit(0 if success else 1)
