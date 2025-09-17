#!/usr/bin/env python3
"""
Test script to validate all notebooks in the Titanic Survival project
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def test_notebook(notebook_path):
    """Test a single notebook by running it"""
    print(f"🧪 Testing {notebook_path}...")
    
    try:
        # Run notebook with nbconvert
        result = subprocess.run([
            'jupyter', 'nbconvert', '--to', 'notebook', '--execute',
            '--ExecutePreprocessor.timeout=300',
            '--output-dir=/tmp',
            notebook_path
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print(f"✅ {notebook_path} - PASSED")
            return True
        else:
            print(f"❌ {notebook_path} - FAILED")
            print(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {notebook_path} - TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ {notebook_path} - ERROR: {str(e)}")
        return False

def check_file_structure():
    """Check if all required files exist"""
    print("📁 Checking file structure...")
    
    required_files = [
        'notebooks/01-eda-analysis.ipynb',
        'notebooks/02-feature-engineering.ipynb',
        'notebooks/03-model-training.ipynb',
        'notebooks/04-hyperparameter-tuning.ipynb',
        'notebooks/05-ensemble-methods.ipynb',
        'notebooks/06-submission-preparation.ipynb',
        'src/data_preprocessing.py',
        'src/models.py',
        'src/evaluation.py',
        'data/raw/train.csv',
        'data/raw/test.csv',
        'requirements.txt'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing files:")
        for file_path in missing_files:
            print(f"   • {file_path}")
        return False
    else:
        print("✅ All required files exist")
        return True

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn',
        'scikit-learn', 'xgboost', 'lightgbm', 'jupyter'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing packages:")
        for package in missing_packages:
            print(f"   • {package}")
        print("💡 Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("✅ All required packages are installed")
        return True

def test_data_loading():
    """Test if data can be loaded correctly"""
    print("📊 Testing data loading...")
    
    try:
        sys.path.append('src')
        from data_preprocessing import load_data
        
        train_df, test_df = load_data('data/raw/train.csv', 'data/raw/test.csv')
        
        if train_df is not None and test_df is not None:
            print(f"✅ Data loaded successfully")
            print(f"   • Training set: {train_df.shape}")
            print(f"   • Test set: {test_df.shape}")
            return True
        else:
            print("❌ Data loading failed")
            return False
            
    except Exception as e:
        print(f"❌ Data loading error: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🚢 TITANIC SURVIVAL PROJECT - NOTEBOOK TESTING")
    print("=" * 60)
    
    # Change to project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run tests
    tests = [
        ("File Structure", check_file_structure),
        ("Dependencies", check_dependencies),
        ("Data Loading", test_data_loading),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name} Test:")
        results[test_name] = test_func()
    
    # Test notebooks (optional - can be slow)
    print(f"\n📓 Notebook Testing (Optional):")
    print("💡 Note: Notebook testing can be slow and requires Jupyter")
    
    notebook_tests = input("Do you want to test notebooks? (y/n): ").lower().strip()
    
    if notebook_tests == 'y':
        notebook_files = [
            'notebooks/01-eda-analysis.ipynb',
            'notebooks/02-feature-engineering.ipynb',
            'notebooks/03-model-training.ipynb',
            'notebooks/04-hyperparameter-tuning.ipynb',
            'notebooks/05-ensemble-methods.ipynb',
            'notebooks/06-submission-preparation.ipynb'
        ]
        
        notebook_results = []
        for notebook in notebook_files:
            if os.path.exists(notebook):
                notebook_results.append(test_notebook(notebook))
            else:
                print(f"⚠️ {notebook} not found")
                notebook_results.append(False)
        
        results['Notebooks'] = all(notebook_results)
    
    # Summary
    print(f"\n📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Project is ready to use.")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
    
    # Save results
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📄 Test results saved to test_results.json")

if __name__ == "__main__":
    main()
