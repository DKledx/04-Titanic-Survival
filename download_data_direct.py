#!/usr/bin/env python3
"""
Script để tải dataset Titanic trực tiếp từ Kaggle API
"""

import os
import zipfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

def download_titanic_data():
    """Tải dataset Titanic từ Kaggle"""
    print("🚢 Đang tải dataset Titanic từ Kaggle...")
    
    # Tạo thư mục data/raw nếu chưa tồn tại
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Khởi tạo Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        # Tải dataset
        print("📥 Đang tải file...")
        api.competition_download_files('titanic', path=str(data_dir))
        
        # Giải nén file
        zip_file = data_dir / "titanic.zip"
        if zip_file.exists():
            print("📦 Đang giải nén file...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            # Xóa file zip
            zip_file.unlink()
            print("✅ File đã được giải nén!")
        
        print("✅ Dataset đã được tải thành công!")
        
        # Kiểm tra các file đã tải
        train_file = data_dir / "train.csv"
        test_file = data_dir / "test.csv"
        
        if train_file.exists() and test_file.exists():
            print(f"✅ Training set: {train_file}")
            print(f"✅ Test set: {test_file}")
            
            # Hiển thị thông tin cơ bản
            import pandas as pd
            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file)
            
            print(f"\n📊 Thông tin dataset:")
            print(f"   Training set: {train_df.shape[0]} hàng, {train_df.shape[1]} cột")
            print(f"   Test set: {test_df.shape[0]} hàng, {test_df.shape[1]} cột")
            print(f"   Features: {list(train_df.columns)}")
            
            return True
        else:
            print("❌ Không tìm thấy file train.csv hoặc test.csv")
            return False
            
    except Exception as e:
        print(f"❌ Lỗi khi tải dataset: {e}")
        print("\n💡 Hướng dẫn thiết lập Kaggle API:")
        print("1. Truy cập https://www.kaggle.com/account")
        print("2. Vào phần 'API' và tạo API token")
        print("3. Tải file kaggle.json")
        print("4. Đặt file vào thư mục ~/.kaggle/")
        print("5. Chạy: chmod 600 ~/.kaggle/kaggle.json")
        return False

def main():
    """Hàm chính"""
    print("🚢 TITANIC DATASET DOWNLOADER")
    print("=" * 40)
    
    if download_titanic_data():
        print("\n🎉 HOÀN THÀNH!")
        print("Dataset Titanic đã được tải và sẵn sàng sử dụng!")
        print("Bạn có thể chạy các notebook để bắt đầu phân tích dữ liệu.")
    else:
        print("\n❌ THẤT BẠI!")
        print("Không thể tải dataset. Vui lòng kiểm tra lại Kaggle API credentials.")

if __name__ == "__main__":
    main()
