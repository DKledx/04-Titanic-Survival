"""
Data preprocessing utilities for Titanic Survival Prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(train_path, test_path):
    """
    Load training and test datasets
    
    Args:
        train_path (str): Path to training data
        test_path (str): Path to test data
    
    Returns:
        tuple: (train_df, test_df)
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"📊 Training set shape: {train_df.shape}")
    print(f"📊 Test set shape: {test_df.shape}")
    
    return train_df, test_df


def extract_title(name):
    """Extract title from passenger name"""
    title = name.split(',')[1].split('.')[0].strip()
    return title


def group_titles(title):
    """Group rare titles into common categories"""
    if title in ['Mr']:
        return 'Mr'
    elif title in ['Miss', 'Mlle']:
        return 'Miss'
    elif title in ['Mrs', 'Mme']:
        return 'Mrs'
    elif title in ['Master']:
        return 'Master'
    elif title in ['Dr', 'Rev', 'Col', 'Major', 'Capt']:
        return 'Officer'
    else:
        return 'Rare'


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


def extract_cabin_features(df):
    """Extract cabin deck and number features"""
    # Extract deck
    df['CabinDeck'] = df['Cabin'].str[0]
    df['CabinDeck'] = df['CabinDeck'].fillna('Unknown')
    
    # Extract cabin number
    df['CabinNumber'] = df['Cabin'].str.extract('(\d+)')
    df['CabinNumber'] = pd.to_numeric(df['CabinNumber'], errors='coerce')
    
    # Has cabin
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    
    return df


def handle_missing_values(df):
    """Handle missing values in the dataset"""
    # Fill missing ages with median by title
    for title in df['TitleGroup'].unique():
        median_age = df[df['TitleGroup'] == title]['Age'].median()
        df.loc[(df['Age'].isna()) & (df['TitleGroup'] == title), 'Age'] = median_age
    
    # Fill remaining missing ages with overall median
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # Fill missing embarked with mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Fill missing fare with median by class
    for pclass in df['Pclass'].unique():
        median_fare = df[df['Pclass'] == pclass]['Fare'].median()
        df.loc[(df['Fare'].isna()) & (df['Pclass'] == pclass), 'Fare'] = median_fare
    
    return df


def create_age_groups(df):
    """Create age groups"""
    df['AgeGroup'] = pd.cut(df['Age'], 
                           bins=[0, 12, 18, 35, 60, 100], 
                           labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    return df


def create_fare_groups(df):
    """Create fare groups"""
    df['FareGroup'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
    return df


def encode_categorical_features(df, categorical_columns):
    """Encode categorical features"""
    le_dict = {}
    
    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le
    
    return df, le_dict


def preprocess_data(train_df, test_df):
    """
    Complete data preprocessing pipeline
    
    Args:
        train_df (pd.DataFrame): Training data
        test_df (pd.DataFrame): Test data
    
    Returns:
        tuple: (processed_train_df, processed_test_df, label_encoders)
    """
    # Combine datasets for consistent preprocessing
    all_data = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    
    # Extract title features
    all_data['Title'] = all_data['Name'].apply(extract_title)
    all_data['TitleGroup'] = all_data['Title'].apply(group_titles)
    
    # Create family features
    all_data = create_family_features(all_data)
    
    # Extract cabin features
    all_data = extract_cabin_features(all_data)
    
    # Handle missing values
    all_data = handle_missing_values(all_data)
    
    # Create age and fare groups
    all_data = create_age_groups(all_data)
    all_data = create_fare_groups(all_data)
    
    # Define categorical columns for encoding
    categorical_columns = ['Sex', 'Embarked', 'TitleGroup', 'FamilySizeGroup', 
                          'CabinDeck', 'AgeGroup', 'FareGroup']
    
    # Encode categorical features
    all_data, label_encoders = encode_categorical_features(all_data, categorical_columns)
    
    # Split back into train and test
    train_size = len(train_df)
    processed_train_df = all_data[:train_size].copy()
    processed_test_df = all_data[train_size:].copy()
    
    print("✅ Data preprocessing completed!")
    print(f"📊 Processed training set shape: {processed_train_df.shape}")
    print(f"📊 Processed test set shape: {processed_test_df.shape}")
    
    return processed_train_df, processed_test_df, label_encoders


def prepare_features(df, feature_columns):
    """
    Prepare features for model training
    
    Args:
        df (pd.DataFrame): Processed dataframe
        feature_columns (list): List of feature columns to use
    
    Returns:
        pd.DataFrame: Features dataframe
    """
    return df[feature_columns].copy()


def get_feature_columns():
    """Get list of feature columns for model training"""
    return [
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
        'TitleGroup', 'FamilySize', 'IsAlone', 'FamilySizeGroup',
        'CabinDeck', 'HasCabin', 'AgeGroup', 'FareGroup'
    ]
