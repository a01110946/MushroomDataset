# config.py
"""
This module contains configuration variables and settings for the project.
It includes data paths, model parameters, and output paths
"""

import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

# Data paths
DATA_DIR = 'data'
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed_data')
TRAIN_DATA = os.path.join(PROCESSED_DATA_DIR, 'train.csv')
VAL_DATA = os.path.join(PROCESSED_DATA_DIR, 'val.csv')
TEST_DATA = os.path.join(PROCESSED_DATA_DIR, 'test.csv')

# Model parameters
MODEL_NAME = 'gradient_boosting'

def get_feature_types(df):
    """
    Identify numeric and categorical features in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        tuple: A tuple containing two lists:
            - numeric_features: List of numeric feature names.
            - categorical_features: List of categorical feature names.
    """
    features_to_drop = ['cap-shape', 'does-bruise-or-bleed', 'gill-spacing', 'gill-color', 'stem-height', 
                        'stem-color', 'ring-type', 'habitat', 'season']
    existing_features_to_drop = set(features_to_drop) & set(df.columns)
    df = df.drop(columns=existing_features_to_drop, errors='ignore')
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    return numeric_features, categorical_features

# Define transformers for numeric and categorical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('standard_scaler', StandardScaler()),
    ('min_max_scaler', MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

def create_preprocessor(numeric_features, categorical_features):
    """
    Create a preprocessor using ColumnTransformer.

    Args:
        numeric_features (list): List of numeric feature names.
        categorical_features (list): List of categorical feature names.

    Returns:
        ColumnTransformer: The preprocessor.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)
        ])
    return preprocessor

MODEL_PARAMS = {
    'model': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8, random_state=42)
}

# Output paths
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create the output directory if it doesn't exist

MODEL_REGISTRY_DIR = 'models'
os.makedirs(MODEL_REGISTRY_DIR, exist_ok=True)  # Create the model registry directory if it doesn't exist

MODEL_PATH = os.path.join(MODEL_REGISTRY_DIR, 'model.pkl')
METRICS_PATH = os.path.join(OUTPUT_DIR, 'metrics.json')