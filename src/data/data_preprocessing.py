# data_preprocessing.py
"""
This module contains functions for preprocessing the mushroom dataset.
It includes functions for detecting and removing outliers, dropping unwanted features, and checking for required columns.
"""

import pandas as pd

def detect_outliers_iqr(dataframe):
    """
    Detect outliers in the numerical columns of a DataFrame using the IQR method.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the outliers.
    """
    outliers_df = pd.DataFrame(columns=dataframe.columns)
    
    for column in dataframe.select_dtypes(include=['int64', 'float64']).columns:
        Q1 = dataframe[column].quantile(0.25)
        Q3 = dataframe[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_in_column = dataframe[(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)]
        outliers_df = pd.concat([outliers_df, outliers_in_column])
    
    return outliers_df

def delete_outliers(df):
    """
    Delete outliers from the dataset.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with outliers removed.
    """
    outliers_df = detect_outliers_iqr(df)
    outlier_indices = outliers_df.index

    # Efficiently drop outliers using index
    df_without_outliers = df.drop(outlier_indices)
    return df_without_outliers

def drop_unwanted_features(df, features_to_drop):
    """
    Drop unwanted features from the dataset.

    Args:
        df (pd.DataFrame): The input DataFrame.
        features_to_drop (list): List of feature names to drop.

    Returns:
        pd.DataFrame: The DataFrame with unwanted features dropped.
    """
    df = df.drop(columns=features_to_drop, errors='ignore')
    return df

def check_required_columns(df):
    required_columns = [
                        'cap-diameter',
                        'cap-surface',
                        'cap-color',
                        'gill-attachment',
                        'stem-width',
                        'stem-root',
                        'stem-surface',
                        'veil-type',
                        'veil-color',
                        'has-ring',
                        'spore-print-color',
                        'class']
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
