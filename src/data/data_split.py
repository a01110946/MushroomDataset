# data_split.py
"""
This module provides a function for splitting the dataset into train, validation,
and test sets, which can be used for training and evaluating machine learning models.
"""

from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.2, val_size=0.2, stratify=None, random_state=42):
    """
    Split the data into training, validation, and testing sets.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable.
        test_size (float): The proportion of the dataset to include in the test split.
        val_size (float): The proportion of the remaining data to include in the validation split.
        stratify (array-like): The target variable used for stratified splitting.
        random_state (int): The random seed for reproducibility.

    Returns:
        tuple: A tuple containing the split data (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=stratify, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, stratify=stratify, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test