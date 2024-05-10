# data_preprocessing.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

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

"""
def preprocess_data(df, numeric_features, categorical_features):
    ""
    Preprocess the data by handling outliers, scaling numeric features, and encoding categorical features.

    Args:
        df (pd.DataFrame): The input DataFrame.
        numeric_features (list): List of numeric feature names.
        categorical_features (list): List of categorical feature names.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    ""
    # Check for required columns
    check_required_columns(df)
    print("After checking required columns:")
    print(df.head())
    print(df.shape)
    
    # Detect and remove outliers
    outliers_df = detect_outliers_iqr(df)
    df = df.drop(outliers_df.index)
    print("After removing outliers:")
    print(df.head())
    print(df.shape)

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
    
    # Create the preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)
        ],
        sparse_threshold=0
    )

    # Fit and transform the data using the preprocessor
    preprocessed_data = preprocessor.fit_transform(df)
    print("After preprocessing:")
    print(preprocessed_data.shape)
    print("Feature names:")
    print(preprocessor.get_feature_names_out())

    # Convert the preprocessed data back to a DataFrame
    preprocessed_df = pd.DataFrame(preprocessed_data, columns=preprocessor.get_feature_names_out())
    print("Preprocessed DataFrame:")
    print(preprocessed_df.head())
    print(preprocessed_df.shape)

    # Export to CSV
    preprocessed_df.to_csv('preprocessed_data.csv', index=False)

    return preprocessed_df
"""
