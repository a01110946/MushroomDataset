# data_transformation.py

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

def split_features(X_train):
    """
    Split the feature matrix into numeric and categorical features.

    Args:
        X_train (pd.DataFrame): The feature matrix of the training set.

    Returns:
        tuple: A tuple containing the numeric and categorical feature names (numeric_features, categorical_features).
    """
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    return numeric_features, categorical_features

def create_preprocessor(numeric_features, categorical_features):
    """
    Create a preprocessor using ColumnTransformer.

    Args:
        numeric_features (list): List of numeric feature names.
        categorical_features (list): List of categorical feature names.

    Returns:
        ColumnTransformer: The preprocessor.
    """
    numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('standard_scaler', StandardScaler()),
    ('min_max_scaler', MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numeric', numeric_transformer, numeric_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    return preprocessor