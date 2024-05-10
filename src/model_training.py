# model_training.py

import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
import mlflow

def create_training_pipeline(preprocessor):
    """
    Create the training pipeline.

    Args:
        preprocessor (ColumnTransformer): The preprocessor.

    Returns:
        Pipeline: The training pipeline.
    """
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier())
    ])
    return pipeline

def train_model(pipeline, X_train, y_train):
    """
    Train the model using the training pipeline.

    Args:
        pipeline (Pipeline): The training pipeline.
        X_train (pd.DataFrame): The feature matrix of the training set.
        y_train (pd.Series): The target variable of the training set.

    Returns:
        Pipeline: The trained pipeline.
    """
    pipeline.fit(X_train, y_train)
    return pipeline

def save_model(pipeline, preprocessor, model_path):
    """
    Serialize and save the trained pipeline and the preprocessor to a file.

    Args:
        pipeline (Pipeline): The trained pipeline.
        preprocessor (ColumnTransformer): The preprocessor.
        model_path (str): The path to save the model.
    """
    joblib.dump(pipeline, model_path)
    joblib.dump(preprocessor, f"{model_path}_preprocessor")