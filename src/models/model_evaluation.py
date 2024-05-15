# model_evaluation.py
"""
This module contains functions for evaluating the trained model on the test set
and saving the evaluation metrics to a file.
"""
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from src.utils.config import METRICS_PATH
import json

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the testing data.

    Args:
        model (object): The trained model object.
        X_test (pd.DataFrame): The testing features.
        y_test (pd.Series): The testing labels.

    Returns:
        dict: A dictionary containing the evaluation metrics.
    """
    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='e', average='binary')
    recall = recall_score(y_test, y_pred, pos_label='e', average='binary')
    f1 = f1_score(y_test, y_pred, pos_label='e', average='binary')

    # Create a dictionary to store the evaluation metrics
    eval_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    # Save evaluation metrics to a JSON file
    with open(METRICS_PATH, 'w') as f:
        json.dump(eval_metrics, f)

    return eval_metrics