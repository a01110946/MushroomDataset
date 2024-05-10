# model_prediction.py

import joblib
from data_preprocessing import preprocess_data

def load_model(model_path):
    """
    Load the trained model from a file.

    Args:
        model_path (str): The path to the saved model file.

    Returns:
        object: The loaded model object.
    """
    model = joblib.load(model_path)
    return model

def predict(new_data, model_path):
    """
    Make predictions on new data using the trained model.

    Args:
        new_data (pd.DataFrame): The new data for prediction.
        model_path (str): The path to the saved model file.

    Returns:
        array: The predicted labels or probabilities.
    """
    # Load the trained model
    model = load_model(model_path)

    # Preprocess the new data
    X_new = preprocess_data(new_data)

    # Make predictions
    predictions = model.predict(X_new)

    return predictions