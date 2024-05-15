# src/main.py
"""
This module provides the FastAPI application for the Mushroom Classifier API.
"""

from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from src.utils.config import MODEL_PATH

app = FastAPI()

class InputData(BaseModel):
    """
    Represents the input data for the prediction endpoint.
    """
    cap_diameter: float
    cap_surface: str
    cap_color: str
    gill_attachment: str
    stem_width: float
    stem_root: str
    stem_surface: str
    veil_type: str
    veil_color: str
    has_ring: str
    spore_print_color: str

def load_model(model_path):
    """
    Loads the trained model from the specified path.
    
    Args:
        model_path (str): The path to the trained model file.
        
    Returns:
        The loaded model object.
        
    Raises:
        FileNotFoundError: If the model file is not found.
        ValueError: If the loaded model does not have the required 'predict' method.
    """
    try:
        model = joblib.load(model_path)
        if not hasattr(model, "predict"):
            raise ValueError("The model does not have the required 'predict' method.")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model file not found: {model_path}") from e
    return model

@app.post("/predict")
def predict(input_data: InputData):
    """
    Endpoint for making predictions based on the input data.
    
    Args:
        input_data (InputData): The input data for the mushroom classification.
        
    Returns:
        A dictionary containing the predicted class.
    """
    try:
        # Convert the input data to a DataFrame
        new_data = pd.DataFrame([{
            'cap-diameter': input_data.cap_diameter,
            'stem-width': input_data.stem_width,
            'cap-surface': input_data.cap_surface,
            'cap-color': input_data.cap_color,
            'gill-attachment': input_data.gill_attachment,
            'stem-root': input_data.stem_root,
            'stem-surface': input_data.stem_surface,
            'veil-type': input_data.veil_type,
            'veil-color': input_data.veil_color,
            'has-ring': input_data.has_ring,
            'spore-print-color': input_data.spore_print_color
        }])

        # Load the trained model and preprocessor
        try:
            model = load_model(MODEL_PATH)
        except FileNotFoundError as e:
            error_message = f"Model file not found: {MODEL_PATH}"
            print(f"Error in /predict endpoint: {error_message}")
            raise HTTPException(status_code=500, detail=error_message) from e

        # Make predictions
        predictions = model.predict(new_data)

        # Print the predictions
        if predictions[0] == 'p':
            print("""RESPONSE: For the given mushroom features,
                  our model is predicting it as a poisonous mushroom. Please be careful!""")
        elif predictions[0] == 'e':
            print("""RESPONSE: For the given mushroom features,
                  our model is predicting it as an edible mushroom. Enjoy!""")
        else:
            print("Unexpected prediction value received. Please check your model.")

        # Return the predictions
        return {"predictions": predictions.tolist()}

    except Exception as e:
        print(f"Error in /predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/")
def root():
    """
    Root endpoint of the API. This endpoint can be used to check if the API is running.
    
    Returns:
        A dictionary containing a welcome message.
    """
    return {"message": "Welcome to the Mushroom Classifier API"}
