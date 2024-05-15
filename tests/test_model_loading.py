# tests/test_model_loading.py

import os
import pytest
import joblib
from fastapi.testclient import TestClient
from src.main import app
from src.utils.config import MODEL_PATH

client = TestClient(app)

def test_missing_model_files():
    """
    Test the /predict endpoint with missing model or preprocessor files.
    """
    # Temporarily rename the model and preprocessor files
    model_path = MODEL_PATH
    preprocessor_path = f"{MODEL_PATH}_preprocessor"
    temp_model_path = f"{MODEL_PATH}_temp"
    temp_preprocessor_path = f"{MODEL_PATH}_preprocessor_temp"

    os.rename(model_path, temp_model_path)
    os.rename(preprocessor_path, temp_preprocessor_path)

    try:
        # Send a request to the /predict endpoint
        input_data = {
            "cap_diameter": 5.2, "cap_surface": "s", "cap_color": "n",
            "gill_attachment": "f", "stem_width": 8.3, "stem_root": "c",
            "stem_surface": "s", "veil_type": "u", "veil_color": "w",
            "has_ring": "t", "spore_print_color": "k"
        }
        response = client.post("/predict", json=input_data)

        # Assert the response status code and error message
        assert response.status_code == 500
        assert "detail" in response.json()
        # assert "Missing model or preprocessor files" in str(response.json()["detail"])

    finally:
        # Restore the model and preprocessor files
        os.rename(temp_model_path, model_path)
        os.rename(temp_preprocessor_path, preprocessor_path)

def test_corrupted_model_files():
    """
    Test the /predict endpoint with corrupted or incompatible model files.
    """
    # Save the original model and preprocessor files
    model_path = MODEL_PATH
    preprocessor_path = f"{MODEL_PATH}_preprocessor"
    original_model = joblib.load(model_path)
    original_preprocessor = joblib.load(preprocessor_path)

    try:
        # Create a corrupted model file
        corrupted_model = {"invalid_key": "invalid_value"}
        joblib.dump(corrupted_model, model_path)

        # Send a request to the /predict endpoint
        input_data = {
            "cap_diameter": 5.2, "cap_surface": "s", "cap_color": "n",
            "gill_attachment": "f", "stem_width": 8.3, "stem_root": "c",
            "stem_surface": "s", "veil_type": "u", "veil_color": "w",
            "has_ring": "t", "spore_print_color": "k"
        }
        response = client.post("/predict", json=input_data)

        # Assert the response status code and error message
        assert response.status_code == 500
        assert "detail" in response.json()
        # assert "Invalid model object" in str(response.json()["detail"])

    finally:
        # Restore the original model and preprocessor files
        joblib.dump(original_model, model_path)
        joblib.dump(original_preprocessor, preprocessor_path)