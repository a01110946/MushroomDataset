# tests/test_preprocessing_prediction.py

import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_input_data_format_mismatch():
    """
    Test the /predict endpoint with input data that doesn't match the expected format.
    """
    # Test with missing columns
    input_data = {
        "cap_diameter": 5.2,
        "cap_surface": "s",
        "cap_color": "n",
        "gill_attachment": "f",
        "stem_width": 8.3,
        "stem_root": "c",
        "stem_surface": "s",
        "veil_type": "u",
        "veil_color": "w",
        "has_ring": "t"
    }
    response = client.post("/predict", json=input_data)
    assert response.status_code == 422
    assert "detail" in response.json()
    assert any("field required" in str(detail).lower() for detail in response.json()["detail"])

    # Test with extra columns
    input_data = {
        "cap_diameter": 5.2,
        "cap_surface": "s",
        "cap_color": "n",
        "gill_attachment": "f",
        "stem_width": 8.3,
        "stem_root": "c",
        "stem_surface": "s",
        "veil_type": "u",
        "veil_color": "w",
        "has_ring": "t",
        "spore_print_color": "k",
        "extra_column": "value"
    }
    response = client.post("/predict", json=input_data)
    assert response.status_code == 422
    assert "detail" in response.json()
    assert any("extra fields not permitted" in str(detail).lower() for detail in response.json()["detail"])

def test_unseen_categorical_values():
    """
    Test the /predict endpoint with input data containing unseen categorical values.
    """
    input_data = {
        "cap_diameter": 5.2,
        "cap_surface": "new_surface",
        "cap_color": "new_color",
        "gill_attachment": "new_attachment",
        "stem_width": 8.3,
        "stem_root": "new_root",
        "stem_surface": "new_surface",
        "veil_type": "new_type",
        "veil_color": "new_color",
        "has_ring": "new_ring",
        "spore_print_color": "new_color"
    }
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    assert "predictions" in response.json()

def test_large_input_data():
    """
    Test the /predict endpoint with extremely large input data.
    """
    # Create a large input data payload
    large_input_data = {
        "cap_diameter": [5.2] * 1000,
        "cap_surface": ["s"] * 1000,
        "cap_color": ["n"] * 1000,
        "gill_attachment": ["f"] * 1000,
        "stem_width": [8.3] * 1000,
        "stem_root": ["c"] * 1000,
        "stem_surface": ["s"] * 1000,
        "veil_type": ["u"] * 1000,
        "veil_color": ["w"] * 1000,
        "has_ring": ["t"] * 1000,
        "spore_print_color": ["k"] * 1000
    }
    response = client.post("/predict", json=large_input_data)
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert len(response.json()["predictions"]) == 1000