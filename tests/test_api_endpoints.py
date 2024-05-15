# tests/test_api_endpoints.py

import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_invalid_http_methods():
    """
    Test the /predict endpoint with invalid HTTP methods.
    """
    # Test with GET request
    response = client.get("/predict")
    assert response.status_code == 405
    assert "detail" in response.json()
    assert "Method Not Allowed" in response.json()["detail"]

    # Test with PUT request
    response = client.put("/predict")
    assert response.status_code == 405
    assert "detail" in response.json()
    assert "Method Not Allowed" in response.json()["detail"]

    # Test with DELETE request
    response = client.delete("/predict")
    assert response.status_code == 405
    assert "detail" in response.json()
    assert "Method Not Allowed" in response.json()["detail"]

def test_missing_request_payload():
    """
    Test the /predict endpoint with missing request payload.
    """
    response = client.post("/predict")
    assert response.status_code == 422
    assert "detail" in response.json()
    assert any("field required" in str(detail).lower() for detail in response.json()["detail"])

def test_invalid_request_payload():
    """
    Test the /predict endpoint with invalid request payload.
    """
    # Test with missing required fields
    invalid_payload = {
        "cap_surface": "s",
        "cap_color": "n",
        "gill_attachment": "f",
        "stem_root": "c",
        "stem_surface": "s",
        "veil_type": "u",
        "veil_color": "w",
        "has_ring": "t",
        "spore_print_color": "k"
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
    assert "detail" in response.json()
    assert any("field required" in str(detail).lower() for detail in response.json()["detail"])

    # Test with invalid JSON format
    invalid_json = "{'cap_diameter': 5.2, 'cap_surface': 's', 'cap_color': 'n'}"
    response = client.post("/predict", data=invalid_json)
    assert response.status_code == 422
    assert "detail" in response.json()
    assert "json decode error" in response.text.lower()
