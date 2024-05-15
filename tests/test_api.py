# test_api.py

import sys
import os
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_predict_valid_input():
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
        "spore_print_color": "k"
    }
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    assert "predictions" in response.json()

def test_predict_missing_required_field():
    input_data = {
        "cap_surface": "s",
        "cap_color": "n",
        "gill_attachment": "f",
        "stem_width": 8.3,
        "stem_root": "c",
        "stem_surface": "s",
        "veil_type": "u",
        "veil_color": "w",
        "has_ring": "t",
        "spore_print_color": "k"
    }
    response = client.post("/predict", json=input_data)
    assert response.status_code == 422
    assert "detail" in response.json()

def test_predict_invalid_data_type():
    input_data = {
        "cap_diameter": "invalid",
        "cap_surface": "s",
        "cap_color": "n",
        "gill_attachment": "f",
        "stem_width": 8.3,
        "stem_root": "c",
        "stem_surface": "s",
        "veil_type": "u",
        "veil_color": "w",
        "has_ring": "t",
        "spore_print_color": "k"
    }
    response = client.post("/predict", json=input_data)
    assert response.status_code == 422
    assert "detail" in response.json()

def test_predict_invalid_http_method():
    response = client.get("/predict")
    assert response.status_code == 405
    assert "detail" in response.json()
