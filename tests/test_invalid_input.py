# tests/test_invalid_input.py

from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_missing_required_fields():
    """Test the /predict endpoint with missing required fields."""
    required_fields = [
        "cap_diameter", "cap_surface", "cap_color", "gill_attachment",
        "stem_width", "stem_root", "stem_surface", "veil_type",
        "veil_color", "has_ring", "spore_print_color"
    ]

    for field in required_fields:
        input_data = {
            "cap_diameter": 5.2, "cap_surface": "s", "cap_color": "n",
            "gill_attachment": "f", "stem_width": 8.3, "stem_root": "c",
            "stem_surface": "s", "veil_type": "u", "veil_color": "w",
            "has_ring": "t", "spore_print_color": "k"
        }
        input_data.pop(field)

        response = client.post("/predict", json=input_data)
        assert response.status_code == 422
        assert "detail" in response.json()
        assert field in str(response.json()["detail"])

def test_invalid_data_types():
    """Test the /predict endpoint with invalid data types."""
    test_cases = {
        "cap_diameter": ["invalid", True],
        "stem_width": ["invalid", True],
        "cap_surface": [123, 3.14],
        "cap_color": [123, 3.14],
        "gill_attachment": [123, 3.14],
        "stem_root": [123, 3.14],
        "stem_surface": [123, 3.14],
        "veil_type": [123, 3.14],
        "veil_color": [123, 3.14],
        "has_ring": [123, 3.14],
        "spore_print_color": [123, 3.14]
    }

    for field, invalid_values in test_cases.items():
        for value in invalid_values:
            input_data = {
                "cap_diameter": 5.2, "cap_surface": "s", "cap_color": "n",
                "gill_attachment": "f", "stem_width": 8.3, "stem_root": "c",
                "stem_surface": "s", "veil_type": "u", "veil_color": "w",
                "has_ring": "t", "spore_print_color": "k"
            }
            input_data[field] = value

            response = client.post("/predict", json=input_data)
            assert response.status_code == 422
            assert "detail" in response.json()
            assert field in str(response.json()["detail"]).lower()

def test_out_of_range_values():
    """Test the /predict endpoint with out-of-range values."""
    numeric_test_cases = {
        "cap_diameter": [-5.2, 1000.0],
        "stem_width": [-8.3, 2000.0]
    }

    categorical_test_cases = {
        "cap_surface": ["invalid_surface", "xyz"],
        "cap_color": ["invalid_color", "123"],
        "gill_attachment": ["invalid_attachment", "abc"],
        "stem_root": ["invalid_root", "def"],
        "stem_surface": ["invalid_surface", "ghi"],
        "veil_type": ["invalid_type", "jkl"],
        "veil_color": ["invalid_color", "mno"],
        "has_ring": ["invalid_ring", "pqr"],
        "spore_print_color": ["invalid_color", "stu"]
    }

    # Test numeric features with out-of-range values
    for field, out_of_range_values in numeric_test_cases.items():
        for value in out_of_range_values:
            input_data = {
                "cap_diameter": 5.2, "cap_surface": "s", "cap_color": "n",
                "gill_attachment": "f", "stem_width": 8.3, "stem_root": "c",
                "stem_surface": "s", "veil_type": "u", "veil_color": "w",
                "has_ring": "t", "spore_print_color": "k"
            }
            input_data[field] = value

            response = client.post("/predict", json=input_data)
            assert response.status_code == 422
            assert "detail" in response.json()
            assert field in str(response.json()["detail"]).lower()

    # Test categorical features with out-of-range values
    for field, out_of_range_values in categorical_test_cases.items():
        for value in out_of_range_values:
            input_data = {
                "cap_diameter": 5.2, "cap_surface": "s", "cap_color": "n",
                "gill_attachment": "f", "stem_width": 8.3, "stem_root": "c",
                "stem_surface": "s", "veil_type": "u", "veil_color": "w",
                "has_ring": "t", "spore_print_color": "k"
            }
            input_data[field] = value

            response = client.post("/predict", json=input_data)
            assert response.status_code == 422
            assert "detail" in response.json()
            assert field in str(response.json()["detail"]).lower()