# inference.py

import joblib
from data_preprocessing import preprocess_data
from config import MODEL_PATH

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def predict(new_data, model_path):
    model = load_model(model_path)
    preprocessed_data = preprocess_data(new_data)
    predictions = model.predict(preprocessed_data)
    return predictions

if __name__ == "__main__":
    new_data = ...  # Load or provide new data for prediction
    predictions = predict(new_data, MODEL_PATH)
    print("Predictions:", predictions)