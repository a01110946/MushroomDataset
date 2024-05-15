from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from src.utils.config import MODEL_PATH
import pandas as pd

app = FastAPI()

class InputData(BaseModel):
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
    model = joblib.load(model_path)
    model_path = f"{model_path}"
    model = joblib.load(model_path)
    return model

@app.post("/predict")
def predict(input_data: InputData):
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
        model = load_model(MODEL_PATH)

        # Make predictions
        predictions = model.predict(new_data)

        # Print the predictions        
        if predictions[0] == 'p':
            print("RESPONSE: For the given mushroom features, our model is predicting it as a poisonous mushroom. Please be careful!")
        elif predictions[0] == 'e':
            print("RESPONSE: For the given mushroom features, our model is predicting it as an edible mushroom. Enjoy!") 
        else:
            print("Unexpected prediction value received. Please check your model.")
        
        # Return the predictions
        return {"predictions": predictions.tolist()}

    except Exception as e:
        print(f"Error in /predict endpoint: {str(e)}")
        raise e

@app.get("/hello")
def hello():
    return {"message": "Hello, World!"}

@app.get("/")
def root():
    return {"message": "Welcome to the Mushroom Classifier API"}