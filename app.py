import joblib
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel, Field

# Initialize FastAPI
app = FastAPI()

# Load the best model
try:
    model = joblib.load('artifacts/best_model.pkl')
except Exception as e:
    raise RuntimeError("Model could not be loaded!") from e

# Input validation
class HousePriceFeatures(BaseModel):
    """House price features

    Args:
        BaseModel (_type_): 
    """
    MSSubClass: float = Field(..., description='The general zoning classification')
    LotFrontage: float = Field(..., description='Linear feet of street connected to property')
    LotArea: float = Field(..., description='Lot size in square feet')
    LotShape: str = Field(..., description='General shape of property')
    LandContour: str = Field(..., description='Flatness of the property')
    Condition1: str = Field(..., description='Proximity to main road or railroad')
    OverallQual: int = Field(..., description='Overall material and finish quality')
    ExterQual: str = Field(..., description='Exterior material quality')
    ExterCond: str = Field(..., description='Present condition of the material on the exterior')
    Foundation: str = Field(..., description='Type of foundation')
    
@app.post("/predict")
def predict_house_price(features: HousePriceFeatures):
    # Ubah data input menjadi format yang diterima oleh model (seperti DataFrame atau array)
    input_data = np.array([[features.MSSubClass, features.LotFrontage, features.LotArea,
                            features.LotShape, features.LandContour, features.Condition1,
                            features.OverallQual, features.ExterQual, features.ExterCond, features.Foundation]])
    
    # Prediksi menggunakan model yang telah diload
    prediction = model.predict(input_data)
    
    return {"predicted_price": prediction[0]}