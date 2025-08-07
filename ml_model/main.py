from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from typing import List

# initialize fastapi app
app = FastAPI(
    title = "House price Prediction API",
    description = "API for predicting house prices using a trained linear regression model",
    version = "1.0.0"
)

try:
    with open('linear_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)    

    print("Model and scaler loaded successfully!")

except FileNotFoundError as e:
    print(f"Error loading model or scaler: {e}")
    print("Please ensure you hace the training notebook and save the model . ")

class HouseFeatures(BaseModel):
   MedInc: float
   HouseAge: float
   AveRooms: float
   AveBedrms: float
   Population: float
   AveOccup: float
   Latitude: float
   Longitude: float    

# define the output model
class PredictionResponse(BaseModel):
    predicted_price: float
    input_feature : dict   

@app.get("/", response_class= HTMLResponse)
async def root():
    """Root endpoint with a user_friendly html page"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Centered Page</title>
        <style>
            body {
                margin: 0;
                padding: 0;
                height: 100vh;
                display: flex;
                justify-content: center;   /* Horizontally center */
                align-items: center;       /* Vertically center */
                background-color: #f0f0f0;
                font-family: Arial, sans-serif;
            }
    
            .container {
                background-color: white;
                padding: 40px;
                border-radius: 12px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
                text-align: center;
            }
    
            h1 {
                color: #333;
                margin-bottom: 20px;
            }
    
            p {
                font-size: 18px;
                color: #666;
                margin-bottom: 20px;
            }
    
            a.button {
                display: inline-block;
                text-decoration: none;
                background-color: #007bff;
                color: white;
                padding: 10px 20px;
                border-radius: 8px;
                transition: background-color 0.3s;
            }
    
            a:hover {
                background-color: #0056b3;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Welcome to House Price Prediction API</h1>
            <p>Welcome! This api predicts california house prices using a machine learning modle.
            <b>Students:</b> Click below to explore and test the API endpoints
            </p>
            <a class="button" href="/docs" target="_blank">Open API Documentation</a>
        </div>
    </body>
    </html>
    
     """    
    
@app.post("/predict", response_model=PredictionResponse)
async def predict_house_price(features: HouseFeatures):

    try: 
        # Convert input to numpy array
        input_data = np.array([[
            features.MedInc,
            features.HouseAge,
            features.AveRooms,
            features.AveBedrms,
            features.Population,
            features.AveOccup,
            features.Latitude,
            features.Longitude
        ]])

        input_scaled = scaler.transform(input_data)

        # Make the prediction
        prediction = model.predict(input_scaled)[0]

        # Convert prediction to a more readable format (multiply by 100,000 for actual price)
        predicted_price = float(prediction * 100000)

        return PredictionResponse(
            predicted_price=predicted_price,
            input_feature=features.dict()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
if __name__=="__main__":
    import uvicorn
    # Run the server
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)