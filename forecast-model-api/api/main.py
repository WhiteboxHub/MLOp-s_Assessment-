from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import lightgbm as lgb
import datetime
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load the trained model (update path as necessary)
model = lgb.Booster(model_file=r'C:\Users\AI_ML PC_4\Desktop\Mlops_assigment\model.txt')  # Ensure you've saved your model with .save_model('model.txt')

# Define request body
class PredictionRequest(BaseModel):
    date: str
    store: int
    item: int

# Preprocessing function (adapt if needed)
def preprocess_input(data: PredictionRequest):
    df = pd.DataFrame([{
        'date': pd.to_datetime(data.date),
        'store': data.store,
        'item': data.item
    }])
    
    # Feature engineering example
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df.drop(columns=['date'], inplace=True)

    return df

# /predict endpoint
@app.post("/predict")
def predict_sales(data: PredictionRequest):
    input_df = preprocess_input(data)
    prediction = model.predict(input_df)[0]
    return {"predicted_sales": round(prediction)}

# /status endpoint
@app.get("/status")
def status():
    return {"status": "API is running ✅"}

# Run with uvicorn
if __name__ == "__main__":
   uvicorn.run(app, host="0.0.0.0", port=8000)
