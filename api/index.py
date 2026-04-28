from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

model_path = 'titanic_classifier_v3_rf.pkl'
model = joblib.load(model_path)

app = FastAPI()

features = ['Age', 'Sex_female', 'Sex_male', 'Family']

class Passenger(BaseModel):
    age : float
    sex_female : float
    sex_male : float
    family : int

@app.post('/predict')
def getPredict(passenger: Passenger):
    data = passenger.dict()
    input = pd.DataFrame([[
        data['age'],
        data['sex_female'],
        data['sex_male'],
        data['family']
    ]], columns = features)
    prediction = model.predict(input)
    result = int(prediction[0])

    return {
        'prediction': result
    }
