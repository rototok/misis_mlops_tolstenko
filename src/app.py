# app.py
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class InputData(BaseModel):
    features: list


@app.post("/predict")
def predict(data: InputData):
    x = np.array(data.features)
    pred = int(x[:3].sum() > 1)
    return {"prediction": pred}
