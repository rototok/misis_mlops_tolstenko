import pytest
from fastapi import FastAPI
from src.app.model import TextClassificationModel
from src.app.schemas import Text, Texts, ModelPredictResponse, ModelPredictBatchResponse


app = FastAPI(
    title="Lab 1",
    description="API для классификации текстов",
    version="0.1"
)

model = TextClassificationModel()


@app.get('/health')
async def health():
    if model is not None:
        status = 'ok'
    else:
        status = 'model is not loaded'

    return {'status': status}


@app.post('/predict')
async def predict(input_text: Text):
    return model.predict(input_text.text)


@app.post('/predict_batch')
async def predict_batch(input_texts: Texts):
    return model.predict_batch(input_texts.texts)


@app.get('/model_info')
async def model_info():
    return {
        'model name': 'russian_toxicity_classifier ',
        'parameters': "0.2B",
        'tensor type': 'F32'
        }