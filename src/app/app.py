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
    predicted_label = model.predict(input_text.text)
    return ModelPredictResponse(predicted_label=predicted_label)


@app.post('/predict_batch')
async def predict_batch(input_texts: Texts):
    predicted_labels = model.predict_batch(input_texts.texts)
    return ModelPredictBatchResponse(predicted_labels=predicted_labels)


@app.get('/model_info')
async def model_info():
    return {
        'model name': 'russian_toxicity_classifier ',
        'parameters': "0.2B",
        'tensor type': 'F32'
        }