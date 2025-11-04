import sys
from src.app.app import app
from fastapi.testclient import TestClient


client = TestClient(app)


def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_predict():
    payload = {'text': 'Привет, я твой друг, я тебя не обижу'}
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    assert response.json()['predicted_label'] == 'neutral'


def test_batch_predict():
    payload = {'texts': ['Привет, я твой друг, я тебя не обижу', 'Привет! Я твой друг! Я тебя не обижу!']}
    response = client.post('/predict_batch', json=payload)
    assert response.status_code == 200
    assert response.json()['predicted_labels'] == ['neutral', 'toxic']


def test_empty_predict_batch():
    payload = {'texts': None}
    response = client.post('/predict', json=payload)
    assert response.status_code == 422


def test_empty_predict():
    payload = {'text': None}
    response = client.post('/predict', json=payload)
    assert response.status_code == 422