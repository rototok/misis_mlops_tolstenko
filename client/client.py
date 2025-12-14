import tritonclient.http as httpclient
import numpy as np
from schemas import ClassificationRequestBatch, ClassificationResponse, ClassificationResponseBatch
from fastapi import FastAPI

app = FastAPI(
    title="Lab 2",
    description="API для классификации текстов c Triton Inference Server",
    version="1.0.0"
)

clinet = httpclient.InferenceServerClient(url="localhost:8000")

@app.post('/predict')
async def predict(input_texts: ClassificationRequestBatch) -> ClassificationResponseBatch:
    input_texts = input_texts.texts
    texts_array = np.array([text.encode('utf-8') for text in input_texts], dtype=np.object_).reshape(-1, 1)

    input = httpclient.InferInput("TEXT", texts_array.shape, "BYTES")
    input.set_data_from_numpy(texts_array)
    inputs = [input]

    outputs = [httpclient.InferRequestedOutput("TOXICITY_LOGITS")]

    response = clinet.infer('toxicity_classifier', inputs=inputs, outputs=outputs)
    classification_results = response.as_numpy("TOXICITY_LOGITS")

    predicted_labels = []
    for result in classification_results:
        predicted_label = "1" if np.argmax(result) == 1 else "0"
        predicted_labels.append(predicted_label)
    
    return ClassificationResponseBatch(predicted_labels=predicted_labels)