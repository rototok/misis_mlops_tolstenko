import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TritonPythonModel:
    def initialize(self, args):
        model_name = "s-nlp/russian_toxicity_classifier"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def execute(self, requests):
        responses = []

        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy()
            texts = [text.decode('utf-8') for text in input_tensor.flatten()]

            enc = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )           

            with torch.no_grad():
                outputs = self.model(**enc)
                logits = outputs.logits
                logits_array = logits.cpu().numpy().astype(np.float32)

            output_tensor = pb_utils.Tensor("TOXICITY_LOGITS", logits_array)
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)
        
        return responses

    def finalize(self):
        del self.model
        del self.tokenizer