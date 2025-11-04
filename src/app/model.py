from transformers import BertTokenizer, BertForSequenceClassification


class TextClassificationModel:
    def __init__(self, tokenizer="s-nlp/russian_toxicity_classifier", model="s-nlp/russian_toxicity_classifier"):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.model = BertForSequenceClassification.from_pretrained(model)

    def predict(self, text):
        batch = self.tokenizer.encode(text, return_tensors='pt')
        logits = self.model(batch).logits
        predicted_class_id = logits.argmax().item()
        
        predicted_label = self.model.config.id2label[predicted_class_id]
        
        return predicted_label
    
    def predict_batch(self, texts):
        predicted_labels = []
        for text in texts:
            predicted_labels.append(self.predict(text))
        
        return predicted_labels
            

if __name__ == "__main__":
    pass