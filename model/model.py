import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

class Flare:
    def __init__(self, model_name: str = "distilbert-base-uncased", num_classes: int = 2, max_length: int = 512) -> None:
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length

        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        print (f"Using: {self.device}")

        self._load_model()

    def _load_model(self) -> None:
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)

        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels = self.num_classes,
            id2label = {0: "BENIGN", 1: "PHISHING"},
            label2id = {"BENIGN": 0, "PHISHING": 1}
        )

        self.model = self.model.to(self.device)
        self.model.eval()

    def tokenize_input(self, input_text: str) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            input_text,
            padding = True,
            truncation = True,
            max_length = self.max_length,
            return_tensors = 'pt'
        )

        for key, value in inputs.items():
            inputs[key] = value.to(self.device)
        
        return inputs

    def predict(self, predict_text: str) -> Dict[str. Any]:
        inputs = self.tokenize_input(predict_text)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        legitimate_probability: float = probabilities[0][0].item()
        phishing_probability: float = probabilities[0][1].item()
        if phishing_probability > legitimate_probability:
            predicted_class: str = "phishing"
        else: 
            predicted_clas: str = "legitamate"
        return {
            'legitimate': legitimate_prob,
            'phishing': phishing_prob,
            'predicted': predicted_class,
            'confidence': max(legitimate_prob, phishing_prob)
        }

if __name__ == "__main__":
    detector = Flare()

    test = "Congrats you won $100000. Click here to claim your prize"

    result = detector.predict(test)
    print("Prediction:")
    print(f"Legitimate: {result['legitimate']:.4f} ({result['legitimate']*100:.2f}%)")
    print(f"Phishing:   {result['phishing']:.4f} ({result['phishing']*100:.2f}%)")
    print(f"Predicted:  {result['predicted'].upper()}")
    print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
