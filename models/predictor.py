import torch
from transformers import AutoModelForQuestionAnswering
from models.preprocessing import Preprocessing
import numpy as np
class Predictor():
    def __init__(self):
        self.process = Preprocessing(flag_training=False)
        self.model = self.load_model()

    def load_model(self):
        model = AutoModelForQuestionAnswering.from_pretrained(
            "./squad",
            use_safetensors=True,
        )
        return model

    def predict(self, question, context):
        self.model.eval()
        inputs = self.process.tokenizer(
            question,
            context,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get start and end logits
        start_logits = outputs.start_logits.cpu().numpy()[0]
        end_logits = outputs.end_logits.cpu().numpy()[0]

        # Find the most probable start and end positions
        start_index = np.argmax(start_logits)
        end_index = np.argmax(end_logits)

        # Decode the answer from the tokenized input
        input_ids = inputs['input_ids'][0]
        answer_tokens = input_ids[start_index:end_index + 1]
        answer = self.process.tokenizer.decode(answer_tokens, skip_special_tokens=True)

        return answer



