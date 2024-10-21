from pathlib import Path

import torch

class TransformerModel:
    def __init__(self, model_path: Path):
         self.model = self.load_model(model_path=model_path)

    def load_model(self, model_path):
        """Load pretrained model or custom model."""
        return torch.load(model_path)

    def predict(self, input_data):
        """Model inference logic. TODO"""
        with torch.no_grad():
            return self.model(input_data)
