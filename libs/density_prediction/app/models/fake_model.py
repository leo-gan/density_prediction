from pathlib import Path

import torch


class FakeModel:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.name = model_path.stem
        self.model = "FakeModel"  # self.load_model(model_path=model_path) # TODO

    def load_model(self, model_path):
        """Load pretrained model or custom model."""
        return torch.load(model_path)

    def predict(self, input_data):
        """Model inference logic."""
        response = input_data[-1]
        return response
