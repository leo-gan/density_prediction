from pathlib import Path


class FakeModel:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.name = model_path.stem
        self.model = "FakeModel"

    def predict(self, input_data):
        """Model inference logic."""
        response = input_data[-1]
        return response
