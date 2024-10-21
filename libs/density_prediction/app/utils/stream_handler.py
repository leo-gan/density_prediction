import os
import torch
from app.models.transformer_model import TransformerModel

def stream_data_to_model(model: TransformerModel, file):
    """
    Mocks streaming data from a file and running inference.
    """
    # Simulate reading chunks from a large file
    with open(file.filename, 'rb') as f:
        while True:
            chunk = f.read(1024)  # read 1KB chunk at a time
            if not chunk:
                break

            # Assuming the model expects the chunk in a specific format
            input_data = process_chunk(chunk)
            output = model.predict(input_data)
            # You can save or stream the output as needed

def process_chunk(chunk):
    """
    Process the streamed chunk into the format required by the model.
    """
    # Mock conversion of the chunk to a tensor
    return torch.tensor([float(i) for i in chunk.split()])
