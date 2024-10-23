from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, : x.size(1), :]


class TransformerTimeSeries(nn.Module):
    def __init__(
        self, num_features, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048
    ):
        super(TransformerTimeSeries, self).__init__()
        self.input_layer = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(
            d_model, num_features
        )

    def forward(self, src):
        src = self.input_layer(src)
        src = self.pos_encoder(src)
        transformer_output = self.transformer_encoder(src)
        output = self.fc(transformer_output[:, -1, :])  # Use the last time step output
        return output


class TransformerTS:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.name = model_path.stem
        self.model = TransformerTimeSeries(num_features=10000)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, input_data):
        """Model inference logic."""
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)[0]
        response = self.model(input_tensor)
        return response
