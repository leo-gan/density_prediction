import os


class Settings:
    # MODEL_PATH = os.getenv("MODEL_PATH", "../../../models/transformer_time_series.10000.512.8.6.2048.0_1.model")
    MODEL_PATH = os.getenv(
        "MODEL_PATH",
        "../../../models/transformer_time_series.10000.512.8.6.2048.0_1.state_dict.model",
    )
    REQUEST_SHAPE = (1, 3, 10000)  # 3 timesteps, 10000 features


settings = Settings()
