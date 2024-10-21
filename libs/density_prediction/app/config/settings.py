import os

class Settings:
    MODEL_PATH = os.getenv('MODEL_PATH', 'path/to/default/model.pt')
    STREAM_CHUNK_SIZE = int(os.getenv('STREAM_CHUNK_SIZE', 1024))  # 1KB chunks

settings = Settings()
