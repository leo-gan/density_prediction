import torch

from app.models.transformer_ts import TransformerTS
from app.utils.logger import logger


def stream_data_to_model(model: TransformerTS, file):
    """TODO
    Mocks streaming data from a file and running inference.
    """
    try:
        logger.info(f"Streaming data from file: {file.filename}")
        with open(file.filename, "rb") as f:
            while True:
                chunk = f.read(1024)  # read 1KB chunk at a time
                if not chunk:
                    logger.info(f"Completed streaming file: {file.filename}")
                    break

                input_data = process_chunk(chunk)
                output = model.predict(input_data)
                logger.info(f"Model inference output: {output}")
                # Optionally, store or stream the result here
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")


def process_chunk(chunk):
    """
    Process the streamed chunk into the format required by the model.
    """
    logger.debug(f"Processing chunk of size {len(chunk)} bytes")
    # Mock conversion of the chunk to a tensor
    return torch.tensor([float(i) for i in chunk.split()])
