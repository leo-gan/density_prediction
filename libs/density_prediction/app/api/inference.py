from pathlib import Path

from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from app.models.transformer_model import TransformerModel
from app.utils.stream_handler import stream_data_to_model
from app.utils.logger import logger
from typing import List

router = APIRouter()

model_path = Path('../../../models/transformer_time_series.10000.512.8.6.2048.0_1.model')
model = TransformerModel(model_path=model_path)

@router.post("/predict")
async def predict(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    Endpoint to receive multiple input streams and run inference in parallel.
    """
    logger.info(f"Received {len(files)} files for inference.")
    for file in files:
        logger.info(f"  Processing file: {file.filename}")
        background_tasks.add_task(stream_data_to_model, model, file)
    print("Inference tasks started.")
    return {"message": "Inference tasks started."}
