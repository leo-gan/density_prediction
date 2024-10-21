from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from app.models.transformer_model import TransformerModel
from app.utils.stream_handler import stream_data_to_model
from typing import List

router = APIRouter()

model = TransformerModel()

@router.post("/predict")
async def predict(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    Endpoint to receive multiple input streams and run inference in parallel.
    """
    for file in files:
        background_tasks.add_task(stream_data_to_model, model, file)

    return {"message": "Inference tasks started."}
