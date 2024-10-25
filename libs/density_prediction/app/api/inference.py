from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.config import settings
from app.models import FakeModel, TransformerTS
from app.utils.logger import logger

router = APIRouter()

model_path = Path(settings.MODEL_PATH)

models = {
    "fake": FakeModel(model_path=model_path),
    "tts_v1": TransformerTS(model_path=model_path),
}


class PredictRequest(BaseModel):
    array: list


class PredictResponse(BaseModel):
    array: list


# Predict endpoint with model selection
@router.post("/predict", response_model=PredictResponse)
async def predict(
    request: dict,
    model: str = Query("fake", description="The model to use for prediction"),
    horizon_steps: int = Query(1, description="How many steps to predict"),
):
    try:
        input_array = np.array(request["array"])
    except Exception:
        logger.error(f"Invalid request: {request}")
        raise HTTPException(
            status_code=422,
            detail="Request must contain 'array' key with a list of values.",
        )

    # Validate the shape of the input
    if input_array.shape != settings.REQUEST_SHAPE:
        logger.error(
            f"Invalid request shape: {input_array.shape}. Expected shape is {settings.REQUEST_SHAPE}."
        )
        raise HTTPException(
            status_code=422,
            detail=f"Input array must be of shape {settings.REQUEST_SHAPE}.",
        )

    logger.info(f"Request shape: {input_array.shape} for model: {model}")

    # Choose the model based on the query parameter
    if model not in models:
        logger.error(f"Model '{model}' not found")
        raise HTTPException(status_code=400, detail=f"Model '{model}' is not available")
    # result = models[model].predict(input_data=input_array)
    result = autoregression(models[model], input_array, horizon_steps)
    # take only the new predictions:
    result = result[-horizon_steps:, :, :]

    logger.info(f"Response shape: {result.shape}")
    return {"array": result.tolist()}


def autoregression(model, input_array, number_of_look_ahead_steps):
    current_sequence = np.copy(input_array)  # Start with the initial array
    new_predictions = []
    for _ in range(number_of_look_ahead_steps):
        new_prediction = model.predict(input_data=current_sequence)
        new_predictions.append(new_prediction)
        current_sequence = np.concatenate(
            (current_sequence, new_prediction[np.newaxis, :]), axis=1
        )

    return np.array(new_predictions)


# @router.post("/stream-predict")
# async def stream_predict(
#     background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)
# ):
#     """TODO
#     Endpoint to receive multiple input streams and run inference in parallel.
#     """
#     logger.info(f"Received {len(files)} files for inference.")
#     for file in files:
#         logger.info(f"  Processing file: {file.filename}")
#         background_tasks.add_task(stream_data_to_model, model, file)
#     print("Inference tasks started.")
#     return {"message": "Inference tasks started."}
