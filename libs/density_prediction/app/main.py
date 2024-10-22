from fastapi import FastAPI
from app.api.inference import router as inference_router
from app.utils.logger import logger

app = FastAPI(
    title="Density Prediction API",
    description="A FastAPI-based inference API with data streaming. It predicts the thermosphere density (space weather science).",
    version="1.0.0",
)

app.include_router(inference_router, prefix="/inference")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Transformer Inference API")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Transformer Inference API")