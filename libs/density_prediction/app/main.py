from fastapi import FastAPI
from app.api.inference import router as inference_router

app = FastAPI(
    title="Transformer Inference API",
    description="API to perform inference on a Transformer model with large data streaming",
    version="1.0.0",
)

app.include_router(inference_router, prefix="/inference")
