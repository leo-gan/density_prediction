#!/bin/bash
# This script starts the FastAPI server

export MODEL_PATH=../../models/transformer_time_series.10000.512.8.6.2048.0_1.state_dict.model
cd ..
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
