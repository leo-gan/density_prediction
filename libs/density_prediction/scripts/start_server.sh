#!/bin/bash

# This script starts the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload