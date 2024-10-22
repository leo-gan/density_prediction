# This script starts the FastAPI server
cd ..
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
