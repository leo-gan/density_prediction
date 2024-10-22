# Use the official Python 3.10 image from the Docker Hub
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy poetry.lock and pyproject.toml files to install dependencies
COPY poetry.lock pyproject.toml /app/

# Install Poetry
RUN pip install poetry

# Install the large package (e.g., torch) first
RUN poetry add numpy
RUN poetry add torch
RUN poetry add fastapi

# Install dependencies from pyproject.toml
RUN poetry install --no-dev

# Show information about the installed packages
RUN poetry show

# Copy the rest of the application code
COPY . .

# Expose the port that the FastAPI app runs on
EXPOSE 8000

# Command to run the FastAPI app
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
