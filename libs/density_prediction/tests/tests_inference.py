import pytest
import asyncio
import logging
from fastapi.testclient import TestClient
from app.main import app  # Update with the correct path to your FastAPI app
from app.utils.logger import logger

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# Create a test client for the FastAPI app
client = TestClient(app)


@pytest.fixture
def sample_data():
    """Fixture to provide sample input data for testing."""
    return {
        "input": [1.0, 2.0, 3.0, 4.0, 5.0]  # Example input data
    }


def test_inference_endpoint(sample_data):
    """Test the inference endpoint of the FastAPI app."""
    logger.info("Testing inference endpoint with sample data: %s", sample_data)

    response = client.post("/inference/predict", json=sample_data)  # Update the endpoint as necessary
    assert response.status_code == 200
    assert "predictions" in response.json()  # Check for expected key in response
    assert len(response.json()["predictions"]) > 0  # Ensure there are predictions
    logger.info("  SUCCESS: Inference endpoint returned successful response: %s", response.json())


def test_invalid_input():
    """Test the inference endpoint with invalid input."""
    invalid_data = {"input": "invalid_data"}
    logger.info("Testing inference endpoint with invalid input: %s", invalid_data)

    response = client.post("/inference/predict", json=invalid_data)
    assert response.status_code == 422  # Unprocessable Entity for invalid input
    assert "detail" in response.json()  # Check for error detail in response
    logger.info("  SUCCESS: Inference endpoint returned error response for invalid input: %s", response.json())


@pytest.mark.asyncio
async def test_simultaneous_requests(sample_data):
    """Test the inference endpoint with simultaneous requests."""

    async def make_request(data):
        """Helper function to make a request asynchronously."""
        logger.info("    Making asynchronous request with data: %s", data)
        response = client.post("/inference/predict", json=data)
        assert response.status_code == 200
        assert "predictions" in response.json()  # Check for expected key in response
        assert len(response.json()["predictions"]) > 0  # Ensure there are predictions
        logger.info("      SUCCESS: Asynchronous request completed with response: %s", response.json())

    # Prepare a list of tasks to send simultaneous requests
    tasks = [make_request(sample_data) for _ in range(10)]  # Adjust the number of requests as needed

    # Execute all tasks concurrently
    await asyncio.gather(*tasks)
    logger.info("  SUCCESS: Completed all simultaneous requests.")
