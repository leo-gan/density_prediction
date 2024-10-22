import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.utils.logger import logger

# Create a test client for the FastAPI app
client = TestClient(app)


TEST_DATA_DIR = Path("data")
INPUT_DATA_FILE = TEST_DATA_DIR / "test_data.3_samples.input.json"
PREDICTED_DATA_FILE = TEST_DATA_DIR / "test_data.prediction.json"


@pytest.fixture
def sample_data():
    with open(INPUT_DATA_FILE, "r") as f:
        array_list = json.load(f)
    return array_list  # np.array(array_list)


@pytest.fixture
def prediction_data():
    with open(PREDICTED_DATA_FILE, "r") as f:
        array_list = json.load(f)
    return np.array(array_list)


def test_inference_endpoint_fake_model(sample_data, prediction_data):
    """Test the inference endpoint of the FastAPI app with FakeModel."""
    logger.info(f"Testing inference endpoint with: {len(sample_data) = }")

    params = {"model": "model_a"}  # Update the query parameter as necessary
    data = {"array": sample_data}
    response = client.post("/inference/predict", params=params, json=data)
    assert response.status_code == 200
    assert "array" in response.json()  # Check for expected key in response
    assert len(response.json()["array"]) > 0  # Ensure there are predictions
    assert np.array_equal(np.array(response.json()["array"]), np.array(sample_data)[-1])
    logger.info(
        f"  SUCCESS: Inference endpoint returned: {len(response.json()['array']) = }"
    )


def test_inference_endpoint_ts_model(sample_data, prediction_data):
    """Test the inference endpoint of the FastAPI app with Tensorflow model."""
    logger.info(f"Testing inference endpoint with: {len(sample_data) = }")

    params = {"model": "model_b"}
    data = {"array": sample_data}
    response = client.post("/inference/predict", params=params, json=data)
    assert response.status_code == 200
    assert "array" in response.json()  # Check for expected key in response
    assert len(response.json()["array"]) > 0  # Ensure there are predictions
    assert np.array(response.json()["array"]).shape == prediction_data.shape
    assert np.allclose(np.array(response.json()["array"]), prediction_data, rtol=1e-03)
    logger.info(
        f"  SUCCESS: Inference endpoint returned: {len(response.json()['array']) = }"
    )


def test_invalid_input():
    """Test the inference endpoint with invalid input."""
    invalid_data = {"input": "invalid_data"}
    logger.info("Testing inference endpoint with invalid input: %s", invalid_data)

    response = client.post("/inference/predict", json=invalid_data)
    assert response.status_code == 422  # Unprocessable Entity for invalid input
    assert "detail" in response.json()  # Check for error detail in response
    logger.info(
        "  SUCCESS: Inference endpoint returned error response for invalid input: %s",
        response.json(),
    )


# @pytest.mark.asyncio
# async def test_simultaneous_requests(sample_data):  # TODO
#     """Test the inference endpoint with simultaneous requests."""
#
#     async def make_request(data):
#         """Helper function to make a request asynchronously."""
#         logger.info("    Making asynchronous request with data: %s", data)
#         response = client.post("/inference/stream-predict", json=data)
#         assert response.status_code == 200
#         assert "predictions" in response.json()  # Check for expected key in response
#         assert len(response.json()["predictions"]) > 0  # Ensure there are predictions
#         logger.info("      SUCCESS: Asynchronous request completed with response: %s", response.json())
#
#     # Prepare a list of tasks to send simultaneous requests
#     tasks = [make_request(sample_data) for _ in range(10)]  # Adjust the number of requests as needed
#
#     # Execute all tasks concurrently
#     await asyncio.gather(*tasks)
#     logger.info("  SUCCESS: Completed all simultaneous requests.")
