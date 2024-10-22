import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# Test for valid prediction request with model A
def test_predict_valid_model_a():
    valid_input = np.random.rand(3, 1000).tolist()

    response = client.post("/predict?model=model_a", json={"array": valid_input})

    assert response.status_code == 200
    result = response.json()
    assert isinstance(result["array"], list)
    assert len(result["array"]) == 1000


# Test for valid prediction request with model B
def test_predict_valid_model_b():
    valid_input = np.random.rand(3, 1000).tolist()

    response = client.post("/predict?model=model_b", json={"array": valid_input})

    assert response.status_code == 200
    result = response.json()
    assert isinstance(result["array"], list)
    assert len(result["array"]) == 1000


# Test for invalid model
def test_predict_invalid_model():
    valid_input = np.random.rand(3, 1000).tolist()

    response = client.post("/predict?model=invalid_model", json={"array": valid_input})

    assert response.status_code == 400
    assert response.json() == {"detail": "Model 'invalid_model' is not available"}


# Test for invalid shape of input array
def test_predict_invalid_shape():
    invalid_input = np.random.rand(2, 1000).tolist()

    response = client.post("/predict?model=model_a", json={"array": invalid_input})

    assert response.status_code == 400
    assert response.json() == {"detail": "Input array must be of shape [3, 1000]."}


# Test for malformed request
def test_predict_malformed_request():
    malformed_input = {"wrong_key": [1, 2, 3]}

    response = client.post("/predict?model=model_a", json=malformed_input)

    assert (
        response.status_code == 422
    )  # Unprocessable Entity due to invalid request format
