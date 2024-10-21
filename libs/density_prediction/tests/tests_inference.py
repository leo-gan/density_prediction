from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_inference():
    files = [("files", ("test_input_1.txt", b"some content")), ("files", ("test_input_2.txt", b"more content"))]
    response = client.post("/inference/predict", files=files)
    assert response.status_code == 200
    assert response.json() == {"message": "Inference tasks started."}
