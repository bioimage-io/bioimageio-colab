from hypha_rpc.sync import connect_to_server
import numpy as np
import requests


SERVER_URL = "https://hypha.aicell.io"
WORKSPACE_NAME = "bioimageio-colab"
SERVICE_ID = "microsam"
MODEL_NAME = "vit_b"


def test_service_available():
    service_url = f"{SERVER_URL}/{WORKSPACE_NAME}/services/{SERVICE_ID}/ping"
    response = requests.get(service_url)
    assert response.status_code == 200
    assert response.json() == "pong"

def test_get_service():
    client = connect_to_server({"server_url": SERVER_URL, "method_timeout": 5})
    assert client

    sid = f"{WORKSPACE_NAME}/{SERVICE_ID}"
    segment_svc = client.get_service(sid, {"mode": "random"})
    assert segment_svc.config.workspace == WORKSPACE_NAME
    assert segment_svc.get("segment")
    assert segment_svc.get("clear_cache")

    # Test segmentation
    image = np.random.rand(256, 256)
    features = segment_svc.segment(model_name=MODEL_NAME, image=image, point_coordinates=[[128, 128]], point_labels=[1])
    assert features

    # Test embedding caching
    features = segment_svc.segment(model_name=MODEL_NAME, image=image, point_coordinates=[[20, 50]], point_labels=[1])
    features = segment_svc.segment(model_name=MODEL_NAME, image=image, point_coordinates=[[180, 10]], point_labels=[1])

    # Test embedding computation for running SAM client-side
    result = segment_svc.compute_embedding(model_name=MODEL_NAME, image=image)
    assert result
    embedding = result["features"]
    assert embedding.shape == (1, 256, 64, 64)

    assert segment_svc.clear_cache()
