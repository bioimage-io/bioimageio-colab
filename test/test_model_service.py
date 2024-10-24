from hypha_rpc.sync import connect_to_server
import numpy as np
import requests


SERVER_URL = "https://hypha.aicell.io"
WORKSPACE_NAME = "bioimageio-colab"
SERVICE_ID = "sam"


def test_service_available():
    service_url = f"{SERVER_URL}/{WORKSPACE_NAME}/services/*:{SERVICE_ID}/hello"
    response = requests.get(service_url)
    assert response.status_code == 200
    assert response.json() == "Welcome to the Interactive Segmentation service!"

def test_get_service():
    client = connect_to_server({"server_url": SERVER_URL, "method_timeout": 5})
    assert client

    sid = f"{WORKSPACE_NAME}/*:{SERVICE_ID}"
    segment_svc = client.get_service(sid, {"mode": "random"})
    assert segment_svc.id == sid
    assert segment_svc.config.workspace == WORKSPACE_NAME
    assert segment_svc.get("segment")
    assert segment_svc.get("clear_cache")

    features = segment_svc.segment(model_name="vit_b", image=np.random.rand(256, 256), point_coordinates=[[128, 128]], point_labels=[1])
    assert features
    assert segment_svc.clear_cache()
