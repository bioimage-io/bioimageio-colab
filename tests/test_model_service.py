import numpy as np
import requests
from hypha_rpc.sync import connect_to_server
from tifffile import imread

SERVER_URL = "https://hypha.aicell.io"
WORKSPACE_NAME = "bioimageio-colab"
SERVICE_ID = "microsam"
CLIENT_ID = ""
MODEL_IDS = ["sam_vit_b", "sam_vit_b_lm", "sam_vit_b_em_organelles"]
IMG_PATH = "./data/example_image.tif"


def test_service_http_api():
    client_str = f"{CLIENT_ID}:" if CLIENT_ID else ""
    service_url = f"{SERVER_URL}/{WORKSPACE_NAME}/services/{client_str}{SERVICE_ID}"

    response = requests.get(f"{service_url}/hello")
    assert response.status_code == 200
    assert response.json() == "Welcome to the Interactive Segmentation service!"

    response = requests.get(f"{service_url}/ping")
    assert response.status_code == 200
    assert response.json() == "pong"

    response = requests.get(f"{service_url}/deployment_status")
    assert response.status_code == 200
    for value in response.json().values():
        assert value["status"] == "RUNNING"
        if "deployment" in value:
            for value in value["deployment"].values():
                assert value["status"] == "HEALTHY"


def test_service_python_api():
    client = connect_to_server({"server_url": SERVER_URL, "method_timeout": 5})
    assert client

    client_str = f"{CLIENT_ID}:" if CLIENT_ID else ""
    sid = f"{WORKSPACE_NAME}/{client_str}{SERVICE_ID}"
    service = client.get_service(sid, {"mode": "first"})
    assert service.config.workspace == WORKSPACE_NAME

    # Test service functions
    response = service.hello()
    assert response == "Welcome to the Interactive Segmentation service!"

    response = service.ping()
    assert response == "pong"

    # Test embedding computation
    image = imread(IMG_PATH)
    for model_id in MODEL_IDS:
        result = service.compute_embedding(
            image=image,
            model_id=model_id,
        )
        embedding = result["features"]
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, 256, 64, 64)

        input_size = result["input_size"]
        assert isinstance(input_size, list)
        assert len(input_size) == 2

    # Test mask computation
    # polygon_features = service.compute_mask(
    #     model_name=MODEL_NAME,
    #     embedding=embedding,
    #     image_size=image.shape[:2],
    #     point_coordinates=[[10, 10]],
    #     point_labels=[1],
    #     format="kaibu",
    # )
    # assert isinstance(polygon_features, list)
    # assert len(polygon_features) == 1  # Only one point given

    # Test service test run
    # result = service.test_model(model_name="vit_b_lm")
    # assert result == {"status": "ok"}
