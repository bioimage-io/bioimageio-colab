import numpy as np
import requests
from hypha_rpc.sync import connect_to_server
from tifffile import imread
import os

SERVER_URL = "https://hypha.aicell.io"
WORKSPACE_NAME = "bioimageio-colab"
SERVICE_ID = "microsam"
CLIENT_ID = os.getenv("CLIENT_ID")
MODEL_IDS = ["sam_vit_b", "sam_vit_b_lm", "sam_vit_b_em_organelles"]
IMG_PATH = "./data/example_image.tif"


def test_service_http_api():
    client_str = f"{CLIENT_ID}:" if CLIENT_ID else ""
    service_url = f"{SERVER_URL}/{WORKSPACE_NAME}/services/{client_str}{SERVICE_ID}"

    response = requests.get(f"{service_url}/hello?_mode=last")
    assert response.status_code == 200
    assert response.json() == "Welcome to the Interactive Segmentation service!"

    response = requests.get(f"{service_url}/ping?_mode=last")
    assert response.status_code == 200
    assert response.json() == "pong"

    response = requests.get(f"{service_url}/deployment_status?assert_status=True&_mode=last")
    assert response.status_code == 200
    for value in response.json().values():
        assert value["status"] == "RUNNING"
        if "deployment" in value:
            for value in value["deployment"].values():
                assert value["status"] == "HEALTHY"


def test_service_hello():
    client = connect_to_server({"server_url": SERVER_URL, "method_timeout": 5})
    assert client

    client_str = f"{CLIENT_ID}:" if CLIENT_ID else ""
    sid = f"{WORKSPACE_NAME}/{client_str}{SERVICE_ID}"
    service = client.get_service(sid, {"mode": "last"})
    assert service.config.workspace == WORKSPACE_NAME

    response = service.hello()
    assert response == "Welcome to the Interactive Segmentation service!"


def test_service_ping():
    client = connect_to_server({"server_url": SERVER_URL, "method_timeout": 5})
    assert client

    client_str = f"{CLIENT_ID}:" if CLIENT_ID else ""
    sid = f"{WORKSPACE_NAME}/{client_str}{SERVICE_ID}"
    service = client.get_service(sid, {"mode": "last"})
    assert service.config.workspace == WORKSPACE_NAME

    response = service.ping()
    assert response == "pong"


def test_service_compute_embedding():
    client = connect_to_server({"server_url": SERVER_URL, "method_timeout": 5})
    assert client

    client_str = f"{CLIENT_ID}:" if CLIENT_ID else ""
    sid = f"{WORKSPACE_NAME}/{client_str}{SERVICE_ID}"
    service = client.get_service(sid, {"mode": "last"})
    assert service.config.workspace == WORKSPACE_NAME

    image = imread(IMG_PATH)
    for model_id in MODEL_IDS:
        result = service.compute_embedding(
            image=image,
            model_id=model_id,
        )
        embedding = result["features"]
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, 256, 64, 64)

        original_image_shape = result["original_image_shape"]
        assert isinstance(original_image_shape, list)
        assert original_image_shape == [512, 512]

        sam_scale = result["sam_scale"]
        assert isinstance(sam_scale, float)
        assert sam_scale == 2.0


def test_service_get_onnx_model():
    client = connect_to_server({"server_url": SERVER_URL, "method_timeout": 5})
    assert client

    client_str = f"{CLIENT_ID}:" if CLIENT_ID else ""
    sid = f"{WORKSPACE_NAME}/{client_str}{SERVICE_ID}"
    service = client.get_service(sid, {"mode": "last"})
    assert service.config.workspace == WORKSPACE_NAME

    for model_id in MODEL_IDS:
        onnx_model = service.get_onnx_model(
            model_id=model_id,
            quantize=True,
        )
        assert isinstance(onnx_model, bytes)
        assert len(onnx_model) > 0


def test_service_segment_image():
    client = connect_to_server({"server_url": SERVER_URL, "method_timeout": 5})
    assert client

    client_str = f"{CLIENT_ID}:" if CLIENT_ID else ""
    sid = f"{WORKSPACE_NAME}/{client_str}{SERVICE_ID}"
    service = client.get_service(sid, {"mode": "last"})
    assert service.config.workspace == WORKSPACE_NAME

    image = imread(IMG_PATH)
    for model_id in MODEL_IDS:
        segmentation_result = service.segment_image(
            image=image,
            model_id=model_id,
            points_per_side=16,
            pred_iou_thresh=0.9,
            stability_score_thresh=0.95,
            min_mask_region_area=100,
        )
        assert isinstance(segmentation_result, dict)
        assert "mask" in segmentation_result
        assert isinstance(segmentation_result["mask"], np.ndarray)
        assert segmentation_result["mask"].size > 0
