import numpy as np
import requests
from hypha_rpc.sync import connect_to_server
# from bioimageio_colab.register_sam_service import compute_image_embedding, compute_mask

SERVER_URL = "https://hypha.aicell.io"
WORKSPACE_NAME = "bioimageio-colab"
SERVICE_ID = "microsam"
MODEL_NAME = "sam_vit_b"


# def test_service_functions():
#     cache_dir = "./model_cache"
#     embedding = compute_image_embedding(
#         cache_dir=cache_dir,
#         model_name=MODEL_NAME,
#         image=np.random.rand(256, 256),
#         context={"user": {"id": "test"}},
#     )
#     assert isinstance(embedding, np.ndarray)
#     assert embedding.shape == (1, 256, 64, 64)

#     polygon_features = compute_mask(
#         cache_dir=cache_dir,
#         model_name=MODEL_NAME,
#         embedding=embedding,
#         image_size=(1024, 1024),
#         point_coords=np.array([[10, 10]]),
#         point_labels=np.array([1]),
#         format="kaibu",
#         context={"user": {"id": "test"}},
#     )
#     assert isinstance(polygon_features, list)
#     assert len(polygon_features) == 1  # Only one point given


def test_service_is_running_http_api():
    service_url = f"{SERVER_URL}/{WORKSPACE_NAME}/services/{SERVICE_ID}/ping"
    response = requests.get(service_url)
    assert response.status_code == 200
    assert response.json() == "pong"


def test_service_python_api():
    client = connect_to_server({"server_url": SERVER_URL, "method_timeout": 5})
    assert client

    sid = f"{WORKSPACE_NAME}/{SERVICE_ID}"
    segment_svc = client.get_service(sid, {"mode": "random"})
    assert segment_svc.config.workspace == WORKSPACE_NAME
    assert segment_svc.get("hello")
    assert segment_svc.get("ping")
    assert segment_svc.get("compute_embedding")
    # assert segment_svc.get("compute_mask")
    assert segment_svc.get("test_model")

    # Test embedding computation
    image = np.random.rand(256, 256)
    result = segment_svc.compute_embedding(
        image=image,
        model_name=MODEL_NAME,
    )
    embedding = result["features"]
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1, 256, 64, 64)

    input_size = result["input_size"]
    assert isinstance(input_size, list)
    assert len(input_size) == 2

    # Test mask computation
    # polygon_features = segment_svc.compute_mask(
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
    # result = segment_svc.test_model(model_name="vit_b_lm")
    # assert result == {"status": "ok"}
