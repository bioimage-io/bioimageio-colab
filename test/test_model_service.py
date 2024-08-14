from hypha_rpc.sync import connect_to_server
import numpy as np


def test_get_service(
        server_url: str="https://hypha.aicell.io",
        workspace_name: str="bioimageio-colab",
        client_id: str="model-server",
        service_id: str="interactive-segmentation",
    ):
    client = connect_to_server({"server_url": server_url, "method_timeout": 5})
    assert client

    sid = f"{workspace_name}/{client_id}:{service_id}"
    segment_svc = client.get_service(sid)
    assert segment_svc.id == sid
    assert segment_svc.config.workspace == workspace_name
    assert segment_svc.get("compute_embedding")
    assert segment_svc.get("segment")
    assert segment_svc.get("reset_embedding")
    assert segment_svc.get("remove_user_id")

    assert segment_svc.compute_embedding("vit_b", np.random.rand(256, 256))
    features = segment_svc.segment([[128, 128]], [1])
    assert features
    assert segment_svc.reset_embedding()
    assert segment_svc.remove_user_id()
