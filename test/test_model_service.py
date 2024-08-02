import os
import pytest

from hypha_rpc.hypha.sync import connect_to_server


def test_get_service():
    client = connect_to_server({"server_url": "https://hypha.aicell.io", "method_timeout": 5})
    assert client

    service = client.getService("bioimageio-colab/model-server:interactive-segmentation")
    assert service.name == "Interactive Segmentation"
    assert service.id == "bioimageio-colab/model-server:interactive-segmentation"
    assert service.config.workspace == "bioimageio-colab"
    assert service.get("compute_embedding")
    assert service.get("segment")
    assert service.get("reset_embedding")
    assert service.get("remove_user_id")
