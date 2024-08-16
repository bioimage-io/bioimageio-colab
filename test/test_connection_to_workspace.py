import os
import pytest

from dotenv import find_dotenv, load_dotenv
from hypha_rpc import connect_to_server


ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

SERVER_URL = "https://hypha.aicell.io"
WORKSPACE = "bioimageio-colab"
WORKSPACE_TOKEN = os.environ.get("WORKSPACE_TOKEN")

@pytest.mark.asyncio
async def test_connection_to_workspace():
    test_client = await connect_to_server(
        {
            "server_url": SERVER_URL,
            "workspace": WORKSPACE,
            "token": WORKSPACE_TOKEN,
        }
    )
    assert test_client.config.workspace == WORKSPACE
