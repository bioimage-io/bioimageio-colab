"""
This script cleans up all dangling services in the workspace.

User interface for this feature available in the Hypha dashboard ( https://hypha.aicell.io/bioimageio-colab#dashboard -> brush icon ).
"""

from hypha_rpc.sync import connect_to_server, login


SERVER_URL = "https://hypha.aicell.io"
WORKSPACE_NAME = "bioimageio-colab"

token = login({"server_url": SERVER_URL})
server = connect_to_server(
    {"server_url": SERVER_URL, "token": token, "workspace": WORKSPACE_NAME}
)

server.cleanup()
