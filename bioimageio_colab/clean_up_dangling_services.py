from hypha_rpc.sync import connect_to_server, login


SERVER_URL = "https://hypha.aicell.io"
WORKSPACE_NAME = "bioimageio-colab"

token = login({"server_url": SERVER_URL})
server = connect_to_server(
    {"server_url": SERVER_URL, "token": token, "workspace": WORKSPACE_NAME}
)

server.cleanup()
