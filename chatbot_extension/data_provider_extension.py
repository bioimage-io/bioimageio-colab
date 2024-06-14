import asyncio
from pydantic import BaseModel, Field
from imjoy_rpc.hypha import login, connect_to_server
from bioimageio_colab_server.annotation_server import start_server
    
class RegisterService(BaseModel):
    """Register collaborative annotation service to start a collaborative annotation session."""
    path2data: str = Field(..., description="Path to data folder from which the images are loaded; example: /mnt/data")
    outpath: str = Field(..., description="Path to output folder to which the annotations are saved; example: /mnt/annotations")


async def register_service(kwargs):
    await start_server(**kwargs)
    return "success"


def get_schema():
        return {
            "move_stage": RegisterService.schema()
        }

async def main():
    # Define an chatbot extension
    microscope_control_extension = {
        "_rintf": True,
        "id": "annotation-colab-provider",
        "config": {"visibility": "public"},
        "type": "bioimageio-chatbot-extension",
        "name": "Annotation Colab Data Provider",
        "description": "This extension starts a collaborative annotation session. It provides data for remote annotation and then saves the annotations.",
        "get_schema": get_schema,
        "tools": {
            "move_stage": register_service,
        }
    }

    # Connect to the chat server
    server_url = "https://chat.bioimage.io"
    token = await login({"server_url": server_url})
    server = await connect_to_server({"server_url": server_url, "token": token})
    # Register the extension service
    svc = await server.register_service(microscope_control_extension)
    print(f"Extension service registered with id: {svc.id}, you can visit the service at: https://bioimage.io/chat?server={server_url}&extension={svc.id}&assistant=Bridget")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()
