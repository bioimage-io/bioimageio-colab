import argparse
import asyncio
from time import sleep
import numpy as np
from hypha_rpc import connect_to_server
from tifffile import imread

SERVER_URL = "https://hypha.aicell.io"
WORKSPACE_NAME = "bioimageio-colab"
SERVICE_ID = "microsam"
MODEL_IDS = ["sam_vit_b", "sam_vit_b_lm", "sam_vit_b_em_organelles"]
IMG_PATH = "./data/example_image.tif"


async def run_client(
    client_id: int, image: np.ndarray, model_id: str, method_timeout: int = 300
):
    print(f"Client {client_id} started", flush=True)
    client = await connect_to_server(
        {"server_url": SERVER_URL, "method_timeout": method_timeout}
    )
    service = await client.get_service(
        f"{WORKSPACE_NAME}/{SERVICE_ID}", {"mode": "random"}
    )
    await service.compute_embedding(model_id=model_id, image=image)
    print(f"Client {client_id} finished", flush=True)


async def stress_test(num_clients: int):
    image = imread(IMG_PATH)
    tasks = []
    for client_id in range(num_clients):
        sleep(0.1)
        model_id = MODEL_IDS[np.random.randint(0, len(MODEL_IDS))]
        tasks.append(run_client(client_id=client_id, image=image, model_id=model_id))
    await asyncio.gather(*tasks)
    print("All clients finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=50)
    args = parser.parse_args()

    asyncio.run(stress_test(args.num_clients))
