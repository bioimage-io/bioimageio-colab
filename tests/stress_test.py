import argparse
import asyncio
import os

import numpy as np
from hypha_rpc import connect_to_server
from tifffile import imread


SERVER_URL = "https://hypha.aicell.io"
WORKSPACE_NAME = "bioimageio-colab"
CLIENT_ID = os.getenv("CLIENT_ID")
SERVICE_ID = "microsam"
MODEL_IDS = ["sam_vit_b", "sam_vit_b_lm", "sam_vit_b_em_organelles"]
IMG_PATH = "./data/example_image.tif"


async def compute_embedding(req_id, service, image):
    # Prepare image and model ID
    image_prep = image + np.random.normal(0, 0.1, image.shape)
    model_id = MODEL_IDS[np.random.randint(0, len(MODEL_IDS))]

    print(f"Sending request {req_id + 1}")
    await service.compute_embedding(
        image=image_prep,
        model_id=model_id,
    )
    print(f"Request {req_id} finished")


async def stress_test(num_requests: int, method_timeout: int = 30):
    # Connect to the server and get the compute service
    service_client_str = f"{CLIENT_ID}:" if CLIENT_ID else ""
    compute_service_id = f"{WORKSPACE_NAME}/{service_client_str}{SERVICE_ID}"
    print(f"Compute service ID: {compute_service_id}")
    client = await connect_to_server(
        {"server_url": SERVER_URL, "method_timeout": method_timeout}
    )
    service = await client.get_service(compute_service_id, {"mode": "first"})

    # Load the image
    image = imread(IMG_PATH)

    # Send requests
    tasks = []
    for req_id in range(num_requests):
        tasks.append(compute_embedding(req_id, service, image))
    await asyncio.gather(*tasks)

    print("All requests completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_requests", type=int, default=30, help="Number of requests"
    )
    args = parser.parse_args()

    asyncio.run(stress_test(args.num_requests))
