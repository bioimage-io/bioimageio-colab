from hypha_rpc import connect_to_server
import numpy as np
import asyncio
import argparse


SERVER_URL = "https://hypha.aicell.io"
WORKSPACE_NAME = "bioimageio-colab"
CLIENT_ID = "kubernetes"
SERVICE_ID = "sam"
SID = f"{WORKSPACE_NAME}/{CLIENT_ID}:{SERVICE_ID}"


async def run_client(client_id: int, image: np.ndarray, point_coordinates: list, point_labels: list):
    print(f"Client {client_id} started")
    client = await connect_to_server({"server_url": SERVER_URL, "method_timeout": 10})
    segment_svc = await client.get_service(SID)
    await segment_svc.segment(model_name="vit_b", image=image, point_coordinates=point_coordinates, point_labels=point_labels)


async def stress_test(num_clients: int):
    image=np.random.rand(256, 256)
    point_coordinates=[[128, 128]]
    point_labels=[1]
    tasks = [
        run_client(client_id, image, point_coordinates, point_labels)
        for client_id in range(num_clients)
    ]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=50)
    args = parser.parse_args()

    asyncio.run(stress_test(args.num_clients))