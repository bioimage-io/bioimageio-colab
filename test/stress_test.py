from hypha_rpc import connect_to_server
import numpy as np
import asyncio
import argparse


SERVER_URL = "https://hypha.aicell.io"
WORKSPACE_NAME = "bioimageio-colab"
SERVICE_ID = "microsam"
SID = f"{WORKSPACE_NAME}/{SERVICE_ID}"


async def run_client(client_id: int, image: np.ndarray):
    print(f"Client {client_id} started", flush=True)
    client = await connect_to_server({"server_url": SERVER_URL, "method_timeout": 30})
    segment_svc = await client.get_service(SID, {"mode": "random"})
    await segment_svc.segment(model_name="vit_b", image=image, point_coordinates=[[128, 128]], point_labels=[1])
    await asyncio.sleep(1)
    await segment_svc.segment(model_name="vit_b", image=image, point_coordinates=[[20, 50]], point_labels=[1])
    await asyncio.sleep(1)
    await segment_svc.segment(model_name="vit_b", image=image, point_coordinates=[[180, 10]], point_labels=[1])
    print(f"Client {client_id} finished", flush=True)


async def stress_test(num_clients: int):
    image=np.random.rand(256, 256)
    tasks = []
    for client_id in range(num_clients):
        await asyncio.sleep(0.1)
        tasks.append(run_client(client_id, image))
    await asyncio.gather(*tasks)
    print("All clients finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=50)
    args = parser.parse_args()

    asyncio.run(stress_test(args.num_clients))