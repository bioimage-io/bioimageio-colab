import argparse
import asyncio

from bioimageio_colab.register_sam_service import register_service

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Register SAM annotation service on BioImageIO Colab workspace."
    )
    parser.add_argument(
        "--server_url",
        default="https://hypha.aicell.io",
        help="URL of the Hypha server",
    )
    parser.add_argument(
        "--workspace_name", default="bioimageio-colab", help="Name of the workspace"
    )
    parser.add_argument(
        "--service_id",
        default="microsam",
        help="Service ID for registering the service",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Workspace token for connecting to the Hypha server",
    )
    parser.add_argument(
        "--cache_dir",
        default="./.model_cache",
        help="Directory for caching the models",
    )
    parser.add_argument(
        "--ray_address",
        default=None,
        help="Address of the Ray cluster for running SAM",
    )
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    loop.create_task(register_service(args=args))
    loop.run_forever()
