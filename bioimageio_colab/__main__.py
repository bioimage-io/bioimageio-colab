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
        default="/tmp/ray/.model_cache",
        help="Directory for caching the models",
    )
    parser.add_argument(
        "--ray_address",
        default=None,
        help="Address of the Ray cluster for running SAM",
    )
    parser.add_argument(
        "--num_replicas",
        default=2,
        type=int,
        help="Number of replicas for the SAM deployment",
    )
    parser.add_argument(
        "--restart_deployment",
        default=False,
        action="store_true",
        help="Restart the Ray deployment if it already exists",
    )
    parser.add_argument(
        "--skip_test_runs",
        default=False,
        action="store_true",
        help="Skip test run of each model",
    )
    parser.add_argument(
        "--max_concurrent_requests",
        default=4,
        type=int,
        help="Maximum number of concurrent requests to the service",
    )
    parser.add_argument(
        "--require_login",
        default=False,
        action="store_true",
        help="Require login to access the function `compute_image_embedding`",
    )
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    loop.create_task(register_service(args=args))
    loop.run_forever()
