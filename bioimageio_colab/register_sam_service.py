import logging
import os
from datetime import datetime, timezone
from functools import partial

import asyncio
import numpy as np
import ray
from dotenv import find_dotenv, load_dotenv
from hypha_rpc import connect_to_server
from hypha_rpc.rpc import RemoteService
# from kaibu_utils import mask_to_features
from ray.serve.config import AutoscalingConfig
from tifffile import imread

from bioimageio_colab.models import SAM_MODELS, SamDeployment

# Load environment variables
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

logger = logging.getLogger(__name__)
logger.setLevel("INFO")
# Disable propagation to avoid duplication of logs
logger.propagate = False
# Create a new console handler
console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Set the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_requirements(file_path) -> list:
    with open(file_path, "r") as file:
        lines = file.readlines()
    # Filter and clean package names (skip comments and empty lines)
    skip_lines = ("#", "-r ")
    packages = [
        line.strip()
        for line in lines
        if line.strip() and not line.startswith(skip_lines)
    ]
    return packages


def connect_to_ray(address: str = None) -> None:
    logger.info("Connecting to Ray...")
    # Create runtime environment
    sam_requirements = parse_requirements(
        os.path.join(BASE_DIR, "requirements-sam.txt")
    )
    runtime_env = {
        "pip": sam_requirements,
        "py_modules": [os.path.join(BASE_DIR, "bioimageio_colab")],
    }

    # Check if Ray is already initialized
    if ray.is_initialized():
        # Disconnect previous client
        ray.shutdown()

    # Connect to Ray
    ray.init(address=address, runtime_env=runtime_env)
    logger.info(f"Successfully connected to Ray (version: {ray.__version__})")


async def deploy_to_ray(
    cache_dir: str,
    app_name: str = "SAM Image Encoder",
    min_replicas: int = 1,
    max_replicas: int = 2,
    restart_deployment: bool = False,
    skip_test_runs: bool = False,
) -> None:
    """
    Deploy the SAM image encoders to Ray Serve.

    Args:
        models (list): List of SAM models to deploy.
    Returns:
        dict: Handles to the deployed image encoders.
    """
    logger.info(f"Deploying the app '{app_name}' with {min_replicas} to {max_replicas} replicas on Ray Serve...")
    # Set autoscaling configuration
    autoscaling_config = AutoscalingConfig(
        min_replicas=min_replicas,
        max_replicas=max_replicas,
    )

    # Create a new Ray Serve deployment
    deployment = SamDeployment.options(
        autoscaling_config=autoscaling_config,
    )

    # Bind the arguments to the deployment and return an Application.
    app = deployment.bind(cache_dir=cache_dir)

    # Check if the application is already deployed
    serve_status = ray.serve.status()
    applications = serve_status.applications

    if app_name in applications and restart_deployment:
        # Delete the existing application
        ray.serve.delete(app_name)
        logger.info(f"Deleted existing application '{app_name}'")

    # Deploy the application
    ray.serve.run(app, name=app_name, route_prefix=None)

    if app_name in applications:
        logger.info(f"Updated application deployment '{app_name}'.")
    else:
        logger.info(f"Deployed application '{app_name}'.")
    
    if skip_test_runs:
        logger.info("Skipping test runs for each model.")
    else:
        # Test run each model
        handle = ray.serve.get_app_handle(name=app_name)
        img_file = os.path.join(BASE_DIR, "data/example_image.tif")
        image = imread(img_file)
        for model_id in SAM_MODELS.keys():
            await handle.remote(model_id=model_id, array=image)
            logger.info(f"Test run for model '{model_id}' completed successfully.")


def hello(context: dict = None) -> str:
    return "Welcome to the Interactive Segmentation service!"


def ping(context: dict = None) -> str:
    return "pong"


async def compute_image_embedding(
    semaphore: asyncio.Semaphore,
    app_name: str,
    image: np.ndarray,
    model_id: str,
    require_login: bool = False,
    context: dict = None,
) -> dict:
    """
    Compute the embeddings of an image using the specified model.
    """
    async with semaphore:
        try:
            user = context["user"]
            if require_login and user["is_anonymous"]:
                raise PermissionError("You must be logged in to use this service.")

            user_id = user["id"]
            logger.info(
                f"User '{user_id}' - Computing embedding (model: '{model_id}')..."
            )

            # Compute the embedding
            # Returns: {"features": embedding, "input_size": input_size}
            handle = ray.serve.get_app_handle(name=app_name)
            result = await handle.remote(model_id=model_id, array=image)

            logger.info(f"User '{user_id}' - Embedding computed successfully.")

            return result
        except Exception as e:
            logger.error(f"User '{user_id}' - Error computing embedding: {e}")
            raise e


# def compute_mask(
#     cache_dir: str,
#     model_id: str,
#     embedding: np.ndarray,
#     image_size: tuple,
#     point_coords: np.ndarray,
#     point_labels: np.ndarray,
#     format: Literal["mask", "kaibu"] = "mask",
#     require_login: bool = False,
#     context: dict = None,
# ) -> np.ndarray:
#     """
#     Segment the image using the specified model and the provided point coordinates and labels.
#     """
#     try:
#         user_id = context["user"].get("id") if context else "anonymous"
#         logger.info(f"User '{user_id}' - Segmenting image (model: '{model_id}')...")

#         if not format in ["mask", "kaibu"]:
#             raise ValueError("Invalid format. Please choose either 'mask' or 'kaibu'.")

#         # Load the model
#         sam_predictor = load_model_from_ckpt(
#             model_id=model_id,
#             cache_dir=cache_dir,
#         )

#         # Set the embedding
#         sam_predictor.original_size = image_size
#         sam_predictor.input_size = tuple(
#             [sam_predictor.model.image_encoder.img_size] * 2
#         )
#         sam_predictor.features = torch.as_tensor(embedding, device=sam_predictor.device)
#         sam_predictor.is_image_set = True

#         # Segment the image
#         masks = segment_image(
#             sam_predictor=sam_predictor,
#             point_coords=point_coords,
#             point_labels=point_labels,
#         )

#         if format == "mask":
#             features = masks
#         elif format == "kaibu":
#             features = [mask_to_features(mask) for mask in masks]

#         logger.info(f"User '{user_id}' - Image segmented successfully.")

#         return features
#     except Exception as e:
#         logger.error(f"User '{user_id}' - Error segmenting image: {e}")
#         raise e


async def check_readiness(client: RemoteService, service_id: str) -> dict:
    """
    Readiness probe for the SAM service.
    """
    logger.info("Checking the readiness of the SAM service...")

    services = await client.list_services()
    service_found = False
    for service in services:
        if service["id"] == service_id:
            service_found = True
            break
    assert service_found, f"Service with ID '{service_id}' not found."

    logger.info(f"Service with ID '{service_id}' is ready.")

    return {"status": "ready"}


def format_time(last_deployed_time_s, tz: timezone = timezone.utc) -> str:
    # Get the current time
    current_time = datetime.now(tz)
    last_deployed_time = datetime.fromtimestamp(last_deployed_time_s, tz)

    # Calculate the duration since the last deployment
    duration = current_time - last_deployed_time

    # Break down the duration into days, hours, minutes, and seconds
    days = duration.days
    seconds = duration.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60

    # Build the duration string
    duration_parts = []
    if days > 0:
        duration_parts.append(f"{days}d")
    if hours > 0:
        duration_parts.append(f"{hours}h")
    if minutes > 0:
        duration_parts.append(f"{minutes}m")
    if remaining_seconds > 0:
        duration_parts.append(f"{remaining_seconds}s")

    duration_str = " ".join(duration_parts)

    return {
        "last_deployed_at": last_deployed_time.strftime("%Y/%m/%d %H:%M:%S"),
        "duration_since": duration_str,
    }


async def check_liveness(app_name: str, client: RemoteService, service_id: str) -> dict:
    """
    Liveness probe for the SAM service.
    """
    logger.info(f"Checking the liveness of the SAM service...")
    output = {}

    # Check the Ray Serve application status
    serve_status = ray.serve.status()
    application = serve_status.applications[app_name]
    assert application.status == "RUNNING"
    formatted_time = format_time(application.last_deployed_time_s)
    output[f"application: {app_name}"] = {
        "status": application.status.value,
        "last_deployed_at": formatted_time["last_deployed_at"],
        "duration_since": formatted_time["duration_since"],
    }

    for name, deployment in serve_status.applications[app_name].deployments.items():
        assert deployment.status == "HEALTHY"
        # assert deployment.replica_states == {"RUNNING": 1}
        output[f"application: {app_name}"][f"deployment: {name}"] = {
            "status": deployment.status.value,
            # "replica_states": deployment.replica_states,
        }

    # Check if the service can be accessed
    service = await client.get_service(service_id)
    assert await service.ping() == "pong"

    output["service"] = {
        "status": "RUNNING",
        "service_id": service_id,
    }

    logger.info(f"Service with ID '{service_id}' is live.")

    return output


async def register_service(args: dict) -> None:
    """
    Register the SAM annotation service on the BioImageIO Colab workspace.
    """
    logger.info("Registering the SAM annotation service...")
    logger.info(f"Available CPU cores: {os.cpu_count()}")

    workspace_token = args.token or os.environ.get("WORKSPACE_TOKEN")
    if not workspace_token:
        raise ValueError("Workspace token is required to connect to the Hypha server.")

    # Connect to the workspace (with random client ID)
    client = await connect_to_server(
        {
            "server_url": args.server_url,
            "workspace": args.workspace_name,
            "name": "SAM Server",
            "token": workspace_token,
            "ping_interval": None,  # For long-running services (hypha>=0.20.47)
        }
    )
    client_id = client.config["client_id"]
    workspace = client.config["workspace"]
    client_base_url = f"{args.server_url}/{workspace}/services/{client_id}"
    logger.info(f"Connected to workspace '{workspace}' with client ID: {client_id}")

    # Connect client to Ray
    connect_to_ray(address=args.ray_address)

    # Deploy SAM image encoders
    cache_dir = os.path.abspath(args.cache_dir)
    app_name = "SAM Image Encoder"
    await deploy_to_ray(
        cache_dir=cache_dir,
        app_name=app_name,
        restart_deployment=args.restart_deployment,
        skip_test_runs=args.skip_test_runs,
    )

    # Register a new service
    semaphore = asyncio.Semaphore(args.max_concurrent_requests)
    logger.info(f"Created semaphore for {args.max_concurrent_requests} concurrent requests.")

    logger.ingo(f"Registering the SAM service: ID='{args.service_id}', require_login={args.require_login}")
    service_info = await client.register_service(
        {
            "name": "Interactive Segmentation",
            "id": args.service_id,
            "config": {
                "visibility": "public",
                "require_context": True,
                "run_in_executor": False,
            },
            # Exposed functions:
            "hello": hello,
            "ping": ping,
            "compute_embedding": partial(
                compute_image_embedding,
                semaphore=semaphore,
                app_name=app_name,
                require_login=args.require_login,
            ),
            # "compute_mask": partial(
            #     compute_mask,
            #     require_login=args.require_login
            # ),
        }
    )

    sid = service_info["id"]
    logger.info(f"Service registered with ID: {sid}")
    logger.info(f"Test the service here: {client_base_url}:{args.service_id}/hello")

    # Register probes for the service
    await client.register_probes(
        {
            "readiness": partial(
                check_readiness,
                client=client,
                service_id=sid,
            ),
            "liveness": partial(
                check_liveness,
                app_name=app_name,
                client=client,
                service_id=sid,
            ),
        }
    )

    # This will register probes service where you can accessed via hypha or the HTTP proxy
    logger.info(f"Probes registered at workspace: {workspace}")
    logger.info(
        f"Test the liveness probe here: {args.server_url}/{workspace}/services/probes/liveness"
    )
