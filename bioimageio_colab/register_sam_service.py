import logging
import os
from functools import partial
from typing import Literal

import numpy as np
import ray
import requests
import torch
from dotenv import find_dotenv, load_dotenv
from hypha_rpc import connect_to_server
from kaibu_utils import mask_to_features

from bioimageio_colab.models import sam_app_registry

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


def hello(context: dict = None) -> str:
    return "Welcome to the Interactive Segmentation service!"


def ping(context: dict = None) -> str:
    return "pong"


async def compute_image_embedding(
    handles: dict,
    image: np.ndarray,
    model_name: str,
    context: dict = None,
) -> dict:
    """
    Compute the embeddings of an image using the specified model.
    """
    try:
        user_id = context["user"].get("id") if context else "anonymous"
        logger.info(
            f"User '{user_id}' - Computing embedding (model: '{model_name}')..."
        )

        # Compute the embedding
        # Returns: {"features": embedding, "input_size": input_size}
        result = await handles[model_name].remote(image)

        logger.info(f"User '{user_id}' - Embedding computed successfully.")

        return result  
    except Exception as e:
        logger.error(f"User '{user_id}' - Error computing embedding: {e}")
        raise e


# def compute_mask(
#     cache_dir: str,
#     model_name: str,
#     embedding: np.ndarray,
#     image_size: tuple,
#     point_coords: np.ndarray,
#     point_labels: np.ndarray,
#     format: Literal["mask", "kaibu"] = "mask",
#     context: dict = None,
# ) -> np.ndarray:
#     """
#     Segment the image using the specified model and the provided point coordinates and labels.
#     """
#     try:
#         user_id = context["user"].get("id") if context else "anonymous"
#         logger.info(f"User '{user_id}' - Segmenting image (model: '{model_name}')...")

#         if not format in ["mask", "kaibu"]:
#             raise ValueError("Invalid format. Please choose either 'mask' or 'kaibu'.")

#         # Load the model
#         sam_predictor = load_model_from_ckpt(
#             model_name=model_name,
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


async def check_readiness(service_url: str) -> dict:
    """
    Readiness probe for the SAM service.
    """
    logger.info("Checking the readiness of the SAM service...")

    response = requests.get(service_url)
    assert response.status_code == 200
    assert response.json() == "pong"

    return {"status": "ok"}


async def check_liveness(handles: dict, model_name: str) -> dict:
    """
    Liveness probe for the SAM service.
    """
    logger.info(f"Checking the liveness of the SAM service (model: '{model_name}')...")

    image = np.random.rand(256, 256)
    result = await compute_image_embedding(handles, image, model_name)

    assert "features" in result
    assert "input_size" in result
    assert isinstance(result["features"], np.ndarray)
    assert result["features"] is not None

    return {"status": "ok"}


async def register_service(args: dict) -> None:
    """
    Register the SAM annotation service on the BioImageIO Colab workspace.
    """
    logger.info("Registering the SAM annotation service...")
    logger.info(f"Available CPU cores: {os.cpu_count()}")
    if torch.cuda.is_available():
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        logger.info(f"Available GPU devices: {torch.cuda.get_device_name()}")
    else:
        logger.info("No GPU devices available.")

    workspace_token = args.token or os.environ.get("WORKSPACE_TOKEN")
    if not workspace_token:
        raise ValueError("Workspace token is required to connect to the Hypha server.")

    # Connect to the workspace (with random client ID)
    colab_client = await connect_to_server(
        {
            "server_url": args.server_url,
            "workspace": args.workspace_name,
            "name": "SAM Server",
            "token": workspace_token,
        }
    )
    client_id = colab_client.config["client_id"]
    workspace = colab_client.config["workspace"]
    logger.info(f"Connected to workspace '{workspace}' with client ID: {client_id}")

    client_base_url = f"{args.server_url}/{workspace}/services/{client_id}"
    cache_dir = os.path.abspath(args.cache_dir)

    if args.ray_address:
        # Create runtime environment
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_requirements = parse_requirements(
            os.path.join(base_dir, "requirements.txt")
        )
        sam_requirements = parse_requirements(
            os.path.join(base_dir, "requirements-sam.txt")
        )
        runtime_env = {
            "pip": base_requirements + sam_requirements,
            "py_modules": [os.path.join(base_dir, "bioimageio_colab")],
        }
    else:
        runtime_env = None

    # Connect to Ray
    ray.init(address=args.ray_address, runtime_env=runtime_env)
    logger.info(f"Successfully connected to Ray (version: {ray.__version__})")

    # Deploy SAM image encoder
    handles = {}
    for model_name, deployment in sam_app_registry.items():
        if model_name not in list(args.model_names):
            continue
        app = deployment(cache_dir=cache_dir)
        handles[model_name] = ray.serve.run(app, name="SAM Image Encoder", route_prefix=None)
        logger.info(f"Deployed SAM image encoder for model '{model_name}'")

    # Register a new service
    service_info = await colab_client.register_service(
        {
            "name": "Interactive Segmentation",
            "id": args.service_id,
            "config": {
                "visibility": "public",
                "require_context": True,  # TODO: only allow the service to be called by logged-in users
                "run_in_executor": False,
            },
            # Exposed functions:
            "hello": hello,
            "ping": ping,
            "compute_embedding": partial(compute_image_embedding, handles),
            # "compute_mask": partial(compute_mask_function, handles),
        }
    )

    sid = service_info["id"]
    logger.info(f"Service registered with ID: {sid}")
    logger.info(f"Test the service here: {client_base_url}:{args.service_id}/hello")

    # Register probes for the service
    service_url = f"{client_base_url}:{args.service_id}/ping"
    await colab_client.register_probes({
        "readiness": partial(check_readiness, service_url),
        "liveness": partial(check_liveness, handles=handles),
    })

    # This will register probes service where you can accessed via hypha or the HTTP proxy
    print(f"Probes registered at workspace: {workspace}")
    print(f"Test the liveness probe here: {args.server_url}/{workspace}/services/probes/liveness?model_name=sam_vit_b_lm")


if __name__ == "__main__":
    model_name = "vit_b_lm"
    cache_dir = "./model_cache"

    embedding = compute_image_embedding(
        cache_dir=cache_dir,
        model_name=model_name,
        image=np.random.rand(1024, 1024),
    )
    # mask = compute_mask(
    #     cache_dir=cache_dir,
    #     model_name="vit_b_lm",
    #     embedding=embedding,
    #     image_size=(1024, 1024),
    #     point_coords=np.array([[10, 10]]),
    #     point_labels=np.array([1]),
    #     format="kaibu",
    #     context={"user": {"id": "test"}},
    # )

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_requirements = parse_requirements(os.path.join(base_dir, "requirements.txt"))
    sam_requirements = parse_requirements(
        os.path.join(base_dir, "requirements-sam.txt")
    )
    runtime_env = {
        "pip": base_requirements + sam_requirements,
        "py_modules": [os.path.join(base_dir, "bioimageio_colab")],
    }
    print(runtime_env)
