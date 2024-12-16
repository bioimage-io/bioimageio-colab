import logging
import os
from functools import partial
from typing import Literal

import numpy as np
import ray
import torch
from dotenv import find_dotenv, load_dotenv
from hypha_rpc import connect_to_server
from kaibu_utils import mask_to_features

from bioimageio_colab.sam import compute_embedding, load_model_from_ckpt, segment_image

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
    skip_lines = ("#", "-r ", "ray")
    packages = [
        line.strip()
        for line in lines
        if line.strip() and not line.startswith(skip_lines)
    ]
    return packages


def hello(context: dict = None) -> str:
    return "Welcome to the Interactive Segmentation service!"


def ping(context: dict = None) -> str:
    return {"status": "ok"}


def compute_image_embedding(
    cache_dir: str,
    model_name: str,
    image: np.ndarray,
    context: dict = None,
) -> np.ndarray:
    """
    Compute the embeddings of an image using the specified model.
    """
    try:
        user_id = context["user"].get("id") if context else "anonymous"
        logger.info(
            f"User '{user_id}' - Computing embedding (model: '{model_name}')..."
        )

        sam_predictor = load_model_from_ckpt(
            model_name=model_name,
            cache_dir=cache_dir,
        )
        sam_predictor = compute_embedding(
            sam_predictor=sam_predictor,
            array=image,
        )

        logger.info(f"User '{user_id}' - Embedding computed successfully.")

        return sam_predictor.features.detach().cpu().numpy()
    except Exception as e:
        logger.error(f"User '{user_id}' - Error computing embedding: {e}")
        raise e


@ray.remote(num_gpus=1)
def compute_image_embedding_ray(kwargs: dict) -> np.ndarray:
    from bioimageio_colab.register_sam_service import compute_image_embedding

    return compute_image_embedding(**kwargs)


def compute_mask(
    cache_dir: str,
    model_name: str,
    embedding: np.ndarray,
    image_size: tuple,
    point_coords: np.ndarray,
    point_labels: np.ndarray,
    format: Literal["mask", "kaibu"] = "mask",
    context: dict = None,
) -> np.ndarray:
    """
    Segment the image using the specified model and the provided point coordinates and labels.
    """
    try:
        user_id = context["user"].get("id") if context else "anonymous"
        logger.info(f"User '{user_id}' - Segmenting image (model: '{model_name}')...")

        if not format in ["mask", "kaibu"]:
            raise ValueError("Invalid format. Please choose either 'mask' or 'kaibu'.")

        # Load the model
        sam_predictor = load_model_from_ckpt(
            model_name=model_name,
            cache_dir=cache_dir,
        )

        # Set the embedding
        sam_predictor.original_size = image_size
        sam_predictor.input_size = tuple(
            [sam_predictor.model.image_encoder.img_size] * 2
        )
        sam_predictor.features = torch.as_tensor(embedding, device=sam_predictor.device)
        sam_predictor.is_image_set = True

        # Segment the image
        masks = segment_image(
            sam_predictor=sam_predictor,
            point_coords=point_coords,
            point_labels=point_labels,
        )

        if format == "mask":
            features = masks
        elif format == "kaibu":
            features = [mask_to_features(mask) for mask in masks]

        logger.info(f"User '{user_id}' - Image segmented successfully.")

        return features
    except Exception as e:
        logger.error(f"User '{user_id}' - Error segmenting image: {e}")
        raise e


@ray.remote(num_gpus=1)
def compute_mask_ray(kwargs: dict) -> np.ndarray:
    from bioimageio_colab.register_sam_service import compute_mask

    return compute_mask(**kwargs)


def test_model(cache_dir: str, model_name: str, context: dict = None) -> dict:
    """
    Test the segmentation service.
    """
    user_id = context["user"].get("id") if context else "anonymous"
    logger.info(f"User '{user_id}' - Test run for model '{model_name}'...")

    image = np.random.rand(1024, 1024)
    embedding = compute_image_embedding(
        cache_dir=cache_dir,
        model_name=model_name,
        image=image,
        context=context,
    )
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1, 256, 64, 64)
    return {"status": "ok"}


@ray.remote(num_gpus=1)
def test_model_ray(kwargs: dict) -> dict:
    from bioimageio_colab.register_sam_service import test_model

    return test_model(**kwargs)


async def register_service(args: dict) -> None:
    """
    Register the SAM annotation service on the BioImageIO Colab workspace.
    """
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
    client_base_url = f"{args.server_url}/{args.workspace_name}/services/{client_id}"
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

        # Connect to Ray
        ray.init(address=args.ray_address, runtime_env=runtime_env)

        def compute_embedding_function(**kwargs: dict):
            kwargs["cache_dir"] = cache_dir
            return ray.get(compute_image_embedding_ray.remote(kwargs))

        def compute_mask_function(**kwargs: dict):
            kwargs["cache_dir"] = cache_dir
            return ray.get(compute_mask_ray.remote(kwargs))

        def test_model_function(**kwargs: dict):
            kwargs["cache_dir"] = cache_dir
            return ray.get(test_model_ray.remote(kwargs))
        
        logger.info(f"Successfully connected to Ray (version: {ray.__version__})")

    else:
        compute_embedding_function = partial(
            compute_image_embedding, cache_dir=cache_dir
        )
        compute_mask_function = partial(compute_mask, cache_dir=cache_dir)
        test_model_function = partial(test_model, cache_dir=cache_dir)

    # Register a new service
    service_info = await colab_client.register_service(
        {
            "name": "Interactive Segmentation",
            "id": args.service_id,
            "config": {
                "visibility": "public",
                "require_context": True,  # TODO: only allow the service to be called by logged-in users
                "run_in_executor": True,
            },
            # Exposed functions:
            "hello": hello,
            "ping": ping,
            "compute_embedding": compute_embedding_function,
            "compute_mask": compute_mask_function,
            "test_model": test_model_function,
        }
    )
    sid = service_info["id"]
    logger.info(f"Service registered with ID: {sid}")
    logger.info(f"Test the service here: {client_base_url}:{args.service_id}/hello")


if __name__ == "__main__":
    model_name = "vit_b_lm"
    cache_dir = "./model_cache"

    embedding = compute_image_embedding(
        cache_dir=cache_dir,
        model_name=model_name,
        image=np.random.rand(1024, 1024),
        context={"user": {"id": "test"}},
    )
    mask = compute_mask(
        cache_dir=cache_dir,
        model_name="vit_b_lm",
        embedding=embedding,
        image_size=(1024, 1024),
        point_coords=np.array([[10, 10]]),
        point_labels=np.array([1]),
        format="kaibu",
        context={"user": {"id": "test"}},
    )

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
