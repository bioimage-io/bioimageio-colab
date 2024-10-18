import argparse
import io
import os
from functools import partial
from logging import getLogger
from typing import Union

import numpy as np
import requests
import torch
from cachetools import TTLCache
from dotenv import find_dotenv, load_dotenv
from hypha_rpc import connect_to_server
from kaibu_utils import mask_to_features
from segment_anything import SamPredictor, sam_model_registry

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

MODELS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_b_lm": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/diplomatic-bug/1/files/vit_b.pt",
    "vit_b_em_organelles": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/noisy-ox/1/files/vit_b.pt",
}

logger = getLogger(__name__)
logger.setLevel("INFO")


def _load_model(
    model_cache: TTLCache, model_name: str, user_id: str
) -> torch.nn.Module:
    if model_name not in MODELS:
        raise ValueError(
            f"Model {model_name} not found. Available models: {list(MODELS.keys())}"
        )

    # Check cache first
    sam = model_cache.get(model_name, None)
    if sam:
        logger.info(
            f"User {user_id} - Loading model '{model_name}' from cache (device={sam.device})..."
        )
    else:
        # Download model if not in cache
        model_url = MODELS[model_name]
        logger.info(
            f"User {user_id} - Loading model '{model_name}' from {model_url}..."
        )
        response = requests.get(model_url)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download model from {model_url}")
        buffer = io.BytesIO(response.content)

        # Load model state
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(buffer, map_location=device)
        model_type = model_name[:5]
        sam = sam_model_registry[model_type]()
        sam.load_state_dict(ckpt)
        logger.info(
            f"User {user_id} - Caching model '{model_name}' (device={device})..."
        )

    # Cache the model / renew the TTL
    model_cache[model_name] = sam

    # Create a SAM predictor
    sam_predictor = SamPredictor(sam)
    return sam_predictor


def _to_image(input_: np.ndarray) -> np.ndarray:
    # we require the input to be uint8
    if input_.dtype != np.dtype("uint8"):
        # first normalize the input to [0, 1]
        input_ = input_.astype("float32") - input_.min()
        input_ = input_ / input_.max()
        # then bring to [0, 255] and cast to uint8
        input_ = (input_ * 255).astype("uint8")
    if input_.ndim == 2:
        image = np.concatenate([input_[..., None]] * 3, axis=-1)
    elif input_.ndim == 3 and input_.shape[-1] == 3:
        image = input_
    else:
        raise ValueError(
            f"Invalid input image of shape {input_.shape}. Expect either 2D grayscale or 3D RGB image."
        )
    return image


def _calculate_embedding(
    embedding_cache: TTLCache,
    sam_predictor: SamPredictor,
    model_name: str,
    image: np.ndarray,
    user_id: str,
) -> np.ndarray:
    # Calculate the embedding if not cached
    predictor_dict = embedding_cache.get(user_id, {})
    if predictor_dict and predictor_dict.get("model_name") == model_name:
        logger.info(
            f"User {user_id} - Loading image embedding from cache (model: '{model_name}')..."
        )
        for key, value in predictor_dict.items():
            if key != "model_name":
                setattr(sam_predictor, key, value)
    else:
        logger.info(
            f"User {user_id} - Computing image embedding (model: '{model_name}')..."
        )
        sam_predictor.set_image(_to_image(image))
        logger.info(
            f"User {user_id} - Caching image embedding (model: '{model_name}')..."
        )
        predictor_dict = {
            "model_name": model_name,
            "original_size": sam_predictor.original_size,
            "input_size": sam_predictor.input_size,
            "features": sam_predictor.features,  # embedding
            "is_image_set": sam_predictor.is_image_set,
        }
    # Cache the embedding / renew the TTL
    embedding_cache[user_id] = predictor_dict

    return sam_predictor


def _segment_image(
    sam_predictor: SamPredictor,
    model_name: str,
    point_coordinates: Union[list, np.ndarray],
    point_labels: Union[list, np.ndarray],
    user_id: str,
):
    if isinstance(point_coordinates, list):
        point_coordinates = np.array(point_coordinates, dtype=np.float32)
    if isinstance(point_labels, list):
        point_labels = np.array(point_labels, dtype=np.float32)
    logger.debug(
        f"User {user_id} - point coordinates: {point_coordinates}, {point_labels}"
    )
    logger.info(f"User {user_id} - Segmenting image (model: '{model_name}')...")
    mask, scores, logits = sam_predictor.predict(
        point_coords=point_coordinates[:, ::-1],  # SAM has reversed XY conventions
        point_labels=point_labels,
        multimask_output=False,
    )
    logger.debug(f"User {user_id} - predicted mask of shape {mask.shape}")
    features = mask_to_features(mask[0])
    return features


def segment(
    model_cache: TTLCache,
    embedding_cache: TTLCache,
    model_name: str,
    image: np.ndarray,
    point_coordinates: Union[list, np.ndarray],
    point_labels: Union[list, np.ndarray],
    context: dict = None,
) -> list:
    user_id = context["user"].get("id")
    if not user_id:
        logger.info("User ID not found in context.")
        return False

    # Load the model
    sam_predictor = _load_model(model_cache, model_name, user_id)

    # Calculate the embedding
    sam_predictor = _calculate_embedding(
        embedding_cache, sam_predictor, model_name, image, user_id
    )

    # Segment the image
    features = _segment_image(
        sam_predictor, model_name, point_coordinates, point_labels, user_id
    )

    return features


def clear_cache(embedding_cache: TTLCache, context: dict = None) -> bool:
    user_id = context["user"].get("id")
    if user_id not in embedding_cache:
        return False
    else:
        logger.info(f"User {user_id} - Resetting embedding cache...")
        del embedding_cache[user_id]
        return True


def hello(context: dict = None) -> str:
    return "Welcome to the Interactive Segmentation service!"


async def register_service(args: dict) -> None:
    """
    Register the SAM annotation service on the BioImageIO Colab workspace.
    """
    workspace_token = args.token or os.environ.get("WORKSPACE_TOKEN")
    if not workspace_token:
        raise ValueError("Workspace token is required to connect to the Hypha server.")

    # Wait until the client ID is available
    test_client = await connect_to_server(
        {
            "server_url": args.server_url,
            "workspace": args.workspace_name,
            "token": workspace_token,
        }
    )
    colab_client_id = f"{args.workspace_name}/{args.client_id}"
    n_failed_attempts = 0
    waiting_for_client = True
    while waiting_for_client:
        all_clients = await test_client.list_clients()
        waiting_for_client = any([colab_client_id == client["id"] for client in all_clients])
        if waiting_for_client:
            n_failed_attempts += 1
            logger.info(
                f"Waiting for client ID '{colab_client_id}' to be available... (attempt {n_failed_attempts})"
            )
            await asyncio.sleep(1)

    # Connect to the workspace
    colab_client = await connect_to_server(
        {
            "server_url": args.server_url,
            "workspace": args.workspace_name,
            "client_id": args.client_id,
            "name": "SAM Server",
            "token": workspace_token,
        }
    )

    # Initialize caches
    model_cache = TTLCache(maxsize=len(MODELS), ttl=args.model_timeout)
    embedding_cache = TTLCache(maxsize=args.max_num_clients, ttl=args.embedding_timeout)

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
            # **Run segmentation**
            # Params:
            # - model name
            # - image to compute the embeddings on
            # - point coordinates (XY format)
            # - point labels
            # Returns:
            # - a list of XY coordinates of the segmented polygon in the format (1, N, 2)
            "segment": partial(segment, model_cache, embedding_cache),
            # **Clear the embedding cache**
            # Returns:
            # - True if the embedding was removed successfully
            # - False if the user was not found in the cache
            "clear_cache": partial(clear_cache, embedding_cache),
        },
        {"overwrite": True},
    )
    sid = service_info["id"]
    assert sid == f"{args.workspace_name}/{args.client_id}:{args.service_id}"
    logger.info(f"Registered service with ID: {sid}")
    logger.info(
        f"Test the service here: {args.server_url}/{args.workspace_name}/services/{args.client_id}:{args.service_id}/hello"
    )


if __name__ == "__main__":
    import asyncio

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
        "--client_id",
        default="kubernetes",
        help="Client ID for registering the service",
    )
    parser.add_argument(
        "--service_id",
        default="sam",
        help="Service ID for registering the service",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Workspace token for connecting to the Hypha server",
    )
    parser.add_argument(
        "--model_timeout",
        type=int,
        default=9600,  # 3 hours
        help="Model cache timeout in seconds",
    )
    parser.add_argument(
        "--embedding_timeout",
        type=int,
        default=600,  # 10 minutes
        help="Embedding cache timeout in seconds",
    )
    parser.add_argument(
        "--max_num_clients",
        type=int,
        default=50,
        help="Maximum number of clients to cache embeddings for",
    )
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    loop.create_task(register_service(args=args))
    loop.run_forever()
