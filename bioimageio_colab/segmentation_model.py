import argparse
import os
from logging import getLogger
from typing import Union

from dotenv import find_dotenv, load_dotenv
import numpy as np
import requests
import torch
from hypha_rpc import connect_to_server
from kaibu_utils import mask_to_features
from segment_anything import SamPredictor, sam_model_registry


ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

WORKSPACE_TOKEN = os.environ.get("WORKSPACE_TOKEN")
MODELS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_b_lm": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/diplomatic-bug/1/files/vit_b.pt",
    "vit_b_em_organelles": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/noisy-ox/1/files/vit_b.pt",
}
STORAGE = {}

if not WORKSPACE_TOKEN:
    raise ValueError("Workspace token is required to connect to the Hypha server.")

logger = getLogger(__name__)
logger.setLevel("INFO")


def _get_sam_model(model_name: str) -> torch.nn.Module:
    """
    Get the model from SAM / micro_sam for the given name.
    """
    model_url = MODELS[model_name]
    checkpoint_path = f"{model_name}.pt"

    if not os.path.exists(checkpoint_path):
        logger.info(f"Downloading model from {model_url} to {checkpoint_path}...")
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(checkpoint_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

    logger.info(f"Loading model {model_name} from {checkpoint_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = model_name[:5]
    sam = sam_model_registry[model_type]()
    ckpt = torch.load(checkpoint_path, map_location=device)
    sam.load_state_dict(ckpt)
    return sam


def _load_model(model_name: str) -> torch.nn.Module:
    if model_name not in MODELS:
        raise ValueError(
            f"Model {model_name} not found. Available models: {MODELS.keys()}"
        )
    model_url = MODELS[model_name]
    if model_url not in STORAGE:
        logger.info(f"Caching model {model_name} with ID '{model_url}'...")
        sam = _get_sam_model(model_name)
        STORAGE[model_url] = sam
        return sam
    else:
        logger.info(f"Loading model {model_name} with ID '{model_url}' from cache...")
        return STORAGE[model_url]


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


def compute_embedding(
    model_name: str, image: np.ndarray, context: dict = None
) -> bool:
    user_id = context["user"].get("id")
    if not user_id:
        logger.info("User ID not found in context.")
        return False
    sam = _load_model(model_name)
    logger.info(f"User {user_id} - computing embedding of model {model_name}...")
    predictor = SamPredictor(sam)
    predictor.set_image(_to_image(image))
    # Save computed predictor values
    logger.info(f"User {user_id} - caching embedding of model {model_name}...")
    predictor_dict = {
        "model_name": model_name,
        "original_size": predictor.original_size,
        "input_size": predictor.input_size,
        "features": predictor.features,  # embedding
        "is_image_set": predictor.is_image_set,
    }
    STORAGE[user_id] = predictor_dict
    return True


def reset_embedding(context: dict = None) -> bool:
    user_id = context["user"].get("id")
    if user_id not in STORAGE:
        logger.info(f"User {user_id} not found in storage.")
        return False
    else:
        logger.info(f"User {user_id} - resetting embedding...")
        STORAGE[user_id].clear()
        return True

def segment(
    point_coordinates: Union[list, np.ndarray],
    point_labels: Union[list, np.ndarray],
    context: dict = None,
) -> list:
    user_id = context["user"].get("id")
    if user_id not in STORAGE:
        logger.info(f"User {user_id} not found in storage.")
        return []

    logger.info(f"User {user_id} - segmenting with model {STORAGE[user_id].get('model_name')}...")
    # Load the model with the pre-computed embedding
    sam = _load_model(STORAGE[user_id].get("model_name"))
    predictor = SamPredictor(sam)
    for key, value in STORAGE[user_id].items():
        if key != "model_name":
            setattr(predictor, key, value)
    # Run the segmentation
    logger.debug(f"User {user_id} - point coordinates: {point_coordinates}, {point_labels}")
    if isinstance(point_coordinates, list):
        point_coordinates = np.array(point_coordinates, dtype=np.float32)
    if isinstance(point_labels, list):
        point_labels = np.array(point_labels, dtype=np.float32)
    mask, scores, logits = predictor.predict(
        point_coords=point_coordinates[:, ::-1],  # SAM has reversed XY conventions
        point_labels=point_labels,
        multimask_output=False,
    )
    logger.debug(f"User {user_id} - predicted mask of shape {mask.shape}")
    features = mask_to_features(mask[0])
    return features


def remove_user_id(context: dict = None) -> bool:
    user_id = context["user"].get("id")
    if user_id not in STORAGE:
        logger.info(f"User {user_id} not found in storage.")
        return False
    else:
        logger.info(f"User {user_id} - removing user from storage...")
        del STORAGE[user_id]
        return True


async def register_service(args: dict) -> None:
    """
    Register the SAM annotation service on the BioImageIO Colab workspace.
    """
    # Wait until the client ID is available
    test_client = await connect_to_server(
        {
            "server_url": args.server_url,
            "workspace": args.workspace_name,
            "token": WORKSPACE_TOKEN,
        }
    )
    colab_client_id = f"{args.workspace_name}/{args.client_id}"
    while colab_client_id not in await test_client.list_clients():
        await asyncio.sleep(1)

    # Connect to the workspace
    colab_client = await connect_to_server(
        {
            "server_url": args.server_url,
            "workspace": args.workspace_name,
            "client_id": args.client_id,
            "name": "Model Server",
            "token": WORKSPACE_TOKEN,
        }
    )

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
            # compute the image embeddings:
            # pass the model-name and the image to compute the embeddings on
            # calls load_model internally
            # returns True if the embeddings were computed successfully
            "compute_embedding": compute_embedding,
            # run interactive segmentation
            # pass the point coordinates and labels
            # returns the predicted mask encoded as geo json
            "segment": segment,
            # reset the embedding for the user
            # returns True if the embedding was removed successfully
            "reset_embedding": reset_embedding,
            # remove the user id from the storage
            # returns True if the user was removed successfully
            "remove_user_id": remove_user_id,  # TODO: add a timeout to remove a user after a certain time
        },
        overwrite=True,
    )
    sid = service_info["id"]
    assert sid == f"{args.workspace_name}/{args.client_id}:{args.service_id}"
    logger.info(f"Registered service with ID: {sid}")


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
        default="model-server",
        help="Client ID for registering the service",
    )
    parser.add_argument(
        "--service_id",
        default="interactive-segmentation",
        help="Service ID for registering the service",
    )
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    loop.create_task(register_service(args=args))
    loop.run_forever()
