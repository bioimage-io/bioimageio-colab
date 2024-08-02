import argparse
import os
from logging import getLogger
from typing import Union

import numpy as np
import requests
import torch
from hypha_rpc.hypha import connect_to_server
from kaibu_utils import mask_to_features
from segment_anything import SamPredictor, sam_model_registry

logger = getLogger(__name__)
logger.setLevel("INFO")

SERVER_URL = "https://hypha.aicell.io"
MODELS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_b_lm": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/diplomatic-bug/1/files/vit_b.pt",
    "vit_b_em_organelles": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/noisy-ox/1/files/vit_b.pt",
}
STORAGE = {}


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
    logger.info(f"Loaded model {model_name} from {checkpoint_path}")
    return sam


def _load_model(model_name: str) -> torch.nn.Module:
    if model_name not in MODELS:
        raise ValueError(
            f"Model {model_name} not found. Available models: {MODELS.keys()}"
        )
    model_url = MODELS[model_name]
    if model_url not in STORAGE:
        sam = _get_sam_model(model_name)
        STORAGE[model_url] = sam
        logger.info(f"Caching model {model_name} with ID '{model_url}'")
        return sam
    else:
        logger.info(f"Loading model {model_name} with ID '{model_url}' from cache")
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
    user_id: str, model_name: str, image: np.ndarray, context: dict = None
) -> None:
    if user_id not in STORAGE:
        logger.info(f"User {user_id} not found in storage.")
        return
    sam = _load_model(model_name)
    logger.info(f"Computing embeddings for model {model_name}...")
    predictor = SamPredictor(sam)
    predictor.set_image(_to_image(image))
    # Save computed predictor values
    predictor_dict = {
        "model_name": model_name,
        "original_size": predictor.original_size,
        "input_size": predictor.input_size,
        "features": predictor.features,  # embedding
        "is_image_set": predictor.is_image_set,
    }
    STORAGE[user_id] = predictor_dict
    logger.info(f"Caching embedding for user {user_id}")


def reset_embedding(user_id: str, context: dict = None) -> bool:
    if user_id not in STORAGE:
        logger.info(f"User {user_id} not found in storage.")
    else:
        STORAGE[user_id].clear()


def segment(
    user_id: str,
    point_coordinates: Union[list, np.ndarray],
    point_labels: Union[list, np.ndarray],
    context: dict = None,
) -> list:
    if user_id not in STORAGE:
        logger.info(f"User {user_id} not found in storage.")
        return

    logger.info(f"Segmenting with embedding from user {user_id}...")
    # Load the model with the pre-computed embedding
    sam = _load_model(STORAGE[user_id].get("model_name"))
    predictor = SamPredictor(sam)
    for key, value in STORAGE[user_id].items():
        if key != "model_name":
            setattr(predictor, key, value)
    # Run the segmentation
    logger.debug(f"Point coordinates: {point_coordinates}, {point_labels}")
    if isinstance(point_coordinates, list):
        point_coordinates = np.array(point_coordinates, dtype=np.float32)
    if isinstance(point_labels, list):
        point_labels = np.array(point_labels, dtype=np.float32)
    mask, scores, logits = predictor.predict(
        point_coords=point_coordinates[:, ::-1],  # SAM has reversed XY conventions
        point_labels=point_labels,
        multimask_output=False,
    )
    logger.debug(f"Predicted mask of shape {mask.shape}")
    features = mask_to_features(mask[0])
    return features


def remove_user_id(user_id: str, context: dict = None) -> bool:
    if user_id not in STORAGE:
        logger.info(f"User {user_id} not found in storage.")
    else:
        del STORAGE[user_id]


async def register_service(token: str, client_id: str, service_id: str):
    """
    Register the SAM annotation service on the BioImageIO Colab workspace.
    """
    if isinstance(token, str) and len(token) == 0:
        raise ValueError("Token is required to connect to the Hypha server.")

    if isinstance(client_id, str) and len(client_id) == 0:
        raise ValueError("Client ID is required to register the service.")

    if isinstance(service_id, str) and len(service_id) == 0:
        raise ValueError("Service ID is required to register the service.")

    # Connect to the Hypha server
    workspace_owner = await connect_to_server(
        {"server_url": SERVER_URL, "token": token}
    )

    # Create the bioimageio-colab workspace
    await workspace_owner.create_workspace(
        {
            "name": "bioimageio-colab",
            "description": "The BioImageIO Colab workspace for serving interactive segmentation models.",
            "owners": [],  # user ID of workspace owner is added automatically
            "allow_list": [],
            "deny_list": [],
            "visibility": "public",  # public/protected
            "persistent": False,  # can not be persistent for anonymous users
        },
        overwrite=True,  # overwrite if the workspace already exists
    )

    # Connect to the new workspace
    colab_client = await connect_to_server(
        {
            "server_url": SERVER_URL,
            "workspace": "bioimageio-colab",
            "client_id": client_id,
            "name": "Model Server",
            "token": token,
        }
    )

    # Register a new service
    service_info = await colab_client.register_service(
        {
            "name": "Interactive Segmentation",
            "id": service_id,
            "config": {
                "visibility": "public",
                "require_context": True,  # TODO: only allow the service to be called by logged-in users
                "run_in_executor": True,
            },
            # Exposed functions:
            # compute the image embeddings:
            # pass the client-id, model-name and the image to compute the embeddings on
            # calls load_model internally
            "compute_embedding": compute_embedding,
            # run interactive segmentation based on prompts
            # pass the client-id the point coordinates and labels
            # returns the predicted mask encoded as geo json
            "segment": segment,
            # remove the embedding
            # pass the client-id for which the embedding should be removed
            # returns True if the embedding was removed successfully
            "reset_embedding": reset_embedding,
            # remove the client id
            # pass the client-id to remove
            "remove_user_id": remove_user_id,  # TODO: add a timeout to remove the client id
        },
        overwrite=True,
    )
    sid = service_info["id"]
    assert sid == "bioimageio-colab/model-server:interactive-segmentation"
    logger.info(f"Registered service with ID: {sid}")

    # Test if the service can be retrieved from another workspace
    assert await workspace_owner.get_service(sid)


if __name__ == "__main__":
    import asyncio

    parser = argparse.ArgumentParser(
        description="Register SAM annotation service on BioImageIO Colab workspace."
    )
    parser.add_argument(
        "--token", required=True, help="Token for connecting to the Hypha server"
    )
    parser.add_argument(
        "--client_id",
        required=False,
        help="Client ID for registering the service",
        default="model-server",
    )
    parser.add_argument(
        "--service_id",
        required=False,
        help="Service ID for registering the service",
        default="interactive-segmentation",
    )
    args = parser.parse_args()

    logger.setLevel("DEBUG")

    loop = asyncio.get_event_loop()
    loop.create_task(
        register_service(
            token=args.token, client_id=args.client_id, service_id=args.service_id
        )
    )
    loop.run_forever()

    # from hypha_rpc.hypha.sync import connect_to_server
    # import numpy as np
    # server = connect_to_server({"server_url": "https://hypha.aicell.io"})
    # biocolab = server.getService(
    #     "bioimageio-colab/model-server:interactive-segmentation"
    # )
    # info = server.getConnectionInfo()
    # id = info.user_info.id
    # biocolab.compute_embedding(id, "vit_b", np.random.rand(256, 256))
    # features = biocolab.segment(id, [[128, 128]], [1])
    # print(features)
    # biocolab.remove_user_id(id)
