import os
from logging import getLogger
import uuid
from typing import Union
import json

import numpy as np
import requests
import torch
from imjoy_rpc.hypha import login, connect_to_server
from kaibu_utils import mask_to_features
from segment_anything import sam_model_registry, SamPredictor

logger = getLogger(__name__)
logger.setLevel("INFO")

MODELS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_b_lm": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/diplomatic-bug/1/files/vit_b.pt",
    "vit_b_em_organelles": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/noisy-ox/1/files/vit_b.pt",
}

STORAGE = {}


def client_id() -> str:
    client_id = str(uuid.uuid4())
    STORAGE[client_id] = {}
    logger.info(f"Generated user ID: {client_id}")
    return client_id


def remove_client_id(client_id: str) -> bool:
    assert client_id in STORAGE, f"User {client_id} not found"
    del STORAGE[client_id]


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


def compute_embedding(client_id: str, model_name: str, image: np.ndarray) -> None:
    assert client_id in STORAGE, f"User {client_id} not found"
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
    STORAGE[client_id] = predictor_dict
    logger.info(f"Caching embedding for user {client_id}")


def reset_embedding(client_id: str) -> bool:
    assert client_id in STORAGE, f"User {client_id} not found"
    STORAGE[client_id] = {}


def segment(
    client_id: str,
    point_coordinates: Union[list, np.ndarray],
    point_labels: Union[list, np.ndarray],
) -> list:
    assert client_id in STORAGE, f"User {client_id} not found"
    logger.info(f"Segmenting with embedding from user {client_id}...")
    # Load the model with the pre-computed embedding
    sam = _load_model(STORAGE[client_id].get("model_name"))
    predictor = SamPredictor(sam)
    for key, value in STORAGE[client_id].items():
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


async def start_server():
    """
    Start the SAM annotation server.

    When multiple people open the link, they can join a common workspace as an ImJoy client
    """

    config = os.path.join(os.path.dirname(__file__), "service_config.json")
    with open(config, "r") as f:
        server_url = json.load(f).get("server_url", "https://ai.imjoy.io")

    server = await connect_to_server({"server_url": server_url})

    svc = await server.register_service(
        {
            "name": "Interactive Segmentation",
            "id": "bioimageio-colab-model",
            "config": {"visibility": "public", "run_in_executor": True},
            # Exposed functions:
            # get a user id
            # returns a unique user id
            "client_id": client_id,
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
            "remove_client_id": remove_client_id,
        }
    )
    sid = svc["id"]

    # Save server_url and sid in model_server.json
    with open(config, "w") as f:
        f.write(json.dumps({"server_url": server_url, "sid": sid}))

    logger.info(f"Registered service with ID: {sid}")

    # print("Test the server on the following link:")
    # print(
    #     f"{server_url}/{server.config['workspace']}/services/{sid.split(':')[1]}/client_id"
    # )


if __name__ == "__main__":
    import asyncio

    logger.setLevel("DEBUG")

    loop = asyncio.get_event_loop()
    loop.create_task(start_server())

    loop.run_forever()

    # from imjoy_rpc.hypha.sync import connect_to_server
    # server = connect_to_server({"server_url": "https://ai.imjoy.io"})
    # sid = "..."
    # biocolab = server.getService(sid)
    # id = biocolab.client_id()
    # biocolab.compute_embedding(id, "vit_b", np.random.rand(256, 256))
    # features = biocolab.segment(id, [[128, 128]], [1])
    # print(features)
